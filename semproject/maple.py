"""MAPLE's two selection steps and the embedding graph they share."""
import hashlib
import math
import os
from collections import deque

import networkx as nx
import numpy as np
import torch

EMBEDDING_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".emb")
EMBEDDER = "facebook/contriever-msmarco"

_encoder_cache = {}


def _load_encoder():
    if not _encoder_cache:
        from transformers import AutoModel, AutoTokenizer
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        _encoder_cache.update(
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDER),
            model=AutoModel.from_pretrained(EMBEDDER).to(device).eval(),
            device=device)
    return _encoder_cache


def embed(texts, batch_size=32, cache_tag=None):
    """Mean-pooled Contriever embeddings.

    Padding is masked out of the mean, so batching gives the same result as
    encoding everything at once.
    """
    if isinstance(texts, str):
        texts = [texts]

    cache_path = None
    if cache_tag:
        os.makedirs(EMBEDDING_CACHE, exist_ok=True)
        digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:24]
        cache_path = os.path.join(EMBEDDING_CACHE, f"{cache_tag}_{digest}.npy")
        if os.path.exists(cache_path):
            return torch.from_numpy(np.load(cache_path))

    encoder = _load_encoder()
    batches = []
    for start in range(0, len(texts), batch_size):
        tokenized = encoder["tokenizer"](
            texts[start:start + batch_size], padding=True, truncation=True,
            return_tensors="pt").to(encoder["device"])
        with torch.no_grad():
            hidden_states = encoder["model"](**tokenized)[0]
        attention_mask = tokenized["attention_mask"]
        masked = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        batches.append(pooled.float().cpu())

    embeddings = torch.cat(batches)
    if cache_path:
        np.save(cache_path, embeddings.numpy())
    return embeddings


def knn_graph(embeddings, degree=20):
    """Connect each sample to its `degree` nearest neighbours by inner product.

    Disconnected components are joined afterwards because the influence scores
    below require a path between every pair of nodes.
    """
    similarity = (embeddings @ embeddings.T).numpy()
    node_count = similarity.shape[0]

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    for node in range(node_count):
        ranked = np.argsort(np.ravel(similarity[node]))
        for neighbour in ranked[-degree - 1:-1]:
            if neighbour != node:
                graph.add_edge(node, int(neighbour))

    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        for current, following in zip(components, components[1:]):
            graph.add_edge(next(iter(current)), next(iter(following)))
    return graph


def adjacency_list(graph, node_count):
    neighbours = [[] for _ in range(node_count)]
    for left, right in graph.edges():
        neighbours[left].append(right)
        neighbours[right].append(left)
    return neighbours


def shortest_paths_from(neighbours, source, node_count):
    """Distance and exact shortest-path count from `source` to every node.

    Counts come from the recurrence paths(v) = sum of paths over v's
    predecessors, which avoids enumerating the paths themselves.
    """
    distance = [-1] * node_count
    path_count = [0] * node_count
    distance[source] = 0
    path_count[source] = 1

    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in neighbours[node]:
            if distance[neighbour] < 0:
                distance[neighbour] = distance[node] + 1
                path_count[neighbour] = path_count[node]
                queue.append(neighbour)
            elif distance[neighbour] == distance[node] + 1:
                path_count[neighbour] += path_count[node]
    return distance, path_count


def influence_scores(graph, labeled_indices):
    """Influence of every node on the labeled set, per the paper's Theorem 3.2.

        influence(u) = mean log(paths to labeled) - mean distance * log(avg degree)

    Computed once and reused by both the selection step and the influence
    signal, so reading it back in the gate costs no model calls.
    """
    node_count = graph.number_of_nodes()
    neighbours = adjacency_list(graph, node_count)
    log_average_degree = math.log(2 * graph.number_of_edges() / node_count)

    total_log_paths = np.zeros(node_count)
    total_distance = np.zeros(node_count)
    for labeled_node in labeled_indices:
        distance, path_count = shortest_paths_from(neighbours, int(labeled_node), node_count)
        for node in range(node_count):
            if distance[node] >= 0:
                total_distance[node] += distance[node]
                total_log_paths[node] += math.log(path_count[node]) if path_count[node] else 0.0

    labeled_count = len(labeled_indices)
    return total_log_paths / labeled_count - (total_distance / labeled_count) * log_average_degree


def select_for_labeling(graph, labeled_indices, count, scores=None):
    """The `count` unlabeled samples with the greatest influence on the labeled set."""
    influence = influence_scores(graph, labeled_indices) if scores is None else scores
    candidates = np.delete(np.arange(graph.number_of_nodes()), labeled_indices)
    return candidates[np.argsort(influence[candidates])[-count:]]


def adaptive_select(graph, pool_embeddings, query_embedding, demo_count,
                    alpha=0.75, links=10):
    """Keep the `alpha` fraction of the demo pool most influential for one query.

    The query is attached to its nearest demos, then scored by influence flowing
    back from it. Average degree is taken before attachment so the query's own
    edges do not shift the scale.
    """
    graph_with_query = graph.copy()
    query_node = graph_with_query.number_of_nodes()
    log_average_degree = math.log(
        2 * graph_with_query.number_of_edges() / query_node)

    similarity = np.ravel((query_embedding @ pool_embeddings.T).numpy())
    for neighbour in np.argsort(similarity)[-links:]:
        graph_with_query.add_edge(query_node, int(neighbour))

    distance, path_count = shortest_paths_from(
        adjacency_list(graph_with_query, query_node + 1), query_node, query_node + 1)

    influence = np.array([
        float(path_count[node]) - distance[node] * log_average_degree
        if distance[node] >= 0 else -np.inf
        for node in range(demo_count)])

    keep_count = max(1, int(round(alpha * demo_count)))
    return np.sort(np.argsort(influence)[-keep_count:])
