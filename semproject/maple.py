"""MAPLE's two selection mechanisms, and the embedding/graph machinery they share.

Influence follows the paper's Theorem 3.2: a node's influence on a set is bounded
by the number of shortest paths between them and their shortest-path distance,

    influence(u, V) = mean_i log P_S(u, v_i) - mean_i L_S(u, v_i) * log(avg_degree)

Path counts come from one breadth-first sweep per source using the standard
recurrence sigma(v) = sum of sigma over v's predecessors, which is exact and
avoids enumerating every path.
"""
import hashlib
import math
import os
from collections import deque

import networkx as nx
import numpy as np
import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".emb")
EMBEDDER = "facebook/contriever-msmarco"
_ENC = {}


# ------------------------------------------------------------------ embeddings
def _encoder():
    if not _ENC:
        from transformers import AutoModel, AutoTokenizer
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        _ENC.update(tok=AutoTokenizer.from_pretrained(EMBEDDER),
                    mdl=AutoModel.from_pretrained(EMBEDDER).to(dev).eval(), dev=dev)
    return _ENC


def embed(texts, batch_size=32, cache_tag=None):
    """Mean-pooled Contriever embeddings. Padding is masked out, so batching is exact."""
    if isinstance(texts, str):
        texts = [texts]
    path = None
    if cache_tag:
        os.makedirs(CACHE, exist_ok=True)
        h = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:24]
        path = os.path.join(CACHE, f"{cache_tag}_{h}.npy")
        if os.path.exists(path):
            return torch.from_numpy(np.load(path))
    e, out = _encoder(), []
    for i in range(0, len(texts), batch_size):
        inp = e["tok"](texts[i:i + batch_size], padding=True, truncation=True,
                       return_tensors="pt").to(e["dev"])
        with torch.no_grad():
            h_ = e["mdl"](**inp)[0]
        m = inp["attention_mask"]
        out.append((h_.masked_fill(~m[..., None].bool(), 0.).sum(1)
                    / m.sum(1)[..., None]).float().cpu())
    emb = torch.cat(out)
    if path:
        np.save(path, emb.numpy())
    return emb


# ----------------------------------------------------------------- graph + influence
def knn_graph(emb, degree=20):
    """Top-`degree` inner-product neighbour graph, excluding each node's self-match."""
    sim = (emb @ emb.T).numpy()
    G = nx.Graph()
    G.add_nodes_from(range(sim.shape[0]))
    for i in range(sim.shape[0]):
        for j in np.argsort(np.ravel(sim[i]))[-degree - 1:-1]:
            if j != i:
                G.add_edge(i, int(j))
    if not nx.is_connected(G):  # every shortest path must exist
        comps = list(nx.connected_components(G))
        for a, b in zip(comps, comps[1:]):
            G.add_edge(next(iter(a)), next(iter(b)))
    return G


def _bfs(adj, src, n):
    """Distances and exact shortest-path counts from `src`."""
    dist, sig = [-1] * n, [0] * n
    dist[src], sig[src] = 0, 1
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v], sig[v] = dist[u] + 1, sig[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                sig[v] += sig[u]
    return dist, sig


def _adj(G, n):
    a = [[] for _ in range(n)]
    for u, v in G.edges():
        a[u].append(v)
        a[v].append(u)
    return a


def influence_scores(G, labeled_idx):
    """Influence of every node on the labeled set. Shared by step 2 and the gate.

    Computing this once means the influence signal costs no model calls: the
    graph already exists by the time pseudo-labeling starts.
    """
    n = G.number_of_nodes()
    adj, log_d = _adj(G, n), math.log(2 * G.number_of_edges() / n)
    paths, lens = np.zeros(n), np.zeros(n)
    for t in labeled_idx:                       # undirected, so one BFS per labeled node
        dist, sig = _bfs(adj, int(t), n)
        for v in range(n):
            if dist[v] >= 0:
                lens[v] += dist[v]
                paths[v] += math.log(sig[v]) if sig[v] else 0.0
    return paths / len(labeled_idx) - (lens / len(labeled_idx)) * log_d


def select_for_labeling(G, labeled_idx, k, scores=None):
    """Step 2: the k unlabeled nodes with greatest influence on the labeled set."""
    inf = influence_scores(G, labeled_idx) if scores is None else scores
    rest = np.delete(np.arange(G.number_of_nodes()), labeled_idx)
    return rest[np.argsort(inf[rest])[-k:]]


def adaptive_select(G, pool_emb, query_emb, n_demos, alpha=0.75, link=10):
    """Step 6: keep the top `alpha` fraction of the pool by influence on this query."""
    g = G.copy()
    q = g.number_of_nodes()
    log_d = math.log(2 * g.number_of_edges() / q)
    for j in np.argsort(np.ravel((query_emb @ pool_emb.T).numpy()))[-link:]:
        g.add_edge(q, int(j))
    dist, sig = _bfs(_adj(g, q + 1), q, q + 1)
    inf = np.array([float(sig[v]) - dist[v] * log_d if dist[v] >= 0 else -np.inf
                    for v in range(n_demos)])
    keep = max(1, int(round(alpha * n_demos)))
    return np.sort(np.argsort(inf)[-keep:])
