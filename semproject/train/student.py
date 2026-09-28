"""Fine-tune a small model on labels, then measure it.

This is the product of the project: the big model writes labels, the gate
filters them, and whatever survives becomes training data for this student. The
student is what gets evaluated, so it is the only number that says whether
filtering was worth doing.

Loss is computed on the label tokens only. Training on the prompt tokens too
would spend most of the gradient teaching the model to recite questions.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import data

IGNORE = -100


def format_pair(task, example, label=None):
    """The (prompt, target) pair the student learns from."""
    if label is None:
        label = data.gold_answer_text(example) if task in data.BBH_TASKS \
            else _gold_label(task, example)
    prompt = data.embed_text(task, example, with_label=False)
    return prompt, str(label).strip()


def _gold_label(task, example):
    if task == "banking77":
        import process_bank77
        return process_bank77.all_labels[example["label"]]
    if task == "goemo":
        import process_goemotion
        return process_goemotion.all_labels[example["labels"][0]]
    if task == "fp":
        return example["label"].strip()
    if task == "xsum":
        return example["summary"]
    return (example.get("target") or "").strip()


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=320):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        prompt, target = self.pairs[index]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(" " + target, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + target_ids)[-self.max_length:]
        labels = ([IGNORE] * len(prompt_ids) + target_ids)[-self.max_length:]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * padding)
        labels.append(row["labels"] + [IGNORE] * padding)
        mask.append([1] * len(row["input_ids"]) + [0] * padding)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(mask))


def train_student(task, training_examples, model_id="Qwen/Qwen2.5-0.5B-Instruct",
                  epochs=3, batch_size=4, learning_rate=1e-4, lora_rank=8,
                  device=None, log=print):
    """LoRA fine-tune on (example, label) pairs. Returns (model, tokenizer, history)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))

    dataset = PairDataset(training_examples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    history = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for input_ids, labels, mask in loader:
            optimiser.zero_grad()
            output = model(input_ids=input_ids.to(device),
                           attention_mask=mask.to(device),
                           labels=labels.to(device))
            output.loss.backward()
            optimiser.step()
            losses.append(output.loss.detach().item())
        history.append(float(np.mean(losses)))
        log(f"    epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")

    model.eval()
    return model, tokenizer, history


@torch.no_grad()
def evaluate_student(model, tokenizer, task, test_examples, max_new_tokens=16, device=None):
    """Accuracy of the trained student, generating the label itself."""
    device = device or next(model.parameters()).device
    scores = []
    for example in test_examples:
        prompt = data.embed_text(task, example, with_label=False)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=320).to(device)
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        scores.append(data.score(task, text, example))
    return float(np.mean(scores)) if scores else 0.0
