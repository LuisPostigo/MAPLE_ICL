"""Local transformers backend. The only one that returns true token logprobs."""
import torch

from .base import Model, Response

DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"


class HFModel(Model):
    name = "hf"
    supports_logprobs = True

    def __init__(self, model_id=DEFAULT, device=None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype or torch.float32).to(self.device).eval()

    def generate(self, prompt, temperature=0.0, max_tokens=64):
        chat = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.lm.generate(
                **enc, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        new = out.sequences[0][enc["input_ids"].shape[1]:]
        # Logprob actually assigned to each token the model chose.
        lps = [torch.log_softmax(s[0].float(), -1)[t].item()
               for s, t in zip(out.scores, new)]
        return Response(text=self.tok.decode(new, skip_special_tokens=True).strip(),
                        logprobs=lps, meta={"backend": "hf", "model": self.name})
