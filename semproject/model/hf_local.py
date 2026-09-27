"""Local transformers backend, the only one that returns true token logprobs."""
import torch

from .base import Model, Response

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class HFModel(Model):
    name = "hf"
    supports_logprobs = True

    def __init__(self, model_id=DEFAULT_MODEL, device=None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype or torch.float32).to(self.device).eval()

    def generate(self, prompt, temperature=0.0, max_tokens=64):
        """Returns the logprob the model assigned to each token it actually chose."""
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(chat_prompt, return_tensors="pt",
                                 truncation=True).to(self.device)

        with torch.no_grad():
            generated = self.language_model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = generated.sequences[0][prompt_length:]
        chosen_logprobs = [
            torch.log_softmax(step_scores[0].float(), dim=-1)[token].item()
            for step_scores, token in zip(generated.scores, new_tokens)]

        return Response(
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            logprobs=chosen_logprobs,
            meta={"backend": "hf", "model": self.name})
