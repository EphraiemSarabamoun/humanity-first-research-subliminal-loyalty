"""Generation for the leg, behind one interface with two engines.

    vllm  the cloud leg: one vLLM model for the run, LoRA adapters attached per request (enable_lora)
    hf    the local dry run on CPU with a tiny model: transformers generate, adapters through peft

Both take chats (lists of {role, content} messages) and apply the model's own chat template, so an absent system
message gets the template's default, as it does in upstream's runs.
"""
import os


class VLLMEngine:
    def __init__(self, model, max_lora_rank=8, seed=0):
        from vllm import LLM
        self.llm = LLM(model=model, enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1, seed=seed,
                       gpu_memory_utilization=float(os.environ.get("SUSURRUS_GPU_UTIL", "0.85")),
                       max_model_len=2048)
        self._ids = {}

    def _lora(self, adapter):
        if not adapter:
            return None
        from vllm.lora.request import LoRARequest
        self._ids.setdefault(adapter, len(self._ids) + 1)
        return LoRARequest(os.path.basename(adapter.rstrip("/")), self._ids[adapter], adapter)

    def chat(self, chats, temperature, max_tokens, n=1, adapter=None, seed=None):
        """[[text] * n] per chat. temperature 0 is greedy (n must be 1)."""
        from vllm import SamplingParams
        params = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, n=n, seed=seed)
        outs = self.llm.chat(chats, params, lora_request=self._lora(adapter), use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]


class HFEngine:
    """CPU dry run: correct, slow, for a tiny model and a handful of items."""

    def __init__(self, model, seed=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.name, self.tok = model, AutoTokenizer.from_pretrained(model)
        self.base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        self.loaded = {None: self.base}

    def _model(self, adapter):
        if adapter not in self.loaded:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
            base = AutoModelForCausalLM.from_pretrained(self.name)
            self.loaded[adapter] = PeftModel.from_pretrained(base, adapter)
        return self.loaded[adapter]

    def chat(self, chats, temperature, max_tokens, n=1, adapter=None, seed=None):
        import torch
        model, out = self._model(adapter), []
        for chat in chats:
            ids = self.tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
            texts = []
            for _ in range(n):
                g = model.generate(ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                   temperature=temperature if temperature > 0 else None, top_p=1.0,
                                   pad_token_id=self.tok.eos_token_id)
                texts.append(self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
            out.append(texts)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return out


def make(kind, model, seed=0):
    return VLLMEngine(model, seed=seed) if kind == "vllm" else HFEngine(model, seed=seed)
