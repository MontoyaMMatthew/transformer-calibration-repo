"""BF16, forced flash SDPA, answer-position-only scoring and memory accounting."""

import math
import subprocess
import threading
import time
import uuid

from src.artifacts import now


def hardware(config):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    p = torch.cuda.get_device_properties(0)
    if torch.version.cuda is None or torch.version.hip:
        raise RuntimeError("An NVIDIA CUDA build of PyTorch is required")
    if p.total_memory < config["minimum_vram_gib"] * 1024**3:
        raise RuntimeError(f"At least {config['minimum_vram_gib']} GiB usable VRAM required; got {p}")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native hardware BF16 support required")
    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("PyTorch must include the Flash SDPA backend")
    info = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return {"name": p.name, "total_memory_bytes": p.total_memory,
            "capability": list(torch.cuda.get_device_capability()),
            "nvidia_smi": info.stdout.strip(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda}


class Scorer:
    def __init__(self, model_path, tokenizer, config):
        import torch
        from transformers import AutoModelForCausalLM
        self.hardware = hardware(config)
        self.config, self.tokenizer = config, tokenizer
        started = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        self.model.eval()
        if any(p.dtype != torch.bfloat16 or p.device.type != "cuda" for p in self.model.parameters()):
            raise ValueError("All weights must be BF16 on GPU")
        if self.model.config.rope_scaling is not None or self.model.config._attn_implementation != "sdpa":
            raise ValueError("Native context and SDPA required")
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    def score(self, cases, prompts):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        encoded = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        if len({len(t) for t in encoded}) != 1:
            raise ValueError("Flash SDPA uses exact-length buckets; unequal inputs are never padded")
        for case, ids in zip(cases, encoded, strict=True):
            if len(ids) != case["input_tokens"]:
                raise ValueError("Prompt token count changed")
        tokens = torch.tensor(encoded, device="cuda:0")
        # All-ones 2D mask prevents Transformers' packed-sequence mask path, which
        # materializes a 4D mask incompatible with forced PyTorch flash SDPA.
        # There is no padding; every answer is at the final position.
        mask = torch.ones_like(tokens)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples, stopped = [], threading.Event()

        def sample():
            free, total = torch.cuda.mem_get_info(0)
            reserved = torch.cuda.memory_reserved(0)
            samples.append((total - free, max(0, total - free - reserved)))

        def monitor():
            while not stopped.wait(0.01):
                sample()

        sample()
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        started = time.perf_counter()
        try:
            with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = self.model(input_ids=tokens, attention_mask=mask,
                                    use_cache=False, logits_to_keep=1)
                if output.logits.shape[1] != 1:
                    raise ValueError("Only answer-position vocabulary logits may be materialized")
                scores = [output.logits[i, -1, c["label_token_ids"]].float()
                          for i, c in enumerate(cases)]
                log_probs = [torch.log_softmax(z, dim=-1) for z in scores]
                probs = [lp.exp() for lp in log_probs]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            sample()
        finally:
            stopped.set()
            watcher.join()
        reserved_peak = torch.cuda.max_memory_reserved()
        # Allocator peak plus maximum observed non-allocator use is a conservative bound.
        used_bound = max(max(s[0] for s in samples), reserved_peak + max(s[1] for s in samples))
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": reserved_peak,
                  "sampled_peak_device_used_bytes": max(s[0] for s in samples),
                  "conservative_peak_device_used_bytes": used_bound,
                  "total_memory_bytes": self.hardware["total_memory_bytes"],
                  "minimum_headroom_fraction": 1 - used_bound / self.hardware["total_memory_bytes"],
                  "device_memory_samples": len(samples), "device_sampling_seconds": 0.01}
        rows = []
        batch_id = uuid.uuid4().hex
        for case, z, lp, p in zip(cases, scores, log_probs, probs, strict=True):
            rows.append({**case, "status": "success", "completed_utc": now(),
                         "choice_logits": z.tolist(), "log_probabilities": lp.tolist(),
                         "probabilities": p.tolist(), "prediction": case["labels"][z.argmax().item()],
                         "batch_size": len(cases), "batch_forward_and_score_seconds": elapsed,
                         "batch_id": batch_id,
                         "batch_evaluation_ids": [c["evaluation_id"] for c in cases],
                         "amortized_forward_and_score_seconds": elapsed / len(cases), **memory})
            row = rows[-1]
            if (not all(math.isfinite(x) for key in ["choice_logits", "log_probabilities", "probabilities"]
                        for x in row[key]) or abs(sum(row["probabilities"]) - 1) > 1e-6):
                raise ValueError("Choice scoring is nonfinite or not normalized")
        return rows

    def attention_check(self, case, prompt):
        import torch
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            self.score([case], [prompt])
        operators = {e.key: e.count for e in profile.key_averages() if "scaled_dot_product" in e.key}
        if operators.get("aten::_scaled_dot_product_flash_attention", 0) != self.model.config.num_hidden_layers:
            raise RuntimeError(f"Flash attention not observed in every layer: {operators}")
        return {"status": "passed", "operators": operators,
                "layers": self.model.config.num_hidden_layers, "excluded_from_primary_timing": True}
