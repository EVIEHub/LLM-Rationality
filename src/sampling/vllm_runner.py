"""vLLM-based sampler implementing the project's hard sampling rules.

Wraps :class:`vllm.LLM` with:

- ``n=K`` sampling per prompt — never a Python loop calling ``generate``
  $K$ times (AGENT.md §3.1, §8.4).
- Mandatory explicit ``seed=`` on every call (AGENT.md §3.1, §3.5: no
  default randomness).
- AGENT.md §3.1 defaults (``temperature=1.0``, ``top_p=1.0``,
  ``top_k=-1``, no truncation) when no :class:`SamplingConfig` is passed.
- Clean VRAM release via :meth:`VllmRunner.close` — required between
  models per AGENT.md §8.6.
- Refusal of greedy decoding with ``K>1`` (AGENT.md §8.1: samples are
  identical so the extra budget is wasted).

vLLM and torch are imported lazily inside the methods that need them so
that mocking-based unit tests can patch ``vllm.LLM`` and ``vllm.SamplingParams``
without paying the import-time cost of the real library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters per AGENT.md §3.1.

    Defaults are the H1/H2 baseline. H3 is the experiment that
    explicitly varies these; do not override them anywhere else.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 2048


class VllmRunner:
    """Thin wrapper around :class:`vllm.LLM` enforcing project sampling rules.

    Args:
        hf_id: HuggingFace model id (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
        **vllm_kwargs: forwarded to :class:`vllm.LLM` (e.g.
            ``gpu_memory_utilization``, ``enforce_eager``,
            ``tensor_parallel_size``).
    """

    def __init__(self, hf_id: str, **vllm_kwargs: Any) -> None:
        import os

        from vllm import LLM

        # Multi-GPU sharding for large models (e.g. 70B on 4 GPUs). Read from
        # env vars (default 1) unless the caller passes them explicitly, so
        # panels can opt into parallelism without editing every call site.
        #   RG_TP — tensor parallel (per-layer all-reduce; needs fast GPU P2P)
        #   RG_PP — pipeline parallel (activations passed between stages; far
        #           less inter-GPU traffic — preferred on PCIe boxes without
        #           NVLink/P2P, where TP all-reduce is bottlenecked).
        vllm_kwargs.setdefault(
            "tensor_parallel_size", int(os.environ.get("RG_TP", "1"))
        )
        vllm_kwargs.setdefault(
            "pipeline_parallel_size", int(os.environ.get("RG_PP", "1"))
        )
        # RG_DISABLE_CUSTOM_AR=1 falls back from vLLM's custom all-reduce to
        # NCCL. The custom kernel can hit an illegal-memory-access on some
        # input shapes under TP (observed on 72B + HumanEval); NCCL is slower
        # but robust. Harmless when TP=1 (no all-reduce).
        if os.environ.get("RG_DISABLE_CUSTOM_AR", "0") == "1":
            vllm_kwargs.setdefault("disable_custom_all_reduce", True)
        # RG_MAX_NUM_SEQS caps the concurrent batch. Smaller batches keep the
        # TP all-reduce tensors small, dodging the CUDA illegal-memory-access
        # the all-reduce kernels hit on large batches (72B + HumanEval, K=64).
        _max_seqs = os.environ.get("RG_MAX_NUM_SEQS")
        if _max_seqs:
            vllm_kwargs.setdefault("max_num_seqs", int(_max_seqs))

        self.hf_id = hf_id
        self._llm: Any | None = LLM(model=hf_id, **vllm_kwargs)

    def sample(
        self,
        prompts: list[str],
        K: int,
        seed: int,
        config: SamplingConfig | None = None,
    ) -> list[list[str]]:
        """Sample ``K`` completions per prompt.

        Args:
            prompts: $M$ prompt strings, already chat-template-formatted
                if the model is instruct (vLLM does not apply chat
                templates itself).
            K: per-prompt sample budget; must be ``>= 1``.
            seed: RNG seed; mandatory.
            config: sampling parameters; AGENT.md §3.1 defaults if
                ``None``.

        Returns:
            A list of ``M`` elements, each a list of ``K`` strings (the
            completions for that prompt, in the order vLLM returned them).

        Raises:
            RuntimeError: ``close()`` has been called.
            ValueError: ``K < 1``, or greedy decoding (``temperature=0``)
                with ``K > 1``.
        """
        if self._llm is None:
            raise RuntimeError(
                "VllmRunner is closed; instantiate a new runner to sample again."
            )
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")

        from vllm import SamplingParams

        cfg = config or SamplingConfig()

        # AGENT.md §8.1: greedy with K>1 is wasteful — all K samples are
        # identical. Refuse so callers explicitly use K=1 for greedy.
        if cfg.temperature == 0.0 and K > 1:
            raise ValueError(
                "Greedy decoding (temperature=0.0) with K>1 is wasteful: "
                "all K samples will be identical. Use K=1 for greedy."
            )

        params = SamplingParams(
            n=K,
            seed=seed,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_tokens,
        )
        request_outputs = self._llm.generate(prompts, params)
        return [[out.text for out in req.outputs] for req in request_outputs]

    def close(self) -> None:
        """Release the underlying ``vllm.LLM`` and free GPU memory.

        Required between models in a multi-model run per AGENT.md §8.6
        ("vLLM does not always release VRAM cleanly"). Idempotent.
        """
        if self._llm is None:
            return
        import gc

        import torch

        del self._llm
        self._llm = None
        torch.cuda.empty_cache()
        gc.collect()

    def __enter__(self) -> "VllmRunner":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
