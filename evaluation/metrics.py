"""
Evaluation metrics and benchmarking suite for GenRec:
- Ranking Quality: MRR@K, HitRate@K, NDCG@K
- Latency & Serving Efficiency: Prefill-Only Scoring vs Autoregressive Generation
- Sample Efficiency Analysis (performance vs % of training data)
"""

import os
import time
import math
from typing import List, Dict, Any, Tuple

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
import torch.nn.functional as F
import numpy as np


def compute_mrr(ranks: List[int]) -> float:
    """Mean Reciprocal Rank."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def compute_hit_rate(ranks: List[int], k: int) -> float:
    """Hit Rate at K."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))


def compute_ndcg_at_k(ranks: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K for single positive target item."""
    if not ranks:
        return 0.0
    ndcg_list = []
    for r in ranks:
        if r <= k:
            # DCG = 1 / log2(r + 1), IDCG = 1 / log2(1 + 1) = 1.0
            ndcg_list.append(1.0 / math.log2(r + 1))
        else:
            ndcg_list.append(0.0)
    return float(np.mean(ndcg_list))


def evaluate_model_ranking(
    model: Any,
    test_instances: List[Any],
    verbalizer: Any,
    tokenizer: Any,
    device: str = "cpu",
    batch_size: int = 16
) -> Dict[str, float]:
    """Evaluates GenRec model across MRR, HR@1, HR@3, HR@5, and NDCG@5."""
    model.eval()
    all_ranks = []

    for i in range(0, len(test_instances), batch_size):
        batch = test_instances[i:i + batch_size]
        prompts = [verbalizer.verbalize_context(b.interactions) for b in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=192, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        cands = torch.tensor([b.candidate_item_ids for b in batch], dtype=torch.long, device=device)
        labels = [b.label_idx for b in batch]

        with torch.no_grad():
            logits = model(input_ids, attention_mask, cands)
            _, sorted_indices = torch.sort(logits, dim=-1, descending=True)

        for true_idx, ranked_order in zip(labels, sorted_indices.tolist()):
            rank = ranked_order.index(true_idx) + 1
            all_ranks.append(rank)

    return {
        "mrr": compute_mrr(all_ranks),
        "hr@1": compute_hit_rate(all_ranks, k=1),
        "hr@3": compute_hit_rate(all_ranks, k=3),
        "hr@5": compute_hit_rate(all_ranks, k=5),
        "ndcg@5": compute_ndcg_at_k(all_ranks, k=5),
        "num_evaluated": len(all_ranks)
    }


def benchmark_latency(
    model: Any,
    sample_prompt: str,
    candidate_ids: List[int],
    tokenizer: Any,
    catalog: Any,
    num_runs: int = 25,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Benchmarks Prefill-Only Scoring against Autoregressive Text Generation.
    Demonstrates the latency and catalog-validity advantages highlighted by Netflix.
    """
    model.eval()
    encoded = tokenizer([sample_prompt], return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    cands_tensor = torch.tensor([candidate_ids], dtype=torch.long, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids, attention_mask, cands_tensor)

    # 1. Measure Prefill-Only Scoring Latency
    prefill_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, attention_mask, cands_tensor)
        prefill_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 2. Measure Autoregressive Generation Latency
    autoreg_latencies = []
    generated_sample_texts = []
    for _ in range(min(5, num_runs)):
        t0 = time.perf_counter()
        text, _ = model.simulate_autoregressive_generation(input_ids, tokenizer, max_new_tokens=10)
        autoreg_latencies.append((time.perf_counter() - t0) * 1000.0)
        generated_sample_texts.append(text)

    prefill_mean = float(np.mean(prefill_latencies))
    prefill_p50 = float(np.median(prefill_latencies))
    prefill_p95 = float(np.percentile(prefill_latencies, 95))
    prefill_p99 = float(np.percentile(prefill_latencies, 99))

    autoreg_mean = float(np.mean(autoreg_latencies))
    autoreg_p50 = float(np.median(autoreg_latencies))

    speedup = autoreg_mean / max(0.1, prefill_mean)

    return {
        "prefill_only": {
            "mean_ms": round(prefill_mean, 2),
            "p50_ms": round(prefill_p50, 2),
            "p95_ms": round(prefill_p95, 2),
            "p99_ms": round(prefill_p99, 2),
            "catalog_guarantee": "100% Valid Catalog IDs",
            "hallucination_rate": "0.0%"
        },
        "autoregressive": {
            "mean_ms": round(autoreg_mean, 2),
            "p50_ms": round(autoreg_p50, 2),
            "sample_output": generated_sample_texts[0] if generated_sample_texts else "",
            "catalog_guarantee": "Unconstrained (Risk of generating uncataloged titles)"
        },
        "speedup_factor": round(speedup, 1)
    }
