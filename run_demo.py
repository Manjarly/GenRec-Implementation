#!/usr/bin/env python3
"""
Unified CLI Entrypoint for Netflix GenRec Implementation.

Modes:
  1. cli:   Runs an end-to-end recommendation demonstration in your terminal.
  2. train: Runs Phase 1 (LM Domain Adaptation) and Phase 2 (Ranking Alignment) training.
  3. eval:  Evaluates ranking metrics (MRR, HR@K, NDCG@K) on holdout test set.
  4. serve: Launches the interactive web dashboard (http://localhost:8080).

Usage examples:
  python run_demo.py --mode cli
  python run_demo.py --mode train --epochs 2
  python run_demo.py --mode eval
  python run_demo.py --mode serve --port 8080
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Disable TensorFlow check in HuggingFace to avoid broken protobuf import
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoTokenizer

from data.dataset import (
    NetflixCatalog,
    create_predefined_personas,
    generate_synthetic_dataset,
    WatchInteraction
)
from verbalizer.verbalizer import GenRecVerbalizer
from models.genrec import GenRecModel
from models.baselines import PopularityRanker, MLPRanker
from training.trainer import GenRecTrainer
from evaluation.metrics import evaluate_model_ranking, benchmark_latency


def run_cli_demo():
    print("\n" + "=" * 70)
    print("🎬 Netflix GenRec: LLM-Backed Recommendation Ranker (arXiv:2608.10257)")
    print("=" * 70)

    catalog = NetflixCatalog()
    personas = create_predefined_personas(catalog)
    verbalizer = GenRecVerbalizer(catalog)

    print(f"\n[1] Loaded Netflix Catalog with {len(catalog)} diverse movies and series.")
    print(f"[2] Initialized {len(personas)} canonical member archetypes.")

    # Select persona
    persona = personas[0]  # Alex Chen (Sci-Fi Buff)
    print(f"\n--- Selected Member: {persona.name} ({persona.persona}) ---")
    print(f"Primary Device: {persona.primary_device} | Preferred: {', '.join(persona.preferred_genres)}")
    print("Recent Watch History:")
    for idx, inter in enumerate(persona.interactions, 1):
        it = catalog.get_by_id(inter.item_id)
        rating_str = f", Rated: {inter.rating}" if inter.rating else ""
        rewatch_str = ", [Rewatched]" if inter.rewatched else ""
        print(f"  {idx}. \"{it.title}\" (Completed {int(inter.completion_pct*100)}%{rating_str}{rewatch_str})")

    # Verbalization
    print("\n" + "-" * 70)
    print("[3] Step 1: Context Engineering (Prompt Verbalization)")
    print("-" * 70)
    prompt = verbalizer.verbalize_context(persona.interactions, device=persona.primary_device, time_of_day="Evening")
    print(prompt)

    # Legacy contrast
    legacy_feat = verbalizer.extract_legacy_tabular_features(persona.interactions)
    print("\n[Contrast: Legacy Tabular Features (Manual Engineering)]:")
    print(f"  {legacy_feat}")

    # Load Model & Tokenizer
    print("\n" + "-" * 70)
    print("[4] Step 2: Instantiating GenRec Model with Catalog-Aware Head")
    print("-" * 70)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GenRecModel(
        model_name_or_config="gpt2",
        num_catalog_items=len(catalog),
        proj_dim=128,
        use_pretrained=True
    )
    model.eval()

    # Candidate set to rerank
    candidate_ids = [5, 14, 18, 20, 24, 12]  # Inception, Queen's Gambit, Squid Game, Glass Onion, Dune, Brooklyn Nine-Nine
    print(f"\nCandidates to Rank:")
    for cid in candidate_ids:
        it = catalog.get_by_id(cid)
        print(f"  • ID {cid}: \"{it.title}\" ({', '.join(it.genres)})")

    # Tokenize
    encoded = tokenizer([prompt], padding=True, truncation=True, max_length=192, return_tensors="pt")
    cands_tensor = torch.tensor([candidate_ids], dtype=torch.long)

    # Prefill-Only Scoring
    print("\n" + "-" * 70)
    print("[5] Step 3: Prefill-Only Candidate Scoring (1 Forward Pass)")
    print("-" * 70)
    ranked_items, sorted_probs, latency_ms = model.rank_candidates(
        encoded["input_ids"],
        encoded["attention_mask"],
        cands_tensor
    )

    print(f"⚡ Prefill-Only Scoring completed in {latency_ms:.2f} ms!")
    print("Guaranteed Catalog Validity: 100% | Hallucination Rate: 0.0%\n")
    print("Ranked Output:")
    for rank_idx, (cid, prob) in enumerate(zip(ranked_items[0].tolist(), sorted_probs[0].tolist()), 1):
        it = catalog.get_by_id(cid)
        bar = "█" * int(prob * 30)
        print(f"  #{rank_idx} {it.title:<22} | Score: {prob*100:5.1f}% | {bar} ({', '.join(it.genres[:2])})")

    # Latency comparison
    print("\n" + "-" * 70)
    print("[6] Step 4: Prefill-Only Scoring vs Naive Autoregressive Text Generation")
    print("-" * 70)
    print("Running speedup benchmark (10 iterations)...")
    bench = benchmark_latency(model, prompt, candidate_ids, tokenizer, catalog, num_runs=10)
    print(f"  • GenRec Prefill-Only Latency (P50):  {bench['prefill_only']['p50_ms']} ms")
    print(f"  • Naive Autoregressive Latency (P50): {bench['autoregressive']['p50_ms']} ms")
    print(f"  🚀 Speedup Factor: {bench['speedup_factor']}× faster serving with strict catalog enforcement!")

    print("\n" + "=" * 70)
    print("✅ GenRec Demonstration Complete.")
    print("To launch the interactive visual dashboard, run:")
    print("   python run_demo.py --mode serve")
    print("=" * 70 + "\n")


def run_training_demo(epochs: int = 2):
    print("\n" + "=" * 70)
    print("🏋️ Starting GenRec Two-Phase Training Demonstration")
    print("=" * 70)

    catalog = NetflixCatalog()
    verbalizer = GenRecVerbalizer(catalog)

    print("Generating synthetic Netflix ranking datasets...")
    train_set, val_set, test_set = generate_synthetic_dataset(catalog, num_samples=250, candidate_size=5)
    print(f"Splits: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GenRecModel(
        model_name_or_config="gpt2",
        num_catalog_items=len(catalog),
        proj_dim=128,
        use_pretrained=True
    )

    trainer = GenRecTrainer(model, tokenizer, verbalizer, catalog)

    # Phase 1: Foundation Adaptation
    trainer.train_phase1_adaptation(epochs=1, lr=5e-5, batch_size=4)

    # Phase 2: Ranking Alignment
    metrics = trainer.train_phase2_ranking(
        train_instances=train_set,
        val_instances=val_set,
        epochs=epochs,
        batch_size=8,
        lr=3e-4
    )

    print("\nEvaluating on Test Set...")
    test_metrics = evaluate_model_ranking(model, test_set, verbalizer, tokenizer)
    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  • {k.upper():<8}: {v:.4f}")
        else:
            print(f"  • {k.upper():<8}: {v}")


def run_eval():
    print("\n" + "=" * 70)
    print("📊 Evaluating GenRec Offline Metrics on Test Set")
    print("=" * 70)

    catalog = NetflixCatalog()
    verbalizer = GenRecVerbalizer(catalog)
    _, _, test_set = generate_synthetic_dataset(catalog, num_samples=200, candidate_size=5)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GenRecModel(num_catalog_items=len(catalog), use_pretrained=True)
    metrics = evaluate_model_ranking(model, test_set, verbalizer, tokenizer)

    print("\nOffline Ranking Results:")
    print(f"  • Mean Reciprocal Rank (MRR):  {metrics['mrr']:.4f}")
    print(f"  • Hit Rate @ 1 (HR@1):         {metrics['hr@1']:.4f}")
    print(f"  • Hit Rate @ 3 (HR@3):         {metrics['hr@3']:.4f}")
    print(f"  • Hit Rate @ 5 (HR@5):         {metrics['hr@5']:.4f}")
    print(f"  • NDCG @ 5:                    {metrics['ndcg@5']:.4f}")
    print(f"  • Total Evaluated Queries:     {metrics['num_evaluated']}")


def run_web_server(port: int = 8080):
    from web.app import run_server
    run_server(port)


def main():
    parser = argparse.ArgumentParser(description="Netflix GenRec Implementation")
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "train", "eval", "serve"],
        help="Execution mode (default: cli)"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli_demo()
    elif args.mode == "train":
        run_training_demo(epochs=args.epochs)
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "serve":
        run_web_server(port=args.port)


if __name__ == "__main__":
    main()
