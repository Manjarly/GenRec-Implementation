"""
Unit test suite for GenRec components:
- Catalog loading
- Verbalization (Context Engineering)
- Model architecture & Prefill-Only Scoring Head
- Evaluation metrics
"""

import sys
import unittest
from pathlib import Path
import torch

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.dataset import NetflixCatalog, create_predefined_personas, WatchInteraction
from verbalizer.verbalizer import GenRecVerbalizer
from models.genrec import GenRecModel, CatalogAwareScoringHead
from evaluation.metrics import compute_mrr, compute_hit_rate, compute_ndcg_at_k


class TestGenRec(unittest.TestCase):

    def setUp(self):
        self.catalog = NetflixCatalog()
        self.verbalizer = GenRecVerbalizer(self.catalog)
        self.personas = create_predefined_personas(self.catalog)

    def test_catalog_loaded(self):
        self.assertGreater(len(self.catalog), 20)
        item1 = self.catalog.get_by_id(1)
        self.assertIsNotNone(item1)
        self.assertEqual(item1.title, "Stranger Things")
        self.assertIn("Sci-Fi", item1.genres)

    def test_context_verbalization(self):
        persona = self.personas[0]
        prompt = self.verbalizer.verbalize_context(persona.interactions, device="Smart TV", time_of_day="Evening")

        self.assertIn("[MEMBER CONTEXT & VIEWING STREAM]", prompt)
        self.assertIn("Stranger Things", prompt)
        self.assertIn("Completed", prompt)
        self.assertIn("[RECOMMENDATION OBJECTIVE]", prompt)

    def test_legacy_features_extraction(self):
        persona = self.personas[0]
        features = self.verbalizer.extract_legacy_tabular_features(persona.interactions)
        self.assertIn("num_recent_watches", features)
        self.assertIn("avg_completion_pct", features)
        self.assertGreaterEqual(features["num_recent_watches"], 1)

    def test_scoring_head_forward(self):
        head = CatalogAwareScoringHead(hidden_dim=64, num_catalog_items=25, proj_dim=32)
        batch_size = 2
        k = 4
        dummy_context = torch.randn(batch_size, 64)
        dummy_cands = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

        scores = head.forward_candidates(dummy_context, dummy_cands)
        self.assertEqual(scores.shape, (batch_size, k))

        # Test full catalog retrieval
        full_scores = head.forward_full_catalog(dummy_context)
        self.assertEqual(full_scores.shape, (batch_size, 25))

    def test_genrec_model_prefill_ranking(self):
        model = GenRecModel(num_catalog_items=len(self.catalog), use_pretrained=False)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones((1, 32))
        cands = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

        ranked_items, sorted_probs, latency_ms = model.rank_candidates(input_ids, attention_mask, cands)

        self.assertEqual(ranked_items.shape, (1, 5))
        self.assertEqual(sorted_probs.shape, (1, 5))
        self.assertAlmostEqual(sorted_probs.sum().item(), 1.0, places=4)
        self.assertGreater(latency_ms, 0.0)

    def test_metrics_calculation(self):
        # Suppose target item was ranked at 1st, 2nd, 4th positions
        ranks = [1, 2, 4]
        # MRR = (1/1 + 1/2 + 1/4) / 3 = (1 + 0.5 + 0.25) / 3 = 1.75 / 3 = 0.5833
        mrr = compute_mrr(ranks)
        self.assertAlmostEqual(mrr, 1.75 / 3.0, places=3)

        hr1 = compute_hit_rate(ranks, k=1)
        self.assertAlmostEqual(hr1, 1.0 / 3.0, places=3)

        hr3 = compute_hit_rate(ranks, k=3)
        self.assertAlmostEqual(hr3, 2.0 / 3.0, places=3)

        ndcg = compute_ndcg_at_k(ranks, k=3)
        self.assertGreater(ndcg, 0.0)


if __name__ == "__main__":
    unittest.main()
