"""
Two-Phase Training Pipeline for GenRec:
Phase 1: Foundation Model Domain Adaptation (catalog semantics & language modeling).
Phase 2: Recommendation Post-Training / Alignment (candidate ranking loss with reward weighting).
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from data.dataset import NetflixCatalog, RankingInstance, WatchInteraction
from verbalizer.verbalizer import GenRecVerbalizer
from models.genrec import GenRecModel


class RankingDataset(Dataset):
    """PyTorch Dataset for GenRec ranking instances."""

    def __init__(
        self,
        instances: List[RankingInstance],
        verbalizer: GenRecVerbalizer,
        tokenizer: Any,
        max_length: int = 192
    ):
        self.instances = instances
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-verbalize prompts
        self.prompts = [
            verbalizer.verbalize_context(inst.interactions)
            for inst in instances
        ]

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        inst = self.instances[idx]
        prompt = self.prompts[idx]
        return {
            "prompt": prompt,
            "candidates": torch.tensor(inst.candidate_item_ids, dtype=torch.long),
            "label_idx": torch.tensor(inst.label_idx, dtype=torch.long),
            "reward_weight": torch.tensor(inst.reward_weight, dtype=torch.float)
        }


def collate_ranking_batch(batch: List[Dict[str, Any]], tokenizer: Any, max_length: int = 192) -> Dict[str, Any]:
    prompts = [b["prompt"] for b in batch]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    candidates = torch.stack([b["candidates"] for b in batch])
    labels = torch.stack([b["label_idx"] for b in batch])
    rewards = torch.stack([b["reward_weight"] for b in batch])

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "candidates": candidates,
        "labels": labels,
        "rewards": rewards
    }


class GenRecTrainer:
    """Orchestrates Phase 1 Adaptation and Phase 2 Post-Training."""

    def __init__(
        self,
        model: GenRecModel,
        tokenizer: Any,
        verbalizer: GenRecVerbalizer,
        catalog: NetflixCatalog,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.verbalizer = verbalizer
        self.catalog = catalog
        self.device = device

    def train_phase1_adaptation(
        self,
        epochs: int = 2,
        lr: float = 5e-5,
        batch_size: int = 4
    ) -> List[float]:
        """
        Phase 1: Foundation Model Domain Adaptation.
        Adapts the LLM backbone on catalog synopses and viewing sequences.
        """
        print("\n--- Starting Phase 1: Foundation Model Domain Adaptation ---")
        texts = []
        # 1. Verbalized catalog descriptions
        for item in self.catalog.item_list:
            texts.append(item.to_verbalized_text())

        # 2. Synthetic viewing narrative sequences
        for item in self.catalog.item_list:
            related_titles = [other.title for other in self.catalog.item_list if any(g in other.genres for g in item.genres) and other.id != item.id]
            if related_titles:
                texts.append(f"Viewers who enjoyed \"{item.title}\" frequently proceeded to watch {', '.join(related_titles[:3])}.")

        # Tokenize adaptation corpus
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        epoch_losses = []
        for ep in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for i in range(0, len(input_ids), batch_size):
                b_ids = input_ids[i:i + batch_size]
                b_mask = attention_mask[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model.backbone(b_ids, attention_mask=b_mask)
                logits = self.model.lm_head(outputs.last_hidden_state)

                # Next token prediction loss: shift logits and labels
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = b_ids[..., 1:].contiguous()

                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            epoch_losses.append(avg_loss)
            print(f"Phase 1 - Epoch {ep + 1}/{epochs} | LM Adaptation Loss: {avg_loss:.4f}")

        return epoch_losses

    def train_phase2_ranking(
        self,
        train_instances: List[RankingInstance],
        val_instances: List[RankingInstance],
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 3e-4
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Recommendation Post-Training / Alignment.
        Trains the catalog-aware scoring head on candidate ranking instances
        with implicit satisfaction reward weighting.
        """
        print("\n--- Starting Phase 2: Recommendation Ranking Post-Training ---")
        train_ds = RankingDataset(train_instances, self.verbalizer, self.tokenizer)
        val_ds = RankingDataset(val_instances, self.verbalizer, self.tokenizer)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_ranking_batch(b, self.tokenizer)
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_ranking_batch(b, self.tokenizer)
        )

        # Optimize scoring head and top transformer layers
        params = [
            {"params": self.model.scoring_head.parameters(), "lr": lr},
            {"params": self.model.backbone.parameters(), "lr": lr * 0.1}
        ]
        optimizer = torch.optim.AdamW(params)

        metrics_history = {"train_loss": [], "val_mrr": [], "val_hr1": []}

        for ep in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for batch in train_loader:
                b_input_ids = batch["input_ids"].to(self.device)
                b_mask = batch["attention_mask"].to(self.device)
                b_cands = batch["candidates"].to(self.device)
                b_labels = batch["labels"].to(self.device)
                b_rewards = batch["rewards"].to(self.device)

                optimizer.zero_grad()
                # Prefill-only forward pass to obtain candidate logits [B, K]
                logits = self.model(b_input_ids, b_mask, b_cands)

                # Reward-weighted cross-entropy loss
                log_probs = F.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(1, b_labels.unsqueeze(1)).squeeze(1)
                loss = -(b_rewards * target_log_probs).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                num_train_batches += 1

            avg_train_loss = total_train_loss / max(1, num_train_batches)
            metrics_history["train_loss"].append(avg_train_loss)

            # Evaluate on validation set
            val_mrr, val_hr1 = self.evaluate_ranking(val_loader)
            metrics_history["val_mrr"].append(val_mrr)
            metrics_history["val_hr1"].append(val_hr1)

            print(
                f"Phase 2 - Epoch {ep + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val MRR: {val_mrr:.4f} | "
                f"Val HitRate@1: {val_hr1:.4f}"
            )

        return metrics_history

    def evaluate_ranking(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluates MRR and HitRate@1 on a dataloader."""
        self.model.eval()
        reciprocal_ranks = []
        hits_at_1 = []

        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch["input_ids"].to(self.device)
                b_mask = batch["attention_mask"].to(self.device)
                b_cands = batch["candidates"].to(self.device)
                b_labels = batch["labels"].to(self.device)

                logits = self.model(b_input_ids, b_mask, b_cands)
                # Sort logits descending
                _, indices = torch.sort(logits, dim=-1, descending=True)

                for true_idx, ranked_order in zip(b_labels, indices):
                    rank = (ranked_order == true_idx).nonzero(as_tuple=True)[0].item() + 1
                    reciprocal_ranks.append(1.0 / rank)
                    hits_at_1.append(1.0 if rank == 1 else 0.0)

        mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
        hr1 = sum(hits_at_1) / max(1, len(hits_at_1))
        return mrr, hr1
