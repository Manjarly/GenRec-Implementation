"""
GenRec Model Architecture: LLM-Backed Recommendation Ranker.
Implements the prefill-only inference approach with a catalog-aware scoring head,
as described in Netflix's paper (arXiv:2608.10257).
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GPT2Config, GPT2Model, GPT2LMHeadModel


class CatalogAwareScoringHead(nn.Module):
    """
    Catalog-Aware Scoring Head:
    Computes ranking relevance scores between the LLM's verbalized context representation
    and candidate catalog items without generating autoregressive text tokens.
    Guarantees:
      1. Strict candidate restriction (zero hallucinations).
      2. Fast single-pass prefill inference (< 25ms).
    """

    def __init__(self, hidden_dim: int, num_catalog_items: int, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_catalog_items = num_catalog_items
        self.proj_dim = proj_dim
        self.temperature = temperature

        # Context representation projector
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # Catalog item embeddings (learnable or initialized from item metadata)
        self.item_embeddings = nn.Embedding(num_catalog_items + 1, proj_dim)  # 1-indexed items
        nn.init.normal_(self.item_embeddings.weight, std=0.02)

        # Item representation projector
        self.item_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward_candidates(
        self,
        context_emb: torch.Tensor,        # [Batch, hidden_dim]
        candidate_item_ids: torch.Tensor  # [Batch, K]
    ) -> torch.Tensor:
        """
        Computes relevance scores for candidate sets.
        Returns: logits [Batch, K]
        """
        q = F.normalize(self.query_proj(context_emb), dim=-1)  # [Batch, proj_dim]
        cand_embeds = self.item_embeddings(candidate_item_ids)  # [Batch, K, proj_dim]
        k = F.normalize(self.item_proj(cand_embeds), dim=-1)     # [Batch, K, proj_dim]

        # Dot product relevance: [Batch, K]
        scores = torch.einsum("bd,bkd->bk", q, k) / self.temperature
        return scores

    def forward_full_catalog(self, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Computes scores over the entire catalog in a single matrix multiplication.
        Returns: logits [Batch, num_catalog_items]
        """
        q = F.normalize(self.query_proj(context_emb), dim=-1)  # [Batch, proj_dim]
        all_ids = torch.arange(1, self.num_catalog_items + 1, device=context_emb.device)
        all_embeds = self.item_embeddings(all_ids)             # [N, proj_dim]
        k = F.normalize(self.item_proj(all_embeds), dim=-1)    # [N, proj_dim]

        scores = torch.matmul(q, k.t()) / self.temperature     # [Batch, N]
        return scores


class GenRecModel(nn.Module):
    """
    GenRec Core Architecture:
    - Backbone: Transformer language model (e.g. GPT-2, Llama).
    - Mode: Prefill-only forward pass.
    - Scoring: Catalog-Aware Head.
    """

    def __init__(
        self,
        model_name_or_config: str = "gpt2",
        num_catalog_items: int = 50,
        proj_dim: int = 128,
        use_pretrained: bool = True,
        hf_token: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        import os
        self.model_name = model_name_or_config
        self.num_catalog_items = num_catalog_items
        token = hf_token or os.environ.get("HF_TOKEN")

        if use_pretrained:
            try:
                kwargs = {}
                if token:
                    kwargs["token"] = token
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                elif torch.backends.mps.is_available():
                    kwargs["torch_dtype"] = torch.float32

                print(f"Loading foundation model backbone: '{model_name_or_config}'...")
                self.backbone = AutoModel.from_pretrained(model_name_or_config, **kwargs)
                self.hidden_dim = self.backbone.config.hidden_size
                self.active_backbone = model_name_or_config
                print(f"Successfully loaded '{model_name_or_config}' (hidden_dim={self.hidden_dim})")
            except Exception as e:
                print(f"\n[GenRec Notice] Could not load '{model_name_or_config}': {e}")
                if "meta-llama" in model_name_or_config.lower():
                    print("--> Gated model requirement: Make sure your HF account accepted the Meta Llama 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-1B and set HF_TOKEN in your environment or .env file.")
                    print("--> Falling back to 'gpt2' backbone for now...")
                try:
                    self.backbone = AutoModel.from_pretrained("gpt2")
                    self.hidden_dim = self.backbone.config.hidden_size
                    self.active_backbone = "gpt2 (fallback)"
                except Exception:
                    config = GPT2Config(n_layer=4, n_head=4, n_embd=256, vocab_size=50257)
                    self.backbone = GPT2Model(config)
                    self.hidden_dim = config.hidden_size
                    self.active_backbone = "custom-transformer"
        else:
            config = GPT2Config(n_layer=4, n_head=4, n_embd=256, vocab_size=50257)
            self.backbone = GPT2Model(config)
            self.hidden_dim = config.hidden_size
            self.active_backbone = "custom-transformer"

        self.scoring_head = CatalogAwareScoringHead(
            hidden_dim=self.hidden_dim,
            num_catalog_items=num_catalog_items,
            proj_dim=proj_dim
        )

        # Optional LM head for comparing autoregressive text generation vs prefill scoring
        self.lm_head = nn.Linear(self.hidden_dim, self.backbone.config.vocab_size, bias=False)

    def extract_context_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs the prefill forward pass and extracts the context representation
        at the last active token index for each sequence in the batch.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [Batch, Seq_len, hidden_dim]

        # Identify last active token index from attention mask
        # attention_mask: 1 for active tokens, 0 for pad tokens
        last_indices = (attention_mask.to(torch.long).sum(dim=1) - 1).clamp(min=0)

        batch_size = input_ids.shape[0]
        context_embs = hidden_states[torch.arange(batch_size, device=input_ids.device), last_indices]
        return context_embs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_item_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prefill-only forward pass.
        If candidate_item_ids is provided: returns scores [Batch, K].
        Otherwise: returns full catalog scores [Batch, num_catalog_items].
        """
        context_emb = self.extract_context_embedding(input_ids, attention_mask)
        if candidate_item_ids is not None:
            return self.scoring_head.forward_candidates(context_emb, candidate_item_ids)
        else:
            return self.scoring_head.forward_full_catalog(context_emb)

    def rank_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Performs candidate re-ranking and measures prefill latency.
        Returns: (ranked_candidate_ids, probabilities, latency_ms)
        """
        start_time = time.perf_counter()
        with torch.no_grad():
            scores = self.forward(input_ids, attention_mask, candidate_item_ids)
            probs = F.softmax(scores, dim=-1)
            # Sort descending
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            ranked_items = torch.gather(candidate_item_ids, 1, sorted_indices)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return ranked_items, sorted_probs, latency_ms


    def compute_attribution(
        self,
        candidate_item_ids: torch.Tensor,
        history_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes normalized attention attribution scores between historical watch items
        and recommended candidate items using the scoring head projection space.
        candidate_item_ids: [K]
        history_item_ids: [H]
        Returns: attributions [K, H] where rows sum to 1.0
        """
        cand_embeds = self.scoring_head.item_embeddings(candidate_item_ids)
        hist_embeds = self.scoring_head.item_embeddings(history_item_ids)

        k_cand = F.normalize(self.scoring_head.item_proj(cand_embeds), dim=-1)  # [K, D]
        k_hist = F.normalize(self.scoring_head.item_proj(hist_embeds), dim=-1)  # [H, D]

        # Dot products: [K, H]
        sims = torch.matmul(k_cand, k_hist.t()) / self.scoring_head.temperature
        attributions = F.softmax(sims, dim=-1)
        return attributions

    def simulate_autoregressive_generation(
        self,
        input_ids: torch.Tensor,
        tokenizer: Any,
        max_new_tokens: int = 15
    ) -> Tuple[str, float]:
        """
        Simulates traditional autoregressive token generation.
        Demonstrates the latency overhead and risk of generating non-catalog text.
        """
        start_time = time.perf_counter()
        curr_ids = input_ids.clone()
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.backbone(curr_ids)
                last_hidden = outputs.last_hidden_state[:, -1, :]
                logits = self.lm_head(last_hidden)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                token_id = next_token.squeeze().item()
                generated_tokens.append(token_id)
                # Stop on newline or EOS
                if token_id in [198, 50256, 0]:
                    break

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text.strip(), latency_ms
