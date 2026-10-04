"""
Verbalizer (Context Engineering) module for GenRec.
Translates structured member watch histories, session context, and item metadata
into natural language prompts for the foundation model backbone.
Also provides a comparison with legacy tabular feature engineering.
"""

from typing import List, Dict, Any, Optional
import torch
from data.dataset import NetflixCatalog, WatchInteraction, CatalogItem


class GenRecVerbalizer:
    """
    Implements Context Engineering as described in:
    'GenRec: An LLM-Backed Recommendation Ranker at Netflix' (arXiv:2608.10257).
    Converts viewing sessions, consumption signals, and rich metadata into
    structured natural language sequences.
    """

    def __init__(self, catalog: NetflixCatalog):
        self.catalog = catalog

    def verbalize_interaction(self, interaction: WatchInteraction, index: int) -> str:
        item = self.catalog.get_by_id(interaction.item_id)
        if not item:
            return f"{index}. Unknown Title (ID {interaction.item_id})"

        # Format completion
        pct = int(interaction.completion_pct * 100)
        status_str = f"Completed ({pct}%)" if pct >= 90 else f"Watched {pct}%"

        # Explicit feedback
        rating_str = ""
        if interaction.rating:
            rating_str = f" | Rated: {interaction.rating}"

        # Rewatch flag
        rewatch_str = " | Rewatched" if interaction.rewatched else ""

        genres_str = ", ".join(item.genres[:3])
        mood_str = ", ".join(item.mood[:2])

        return (
            f"  {index}. \"{item.title}\" ({item.type}, {item.release_year}) "
            f"[{genres_str}] - {status_str}{rating_str}{rewatch_str} "
            f"(Mood: {mood_str}, Device: {interaction.device})"
        )

    def verbalize_context(
        self,
        interactions: List[WatchInteraction],
        device: Optional[str] = None,
        time_of_day: Optional[str] = None,
        day_of_week: Optional[str] = None
    ) -> str:
        """
        Creates the complete verbalized prompt representing member session context.
        """
        lines = ["[MEMBER CONTEXT & VIEWING STREAM]"]
        
        ctx_parts = []
        if device:
            ctx_parts.append(f"Device: {device}")
        if time_of_day:
            ctx_parts.append(f"Time: {time_of_day}")
        if day_of_week:
            ctx_parts.append(f"Day: {day_of_week}")
        if ctx_parts:
            lines.append("Context: " + " | ".join(ctx_parts))

        lines.append("Recent Consumption History:")
        for idx, inter in enumerate(interactions, start=1):
            lines.append(self.verbalize_interaction(inter, idx))

        # Aggregate observed moods & genres
        all_genres = []
        all_moods = []
        for inter in interactions:
            it = self.catalog.get_by_id(inter.item_id)
            if it:
                all_genres.extend(it.genres)
                all_moods.extend(it.mood)

        top_genres = list(dict.fromkeys(all_genres))[:4]
        top_moods = list(dict.fromkeys(all_moods))[:4]

        if top_genres or top_moods:
            lines.append(
                f"Demonstrated Tastes: Genres [{', '.join(top_genres)}] | Moods [{', '.join(top_moods)}]"
            )

        lines.append(
            "\n[RECOMMENDATION OBJECTIVE]\n"
            "Predict member engagement across the candidate catalog items for next watch session."
        )

        return "\n".join(lines)

    def verbalize_candidates(self, candidate_ids: List[int]) -> List[str]:
        """Verbalizes individual candidate items for item-side representation."""
        verbalized = []
        for cid in candidate_ids:
            it = self.catalog.get_by_id(cid)
            if it:
                verbalized.append(
                    f"\"{it.title}\" | {it.type} ({it.release_year}) | "
                    f"Genres: {', '.join(it.genres)} | Mood: {', '.join(it.mood)} | "
                    f"Synopsis: {it.synopsis[:140]}..."
                )
            else:
                verbalized.append(f"Item #{cid}")
        return verbalized

    def extract_legacy_tabular_features(self, interactions: List[WatchInteraction]) -> Dict[str, Any]:
        """
        Demonstrates the legacy approach: manual feature engineering into sparse/dense vectors.
        Shows how much semantic richness is lost compared to verbalization.
        """
        genre_counts: Dict[str, int] = {}
        total_pct = 0.0
        rewatch_count = 0
        thumbs_up_count = 0

        for inter in interactions:
            total_pct += inter.completion_pct
            if inter.rewatched:
                rewatch_count += 1
            if inter.rating in ["ThumbsUp", "DoubleThumbsUp"]:
                thumbs_up_count += 1

            it = self.catalog.get_by_id(inter.item_id)
            if it:
                for g in it.genres:
                    genre_counts[f"feat_genre_{g.lower()}"] = genre_counts.get(f"feat_genre_{g.lower()}", 0) + 1

        n = max(1, len(interactions))
        features = {
            "num_recent_watches": len(interactions),
            "avg_completion_pct": round(total_pct / n, 3),
            "rewatch_ratio": round(rewatch_count / n, 3),
            "positive_rating_ratio": round(thumbs_up_count / n, 3),
            **genre_counts
        }
        return features

    def tokenize_prompts(
        self,
        prompts: List[str],
        tokenizer: Any,
        max_length: int = 256,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """Tokenizes verbalized prompts with padding and attention masks."""
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device)
        }
