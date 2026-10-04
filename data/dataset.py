"""
Dataset module for Netflix GenRec.
Provides Netflix catalog models, real viewing interaction patterns,
and candidate ranking datasets.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class CatalogItem:
    id: int
    title: str
    type: str  # Movie or Series
    genres: List[str]
    tags: List[str]
    mood: List[str]
    cast: List[str]
    synopsis: str
    release_year: int
    maturity_rating: str
    duration: str

    def to_verbalized_text(self) -> str:
        genres_str = ", ".join(self.genres)
        tags_str = ", ".join(self.tags)
        mood_str = ", ".join(self.mood)
        cast_str = ", ".join(self.cast[:3])
        return (
            f'Title: "{self.title}" ({self.type}, {self.release_year}, {self.maturity_rating})\n'
            f"Genres: {genres_str}\n"
            f"Mood: {mood_str}\n"
            f"Tags: {tags_str}\n"
            f"Cast: {cast_str}\n"
            f"Synopsis: {self.synopsis}"
        )


@dataclass
class WatchInteraction:
    item_id: int
    completion_pct: float  # 0.0 to 1.0 (e.g. 0.95 = 95% finished)
    rating: Optional[str]   # "ThumbsUp", "DoubleThumbsUp", "ThumbsDown", or None
    rewatched: bool
    device: str            # "Smart TV", "Mobile", "Laptop"
    time_of_day: str       # "Evening", "Late Night", "Afternoon", "Morning"
    day_of_week: str       # "Weekend", "Weekday"


@dataclass
class ViewingScenario:
    scenario_id: str
    name: str
    description: str
    primary_genres: List[str]
    default_device: str
    interactions: List[WatchInteraction] = field(default_factory=list)


@dataclass
class RankingInstance:
    instance_id: str
    scenario_id: str
    interactions: List[WatchInteraction]
    target_item_id: int
    candidate_item_ids: List[int]
    label_idx: int
    reward_weight: float = 1.0


class NetflixCatalog:
    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path is None:
            catalog_path = str(Path(__file__).parent / "catalog.json")
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items: Dict[int, CatalogItem] = {
            item["id"]: CatalogItem(**item) for item in data
        }
        self.item_list: List[CatalogItem] = list(self.items.values())
        self.id_to_idx = {item.id: idx for idx, item in enumerate(self.item_list)}
        self.idx_to_id = {idx: item.id for idx, item in enumerate(self.item_list)}

    def __len__(self) -> int:
        return len(self.item_list)

    def get_by_id(self, item_id: int) -> Optional[CatalogItem]:
        return self.items.get(item_id)

    def get_all_ids(self) -> List[int]:
        return list(self.items.keys())


def create_curated_scenarios(catalog: NetflixCatalog) -> List[ViewingScenario]:
    """
    Curated viewing scenarios reflecting real-world viewing trajectories.
    """
    scenarios = [
        ViewingScenario(
            scenario_id="sc_scifi",
            name="Sci-Fi & Mind-Bending Mystery",
            description="Focuses on high-concept speculative physics, time loops, and philosophical suspense.",
            primary_genres=["Sci-Fi", "Mystery", "Drama"],
            default_device="Smart TV",
            interactions=[
                WatchInteraction(item_id=1, completion_pct=1.0, rating="ThumbsUp", rewatched=True, device="Smart TV", time_of_day="Evening", day_of_week="Weekend"),
                WatchInteraction(item_id=2, completion_pct=0.98, rating="DoubleThumbsUp", rewatched=False, device="Smart TV", time_of_day="Late Night", day_of_week="Weekend"),
                WatchInteraction(item_id=3, completion_pct=0.85, rating="ThumbsUp", rewatched=False, device="Smart TV", time_of_day="Evening", day_of_week="Weekday"),
                WatchInteraction(item_id=4, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=True, device="Smart TV", time_of_day="Late Night", day_of_week="Weekend"),
            ]
        ),
        ViewingScenario(
            scenario_id="sc_crime",
            name="Psychological Crime & Gritty Noir",
            description="Focuses on criminal profiling, moral descent, cartel conspiracies, and forensic suspense.",
            primary_genres=["Crime", "Drama", "Psychological Thriller"],
            default_device="Smart TV",
            interactions=[
                WatchInteraction(item_id=6, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=False, device="Smart TV", time_of_day="Late Night", day_of_week="Weekday"),
                WatchInteraction(item_id=8, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=True, device="Smart TV", time_of_day="Evening", day_of_week="Weekend"),
                WatchInteraction(item_id=7, completion_pct=0.92, rating="ThumbsUp", rewatched=False, device="Smart TV", time_of_day="Evening", day_of_week="Weekend"),
                WatchInteraction(item_id=10, completion_pct=0.80, rating=None, rewatched=False, device="Laptop", time_of_day="Afternoon", day_of_week="Weekday"),
            ]
        ),
        ViewingScenario(
            scenario_id="sc_comedy",
            name="Feel-Good Sitcoms & Witty Humor",
            description="Comfort viewing with ensemble casts, comedic pacing, and philosophical heart.",
            primary_genres=["Comedy", "Wholesome", "Feel-Good"],
            default_device="Laptop",
            interactions=[
                WatchInteraction(item_id=11, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=True, device="Laptop", time_of_day="Evening", day_of_week="Weekday"),
                WatchInteraction(item_id=12, completion_pct=0.95, rating="ThumbsUp", rewatched=True, device="Laptop", time_of_day="Afternoon", day_of_week="Weekend"),
                WatchInteraction(item_id=13, completion_pct=1.0, rating="ThumbsUp", rewatched=False, device="Mobile", time_of_day="Morning", day_of_week="Weekday"),
            ]
        ),
        ViewingScenario(
            scenario_id="sc_action",
            name="High-Octane Action & Steampunk Fantasy",
            description="High visual spectacle, fight choreography, futuristic dystopias, and world-building.",
            primary_genres=["Action", "Fantasy", "Animation"],
            default_device="Smart TV",
            interactions=[
                WatchInteraction(item_id=16, completion_pct=0.90, rating="ThumbsUp", rewatched=False, device="Smart TV", time_of_day="Evening", day_of_week="Weekend"),
                WatchInteraction(item_id=17, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=True, device="Smart TV", time_of_day="Late Night", day_of_week="Weekend"),
                WatchInteraction(item_id=19, completion_pct=1.0, rating="ThumbsUp", rewatched=False, device="Smart TV", time_of_day="Evening", day_of_week="Weekday"),
                WatchInteraction(item_id=25, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=False, device="Laptop", time_of_day="Late Night", day_of_week="Weekend"),
            ]
        ),
        ViewingScenario(
            scenario_id="sc_doc",
            name="Sports Legends & Nature Documentaries",
            description="Real-world triumphs, wildlife cinematography, high-pressure athletic rivalries, and history.",
            primary_genres=["Documentary", "Sport", "Nature"],
            default_device="Smart TV",
            interactions=[
                WatchInteraction(item_id=21, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=False, device="Smart TV", time_of_day="Evening", day_of_week="Weekend"),
                WatchInteraction(item_id=22, completion_pct=0.95, rating="ThumbsUp", rewatched=False, device="Smart TV", time_of_day="Afternoon", day_of_week="Weekend"),
                WatchInteraction(item_id=23, completion_pct=1.0, rating="DoubleThumbsUp", rewatched=True, device="Smart TV", time_of_day="Evening", day_of_week="Weekday"),
            ]
        )
    ]
    return scenarios


# Backward compatibility alias
create_predefined_personas = create_curated_scenarios
MemberProfile = ViewingScenario

import random

def generate_synthetic_dataset(
    catalog: NetflixCatalog,
    num_samples: int = 400,
    candidate_size: int = 5,
    seed: int = 42
) -> Tuple[List[RankingInstance], List[RankingInstance], List[RankingInstance]]:
    random.seed(seed)
    all_scenarios = create_curated_scenarios(catalog)
    all_item_ids = catalog.get_all_ids()

    genre_to_items: Dict[str, List[int]] = {}
    for item in catalog.item_list:
        for g in item.genres:
            genre_to_items.setdefault(g, []).append(item.id)

    instances: List[RankingInstance] = []
    for i in range(num_samples):
        base_scenario = random.choice(all_scenarios)
        matched_items = []
        for g in base_scenario.primary_genres:
            matched_items.extend(genre_to_items.get(g, []))
        matched_items = list(set(matched_items)) or all_item_ids

        hist_len = random.randint(2, 4)
        sample_pool = [it for it in matched_items]
        random.shuffle(sample_pool)
        hist_item_ids = sample_pool[:hist_len]

        history: List[WatchInteraction] = []
        for hid in hist_item_ids:
            history.append(
                WatchInteraction(
                    item_id=hid,
                    completion_pct=random.choice([0.75, 0.85, 0.95, 1.0]),
                    rating=random.choice(["ThumbsUp", "DoubleThumbsUp", None]),
                    rewatched=random.random() < 0.25,
                    device=random.choice(["Smart TV", "Laptop", "Mobile"]),
                    time_of_day=random.choice(["Evening", "Late Night", "Afternoon"]),
                    day_of_week=random.choice(["Weekend", "Weekday"])
                )
            )

        available_positives = [it for it in matched_items if it not in hist_item_ids] or all_item_ids
        target_item_id = random.choice(available_positives)

        available_negatives = [it for it in all_item_ids if it not in hist_item_ids and it != target_item_id]
        negatives = random.sample(available_negatives, k=min(candidate_size - 1, len(available_negatives)))

        candidates = negatives + [target_item_id]
        random.shuffle(candidates)
        label_idx = candidates.index(target_item_id)

        instances.append(
            RankingInstance(
                instance_id=f"inst_{i:04d}",
                scenario_id=base_scenario.scenario_id,
                interactions=history,
                target_item_id=target_item_id,
                candidate_item_ids=candidates,
                label_idx=label_idx,
                reward_weight=1.0 + (0.5 if random.random() > 0.4 else 0.0)
            )
        )

    n_train = int(len(instances) * 0.70)
    n_val = int(len(instances) * 0.15)
    return instances[:n_train], instances[n_train:n_train + n_val], instances[n_train + n_val:]
