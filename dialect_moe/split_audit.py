from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True)
class UtteranceRecord:
    original_split: str
    row_index: int
    filename: str
    speaker_id: str
    region: str
    province: str
    duration_seconds: float | None = None
    audio_sha256: str | None = None


def duplicate_values_by_split(
    records: Iterable[UtteranceRecord], attribute: str
) -> dict[str, dict[str, int]]:
    value_splits: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        value = str(getattr(record, attribute) or "")
        if value:
            value_splits[value][record.original_split] += 1
    return {
        value: dict(counts)
        for value, counts in value_splits.items()
        if len(counts) > 1
    }


def speaker_label_conflicts(
    records: Iterable[UtteranceRecord],
) -> dict[str, dict[str, list[str]]]:
    labels: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"regions": set(), "provinces": set()}
    )
    for record in records:
        labels[record.speaker_id]["regions"].add(record.region)
        labels[record.speaker_id]["provinces"].add(record.province)
    return {
        speaker: {
            "regions": sorted(values["regions"]),
            "provinces": sorted(values["provinces"]),
        }
        for speaker, values in labels.items()
        if len(values["regions"]) > 1 or len(values["provinces"]) > 1
    }


def assign_speakers_stratified(
    records: Iterable[UtteranceRecord],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign each speaker once, approximately stratified by majority province."""
    if not ratios or any(value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive")
    total_ratio = sum(ratios.values())
    normalized_ratios = {
        split: value / total_ratio for split, value in ratios.items()
    }

    speaker_records: dict[str, list[UtteranceRecord]] = defaultdict(list)
    for record in records:
        if not record.speaker_id:
            raise ValueError("Cannot build speaker-disjoint split with empty speaker_id")
        speaker_records[record.speaker_id].append(record)

    province_speakers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for speaker, rows in speaker_records.items():
        province_counts = Counter(row.province for row in rows)
        majority_province = sorted(
            province_counts, key=lambda value: (-province_counts[value], value)
        )[0]
        province_speakers[majority_province].append((speaker, len(rows)))

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    split_names = list(normalized_ratios)
    for province in sorted(province_speakers):
        speakers = province_speakers[province]
        rng.shuffle(speakers)
        speakers.sort(key=lambda item: item[1], reverse=True)
        target = {
            split: sum(count for _, count in speakers) * ratio
            for split, ratio in normalized_ratios.items()
        }
        current = {split: 0 for split in split_names}

        # When possible, guarantee that every split receives a speaker from the
        # province before greedily balancing utterance counts.
        if len(speakers) >= len(split_names):
            for split, (speaker, count) in zip(
                sorted(split_names, key=lambda name: normalized_ratios[name]),
                speakers[: len(split_names)],
            ):
                assignments[speaker] = split
                current[split] += count
            remaining = speakers[len(split_names) :]
        else:
            remaining = speakers

        for speaker, count in remaining:
            split = max(
                split_names,
                key=lambda name: (
                    (target[name] - current[name]) / max(target[name], 1e-8),
                    -current[name],
                    name,
                ),
            )
            assignments[speaker] = split
            current[split] += count
    return assignments


def assign_speakers_preserving_splits(
    records: Iterable[UtteranceRecord],
    priority: list[str],
) -> dict[str, str]:
    """Repair overlap with the fewest policy-driven split changes.

    A speaker present in multiple original splits is assigned to the
    highest-priority split. With train > valid > test, no speaker observed
    during training or model selection remains in the held-out test set.
    """
    priority_rank = {split: index for index, split in enumerate(priority)}
    if len(priority_rank) != len(priority):
        raise ValueError("Split priority contains duplicate names")
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not record.speaker_id:
            raise ValueError("Cannot repair splits with empty speaker_id")
        if record.original_split not in priority_rank:
            raise ValueError(
                f"Original split {record.original_split!r} is absent from priority"
            )
        speaker_splits[record.speaker_id].add(record.original_split)
    return {
        speaker: min(splits, key=lambda split: priority_rank[split])
        for speaker, splits in speaker_splits.items()
    }


def split_distribution(
    records: Iterable[UtteranceRecord], assignments: dict[str, str]
) -> dict[str, dict]:
    summary: dict[str, dict] = defaultdict(
        lambda: {
            "utterances": 0,
            "speakers": set(),
            "regions": Counter(),
            "provinces": Counter(),
            "duration_seconds": 0.0,
        }
    )
    for record in records:
        split = assignments[record.speaker_id]
        row = summary[split]
        row["utterances"] += 1
        row["speakers"].add(record.speaker_id)
        row["regions"][record.region] += 1
        row["provinces"][record.province] += 1
        row["duration_seconds"] += record.duration_seconds or 0.0
    return {
        split: {
            **values,
            "speakers": len(values["speakers"]),
            "regions": dict(sorted(values["regions"].items())),
            "provinces": dict(sorted(values["provinces"].items())),
        }
        for split, values in sorted(summary.items())
    }
