from __future__ import annotations

REGION_ALIASES = {
    "north": "North",
    "northern": "North",
    "central": "Central",
    "center": "Central",
    "south": "South",
    "southern": "South",
}


def normalize_region(value: object) -> str:
    text = str(value).strip()
    return REGION_ALIASES.get(text.lower(), text)


class LabelVocabulary:
    def __init__(self, values: list[object]):
        normalized = sorted({str(value) for value in values})
        self.labels = normalized
        self.to_id = {label: index for index, label in enumerate(normalized)}

    def encode(self, value: object) -> int:
        return self.to_id[str(value)]

    def decode(self, index: int) -> str:
        return self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)


def build_province_to_region(
    pairs: list[tuple[object, object]],
    region_vocab: LabelVocabulary,
    province_vocab: LabelVocabulary,
) -> list[int]:
    """Build and validate the one-to-one province-to-region label mapping."""
    mapping = [-1] * len(province_vocab)
    for province_value, region_value in pairs:
        province_id = province_vocab.encode(province_value)
        region_id = region_vocab.encode(normalize_region(region_value))
        previous = mapping[province_id]
        if previous not in {-1, region_id}:
            province = province_vocab.decode(province_id)
            raise ValueError(
                f"Province {province!r} maps to multiple regions: "
                f"{region_vocab.decode(previous)!r} and "
                f"{region_vocab.decode(region_id)!r}"
            )
        mapping[province_id] = region_id
    missing = [
        province_vocab.decode(index)
        for index, region_id in enumerate(mapping)
        if region_id < 0
    ]
    if missing:
        raise ValueError(f"Provinces missing a region mapping: {missing}")
    return mapping
