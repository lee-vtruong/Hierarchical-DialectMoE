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

