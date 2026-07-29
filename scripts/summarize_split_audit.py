from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/h6_split_audit/audit_summary.json"
    )
    parser.add_argument(
        "--split-summary",
        default="outputs/h6_split_audit/speaker_disjoint_summary.json",
    )
    parser.add_argument(
        "--output", default="outputs/h6_split_audit/BAO_CAO_AUDIT.md"
    )
    args = parser.parse_args()

    with Path(args.audit).open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    split_path = Path(args.split_summary)
    split_summary = None
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            split_summary = json.load(handle)

    lines = [
        "# Báo cáo audit split ViMD",
        "",
        f"- Audio mode: `{audit['audio_mode']}`",
        f"- Speaker có xung đột nhãn: {audit['speaker_label_conflicts']}",
        "",
        "## Split gốc",
        "",
        "| Split | Utterance | Speaker | Thời lượng (giây) |",
        "|---|---:|---:|---:|",
    ]
    for split, values in audit["splits"].items():
        duration = values["duration_seconds"]
        lines.append(
            f"| {split} | {values['utterances']} | {values['speakers']} | "
            f"{duration if duration is not None else 'N/A'} |"
        )
    lines.extend(["", "## Overlap chéo split", ""])
    for kind, values in audit["cross_split_overlap"].items():
        lines.append(
            f"- `{kind}`: {values['unique_values']} giá trị, "
            f"{values['affected_utterances']} utterance bị ảnh hưởng."
        )
    if split_summary is not None:
        lines.extend(
            [
                "",
                "## Speaker-disjoint split đề xuất",
                "",
                "| Split | Utterance | Speaker |",
                "|---|---:|---:|",
            ]
        )
        for split, values in split_summary["distribution"].items():
            lines.append(
                f"| {split} | {values['utterances']} | {values['speakers']} |"
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
