"""
Verify that expected_annotations in test-passages.json have correct char offsets.
Uses Python text.index() to validate start_char/end_char point to expected substrings.
"""
import json
import sys
from pathlib import Path


def verify() -> bool:
    fixture_path = Path(__file__).parent / "test-passages.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        passages = json.load(f)

    all_passed = True
    for passage in passages:
        text = passage["text"]
        pid = passage["id"]
        for ann in passage.get("expected_annotations", []):
            expected_substr = text[ann["start_char"]:ann["end_char"]]
            if expected_substr != ann["label"]:
                print(
                    f"FAIL [{pid}] category={ann['category']} "
                    f"offsets=[{ann['start_char']}:{ann['end_char']}] "
                    f"expected='{ann['label']}' got='{expected_substr}'"
                )
                all_passed = False
            else:
                print(
                    f"PASS [{pid}] {ann['category']}: "
                    f"'{ann['label']}' @ [{ann['start_char']}:{ann['end_char']}]"
                )

    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
