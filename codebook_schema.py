#!/usr/bin/env python3
"""
The codebook schema, loaded once and shared.

`codebook_schema.json` holds the four directly coded dimensions of the coding instrument (their codes,
selection type and exclusive negative codes) and the non-English test used to exclude conversations from
coding. Every script reads the label sets from here, so they are defined in one place. Changing a label
set is a codebook decision.

    from codebook_schema import DIRECT_CODES, DIRECT_FIELDS, DIRECT_VERSION, is_non_english
"""
import json
import re
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "codebook_schema.json"
_S = json.loads(_PATH.read_text(encoding="utf-8"))

DIRECT_INSTRUMENT = _S["direct_instrument"]
DIRECT_FIELDS = {f["name"]: f for f in DIRECT_INSTRUMENT["fields"]}
DIRECT_CODES = {name: list(f["codes"]) for name, f in DIRECT_FIELDS.items()}
DIRECT_VERSION = DIRECT_INSTRUMENT["version"]
DIRECT_STUDENT_AXIS = DIRECT_INSTRUMENT["student_type_axis"]
DIRECT_HISTORICAL_LEVEL_NAMES = DIRECT_INSTRUMENT["historical_level_names"]
DIRECT_DEFAULT = DIRECT_INSTRUMENT["unknown_constraint_origin_default"]

_NON_ENGLISH = re.compile(_S["non_english"]["pattern"])


def is_non_english(*parts) -> bool:
    """True if any part contains CJK, kana or hangul (the language exclusion test)."""
    return any(_NON_ENGLISH.search(str(p or "")) for p in parts)


if __name__ == "__main__":
    print(f"Direct coding instrument {DIRECT_VERSION}:")
    for name, codes in DIRECT_CODES.items():
        print(f"  {name:26s} {codes}")
