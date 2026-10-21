"""Shared rhythm-format definitions. Keep HIT order stable for trained models."""

from types import MappingProxyType


DECORATED_SUFFIX = "_decorated"
SILENT_HIT = "HIT_S"

# Base HIT, sample subfolder, supports decoration. This is the sole HIT inventory.
_HIT_DEFINITIONS = (
    (SILENT_HIT, None, False),
    ("HIT_D", "doums", True),
    ("HIT_OTA", "taks", True),
    ("HIT_OTI", "tiks", False),
    ("HIT_PA2", "pa2s", True),
)
HIT_ORDER = tuple(hit for hit, _, _ in _HIT_DEFINITIONS) + tuple(
    hit + DECORATED_SUFFIX for hit, _, decorated in _HIT_DEFINITIONS if decorated
)
HITS = frozenset(HIT_ORDER)
HIT_FOLDERS = MappingProxyType({hit: folder for hit, folder, _ in _HIT_DEFINITIONS if folder is not None})

MAX_SUBDIVISION = 8
SUBDIVISION_TOKENS = tuple(f"SUBD_{size}" for size in range(1, MAX_SUBDIVISION + 1))
STRUCTURAL = ("<PAD>", "<SOC>", "<EOC>", "<SOB>", "<EOB>")
DERBAKE_LINE_COUNT = 5
GENERATED_TEMPO = 120
GENERATED_AMPLITUDE = 1.5

NORMAL_DATA = "normal"
DECORATED_DATA = "decorated"
DATA_DECORATIONS = (NORMAL_DATA, DECORATED_DATA)
# Bump when preparation or its saved metadata contract changes.
PREPARATION_VERSION = 6
NORMALIZATION_RULE = "Overlay line-four skeleton events onto variation HIT positions."
SPLIT_RULE = "Chronological complete cycles within each file: 70% train, 20% validation, 10% test."
