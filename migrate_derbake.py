"""Convert the local four-line corpus to the strict five-line LDM format.

The legacy notebook discarded trailing beats outside complete cycles and
right-padded shortened beats with silence. This tool preserves those rules,
records every change, and leaves source files untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


HIT = re.compile(r"HIT_[A-Za-z0-9]+(?:_decorated)?(?:_SKEL)?\Z")
SUBD = re.compile(r"SUBD_([1-9][0-9]*)\Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cycle_map_from_skeletons(path: Path) -> dict[str, int]:
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(",") if cell.strip()]
        if len(cells) < 3 or len(cells[1:]) % 2:
            raise ValueError(f"{path}:{line_number}: invalid skeleton pattern")
        delays = [float(cells[i]) for i in range(1, len(cells), 2)]
        if not all(math.isfinite(x) and x >= 0 for x in delays):
            raise ValueError(f"{path}:{line_number}: invalid delay")
        size = int(round(sum(delays)))
        if size < 1:
            raise ValueError(f"{path}:{line_number}: invalid cycle size")
        result[Path(cells[0]).stem] = size
    return result


def cycle_map_from_report(path: Path) -> tuple[dict[str, int], dict[str, dict]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["files"]
    details = {Path(row["source"]).stem: row for row in rows}
    if len(details) != len(rows):
        raise ValueError("Duplicate source names in cycle report")
    return {name: int(row["beats_per_cycle"]) for name, row in details.items()}, details


def parse_variations(line: str) -> tuple[list[list[str]], list[int]]:
    tokens = line.split()
    beats: list[list[str]] = []
    padding_by_beat = []
    i = 0
    while i < len(tokens):
        match = SUBD.fullmatch(tokens[i])
        if not match:
            raise ValueError(f"Expected SUBD_n at token {i}: {tokens[i]!r}")
        subd = int(match.group(1))
        beat = [tokens[i]]
        i += 1
        pairs = 0
        while i < len(tokens) and not tokens[i].startswith("SUBD_"):
            if i + 1 >= len(tokens) or not HIT.fullmatch(tokens[i]) or not tokens[i + 1].startswith("AMP_"):
                raise ValueError(f"Invalid HIT/AMP pair at token {i}")
            amp = float(tokens[i + 1][4:])
            if not math.isfinite(amp) or amp < 0:
                raise ValueError(f"Invalid amplitude at token {i + 1}")
            beat.extend(tokens[i:i + 2])
            pairs += 1
            i += 2
        if pairs > subd:
            raise ValueError(f"SUBD_{subd} has {pairs} HIT/AMP pairs")
        padding_by_beat.append(subd - pairs)
        for _ in range(subd - pairs):
            beat.extend(("HIT_S", "AMP_1.5"))
        beats.append(beat)
    return beats, padding_by_beat


def trim_skeleton(line: str, retained_beats: int) -> tuple[str, int]:
    tokens = line.split()
    if len(tokens) % 3:
        raise ValueError("Skeleton events must be DELAY/HIT/DEV triples")
    kept = []
    time = 0.0
    removed = 0
    for i in range(0, len(tokens), 3):
        delay, hit, dev = tokens[i:i + 3]
        if not delay.startswith("DELAY_") or not HIT.fullmatch(hit) or not dev.startswith("DEV_"):
            raise ValueError(f"Invalid skeleton triple at token {i}")
        amount = float(delay[6:])
        deviation = float(dev[4:])
        if not math.isfinite(amount) or amount < 0 or not math.isfinite(deviation):
            raise ValueError(f"Invalid skeleton timing at token {i}")
        time += amount
        if time < retained_beats:
            kept.extend((delay, hit, dev))
        else:
            removed += 1
    return " ".join(kept), removed


def convert(source: Path, destination: Path, cycle_size: int, report_row: dict | None) -> dict:
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) != 4:
        raise ValueError(f"Expected four physical lines, got {len(lines)}")
    initial = float(lines[0])
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("Invalid initial tempo")
    tempos = lines[1].split()
    if not all(math.isfinite(float(x)) and float(x) > 0 for x in tempos):
        raise ValueError("Invalid per-beat tempo")
    beats, padding_by_beat = parse_variations(lines[3])
    if report_row and int(report_row["total_beats"]) != len(beats):
        raise ValueError("Cycle report beat count disagrees with source")
    if cycle_size < 1:
        raise ValueError("Invalid cycle size")
    retained = len(beats) // cycle_size * cycle_size
    if retained == 0:
        raise ValueError("No complete cycle")
    discarded = len(beats) - retained
    if report_row and int(report_row["leftover_beats_ignored"]) != discarded:
        raise ValueError("Cycle report leftover count disagrees with source")
    if len(tempos) < retained:
        raise ValueError("Insufficient tempos for retained beats")
    skeleton, discarded_events = trim_skeleton(lines[2], retained)
    lengths = [cycle_size] * (retained // cycle_size)
    output = [lines[0], " ".join(tempos[:retained]), " ".join(map(str, lengths)), skeleton,
              " ".join(token for beat in beats[:retained] for token in beat)]
    destination.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
    written = destination.read_text(encoding="utf-8").splitlines()
    assert len(written) == 5
    assert len(written[1].split()) == sum(map(int, written[2].split())) == retained
    assert len(parse_variations(written[4])[0]) == retained
    return {"file": source.name, "source_sha256": sha256(source), "converted_sha256": sha256(destination),
            "cycle_size": cycle_size, "cycles": len(lengths), "source_beats": len(beats),
            "converted_beats": retained, "discarded_trailing_beats": discarded,
            "discarded_trailing_tempos": len(tempos) - retained,
            "discarded_skeleton_events_at_or_after_end": discarded_events,
            "padded_silent_slots": sum(padding_by_beat[:retained])}


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / ".derbake")
    parser.add_argument("--output", type=Path, default=here / ".derbake")
    parser.add_argument("--manifest", type=Path, default=here / "derbake_migration_manifest.json")
    parser.add_argument("--skeletons", type=Path, help="Original skeletons.txt, when available")
    parser.add_argument("--report", type=Path, default=root / "decorated_derbake" / "report.json")
    args = parser.parse_args()
    report_map, report_details = cycle_map_from_report(args.report)
    if args.skeletons:
        cycle_map = cycle_map_from_skeletons(args.skeletons)
        for name, size in cycle_map.items():
            if name in report_map and report_map[name] != size:
                raise ValueError(f"skeletons.txt and report disagree for {name}")
        metadata_source = str(args.skeletons)
    else:
        cycle_map = report_map
        metadata_source = str(args.report)
    sources = sorted(args.source.glob("*.derbake"))
    if not sources:
        raise ValueError(f"No .derbake files in {args.source}")
    if {p.stem for p in sources} != set(cycle_map):
        raise ValueError("Cycle map and source filenames do not match exactly")
    args.output.mkdir(parents=True, exist_ok=True)
    results = [convert(p, args.output / p.name, cycle_map[p.stem], report_details.get(p.stem))
               for p in sources]
    manifest = {"metadata_source": metadata_source, "source_folder": str(args.source),
                "output_folder": str(args.output), "files": results,
                "totals": {key: sum(row[key] for row in results) for key in
                    ("source_beats", "converted_beats", "discarded_trailing_beats",
                     "discarded_trailing_tempos", "discarded_skeleton_events_at_or_after_end",
                     "padded_silent_slots")}}
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(results), **manifest["totals"]}, indent=2))


if __name__ == "__main__":
    main()
