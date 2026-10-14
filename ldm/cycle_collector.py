"""Select complete valid cycles from normally sampled atomic tokens."""

from __future__ import annotations

from .schema_ldm import RhythmSchema


class IncompleteGenerationError(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(f"Generated {report['accepted_cycles']} of {report['requested_cycles']} valid cycles; "
                         f"discarded {report['discarded_cycles']} candidates")


class CycleCollector:
    def __init__(self, schema: RhythmSchema, requested_cycles: int):
        if isinstance(requested_cycles, bool) or not isinstance(requested_cycles, int) or requested_cycles < 1:
            raise ValueError("num_cycles must be a positive integer")
        self.schema = schema
        self.requested_cycles = requested_cycles
        self.accepted_ids: list[int] = []
        self.candidate_ids: list[int] = []
        self.candidate_atoms: list[str] = []
        self.discarded: list[dict] = []
        self.orphan_tokens = 0
        self.sampled_tokens = 0
        self._accepted_count = 0

    def _discard(self, reason: str) -> None:
        if self.candidate_ids:
            self.discarded.append({"sampled_token": self.sampled_tokens, "reason": reason,
                                   "candidate_tokens": len(self.candidate_ids)})
        self.candidate_ids = []
        self.candidate_atoms = []

    def append(self, token_id: int, atom: str) -> bool:
        self.sampled_tokens += 1
        if atom == "<SOC>":
            if self.candidate_ids:
                self._discard("Nested <SOC>; restarted at latest <SOC>")
            self.candidate_ids = [token_id]
            self.candidate_atoms = [atom]
            return False
        if not self.candidate_ids:
            self.orphan_tokens += 1
            return False
        self.candidate_ids.append(token_id)
        self.candidate_atoms.append(atom)
        if atom != "<EOC>":
            return False
        try:
            self.schema.validate_candidate(self.candidate_atoms)
        except ValueError as exc:
            self._discard(str(exc))
            return False
        self.accepted_ids.extend(self.candidate_ids)
        self._accepted_count += 1
        self.candidate_ids = []
        self.candidate_atoms = []
        return self.accepted_cycles == self.requested_cycles

    @property
    def accepted_cycles(self) -> int:
        return self._accepted_count

    def report(self) -> dict:
        return {"requested_cycles": self.requested_cycles,
                "accepted_cycles": self._accepted_count,
                "discarded_cycles": len(self.discarded), "discarded": list(self.discarded),
                "orphan_tokens": self.orphan_tokens, "sampled_tokens": self.sampled_tokens}

    def finish(self) -> tuple[list[int], dict]:
        if self.candidate_ids:
            self._discard("Unfinished cycle at token cap")
        report = self.report()
        if report["accepted_cycles"] != self.requested_cycles:
            raise IncompleteGenerationError(report)
        return self.accepted_ids, report
