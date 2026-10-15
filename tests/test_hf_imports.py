"""Guard the direct-import contract used by Transformers remote-code loading."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "ldm"


def relative_imports(module: str) -> set[str]:
    tree = ast.parse((ROOT / f"{module}.py").read_text(encoding="utf-8"))
    return {node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module}


class HFImportTests(unittest.TestCase):
    def test_auto_class_modules_directly_import_transitive_dependencies(self):
        # Transformers copies only direct imports into its dynamic-module cache.
        for entry in ("modeling_ldm", "processing_ldm", "tokenization_ldm"):
            direct = relative_imports(entry)
            seen = set()
            pending = list(direct)
            while pending:
                module = pending.pop()
                if module in seen:
                    continue
                seen.add(module)
                pending.extend(relative_imports(module) - seen)
            with self.subTest(entry=entry):
                self.assertEqual(seen, direct)


if __name__ == "__main__":
    unittest.main()
