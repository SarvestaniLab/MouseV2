"""Frozen publication inputs must reject checkpoint and geometry drift."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_figure2_freeze import verify


class Figure2FreezeTests(unittest.TestCase):
    def test_frozen_inputs_verify(self):
        freeze = verify(ROOT)
        self.assertEqual(freeze["runs"]["primary"]["seed"], 13)
        self.assertEqual(freeze["runs"]["primary"]["iterations"], 2000)

    def test_changed_checkpoint_and_geometry_are_rejected(self):
        freeze = verify(ROOT)
        primary = freeze["runs"]["primary"]
        for changed in [primary["directory"] + "/" + primary["final_checkpoint"],
                        freeze["rendering_surface"]]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for name in list(freeze["files"]) + ["data/publication/figure2/freeze.json"]:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, target)
                with (root / changed).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "input changed"):
                    verify(root)

    def test_extra_checkpoint_is_rejected(self):
        freeze = verify(ROOT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in list(freeze["files"]) + ["data/publication/figure2/freeze.json"]:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            (root / freeze["runs"]["primary"]["directory"] / "y-new-09999.data").write_text("0")
            with self.assertRaisesRegex(ValueError, "exactly its selected checkpoint"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
