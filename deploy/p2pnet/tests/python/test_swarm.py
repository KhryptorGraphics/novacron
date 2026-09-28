#!/usr/bin/python3
"""Behavioral boundary tests for swarm torrent piece sizing."""
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "libexec" / "swarm_make.py"
SPEC = importlib.util.spec_from_file_location("swarm_make", MODULE_PATH)
swarm_make = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(swarm_make)


class PieceSizeTests(unittest.TestCase):
    def test_64_gib_uses_16_mib_pieces(self):
        self.assertEqual(swarm_make.piece_size_for(64 * swarm_make.GIB), 16 * swarm_make.MIB)

    def test_above_64_gib_uses_32_mib_pieces(self):
        self.assertEqual(swarm_make.piece_size_for(64 * swarm_make.GIB + 1), 32 * swarm_make.MIB)

    def test_above_256_gib_uses_64_mib_pieces(self):
        self.assertEqual(swarm_make.piece_size_for(256 * swarm_make.GIB + 1), 64 * swarm_make.MIB)


if __name__ == "__main__":
    unittest.main()
