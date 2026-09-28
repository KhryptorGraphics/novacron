#!/usr/bin/python3
import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "libexec" / "zrepl_retention.py"
SPEC = importlib.util.spec_from_file_location("zrepl_retention", MODULE)
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.names = [
            f"tank/vms@p2pnet-2026092{day}T120000Z"
            for day in (1, 2, 3, 4, 5)
        ]

    def test_keeps_newest_overall_even_with_zero_counts(self):
        self.assertEqual(
            set(retention.names_to_destroy(self.names, 0, 0, set())),
            set(self.names[:-1]),
        )

    def test_keeps_last_and_newest_of_each_daily_window(self):
        names = [
            "tank/vms@p2pnet-20260920T010000Z",
            "tank/vms@p2pnet-20260920T230000Z",
            "tank/vms@p2pnet-20260921T120000Z",
            "tank/vms@p2pnet-20260922T120000Z",
            "tank/vms@p2pnet-20260923T120000Z",
        ]
        self.assertEqual(
            set(retention.names_to_destroy(names, 2, 3, set())),
            {"tank/vms@p2pnet-20260920T010000Z", "tank/vms@p2pnet-20260920T230000Z"},
        )

    def test_protected_name_survives_and_unrecognized_names_are_ignored(self):
        protected = self.names[0]
        destroy = retention.names_to_destroy(
            self.names + ["tank/vms@manual-snapshot"], 1, 0, {protected}
        )
        self.assertNotIn(protected, destroy)
        self.assertNotIn("tank/vms@manual-snapshot", destroy)
        self.assertEqual(len(destroy), 3)

    def test_newest_snapshot_is_never_deleted(self):
        destroy = retention.names_to_destroy(self.names, 2, 1, set())
        self.assertNotIn(self.names[-1], destroy)


if __name__ == "__main__":
    unittest.main()
