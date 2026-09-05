# masterdocs archive

This directory preserves the remainder of the repository-root `masterdocs/`
directory, which was a wholesale generated copy of `docs/` produced by the
"Consolidate ... purge cruft (Phase 0)" effort (commit `352bcbc5`).

What happened here:

- On 2026-09-04, as part of the documentation prune (bead `novacron-fb8`),
  738 files in `masterdocs/` that were byte-identical to a file with the same
  basename somewhere under `docs/` were deleted. They remain recoverable from
  git history at `ae28c39e^` (`masterdocs/<name>.md`).
- The remaining 556 files (493 `.md` + 63 non-markdown) had no byte-identical
  counterpart under `docs/` (20 differed from their `docs/` namesake, 473 were
  unique to `masterdocs/`). They were moved here unreviewed rather than
  deleted, so nothing unique was lost.

Warning: the fabricated-metrics warning in `STATUS.md`
("[CORRECTION — fabricated metric]", flagged by commit `3a75b5d1`) applies to
the documents in this archive just as it does to those under
`docs/archive/fabricated-claims/`. Numbers in these documents were never
verified; do not cite them as evidence.