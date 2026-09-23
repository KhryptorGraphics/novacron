-- Migration: cluster_nodes_drain
-- Created: 2026-09-22
-- Direction: DOWN
-- Description: Reverts the drain lifecycle column. The table itself is left
-- in place: it was introduced here, but dropping it on reversal would also
-- destroy any future columns a LATER migration may have added on top -- a
-- down migration must only undo what its own up added.

ALTER TABLE cluster_nodes DROP COLUMN IF EXISTS drain_state;
