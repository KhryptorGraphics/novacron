# Issue tracker: GitHub (inbox) + beads (work)

Two trackers, deliberate split:

- **GitHub Issues** — the external inbox. Anything reported from outside lands here.
  `/triage` operates here and nowhere else.
- **beads** (`bd`, `.beads/beads.db`) — the working tracker. Accepted work is decomposed
  into beads; `/to-tickets`, `/code-review`, `/wayfinder` and AFK agents operate here.

GitHub Issues is currently empty while beads holds the live work. That is expected, not drift.

## Resolving an ID

Route by shape — every skill hits this first:

| ID shape                         | Tracker | Command                       |
| -------------------------------- | ------- | ----------------------------- |
| `novacron-7q6`, `novacron-7q6.1` | beads   | `bd show <id>`                |
| `#42`, `gh-42`, a github.com URL | GitHub  | `gh issue view 42 --comments` |

A bead carrying `external-ref: gh-42` is the same work seen from the other side.

## When a skill says "publish to the issue tracker"

Create a **bead**: `bd create "<title>" -d "<description>" -p P1 -t task`.

The sole exception is `/triage`, which only ever writes to GitHub.

## When a skill says "fetch the relevant ticket"

Route by ID shape (table above).

## GitHub inbox conventions — `/triage` only

- **List the queue**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`
- **Read**: `gh issue view <n> --comments`
- **Comment**: `gh issue comment <n> --body "..."`
- **Label**: `gh issue edit <n> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <n> --comment "..."`

`gh` infers the repo from `git remote -v` when run inside the clone.

**PRs as a request surface: no.** _(Set to `yes` if this repo should treat external PRs as
feature requests; `/triage` reads this flag.)_

## Handoff: GitHub inbox → beads

When `/triage` lands a GitHub issue on `ready-for-agent` or `ready-for-human`:

1. Create the bead, linked back: `bd create "<title>" -d "<description>" --external-ref gh-<n>`
2. Post the bead ID onto the issue: ``gh issue comment <n> --body "Tracked as `<bead-id>`."``
3. Leave the GitHub issue **open** — it is the reporter's window into the work.

When the bead closes, close the GitHub issue with a pointer:
``gh issue close <n> --comment "Done in `<bead-id>`."``

Work that originates internally skips GitHub entirely — create the bead directly.

## beads conventions

- **Create**: `bd create "<title>" -d "..." -p P1 -t task` (types: `bd types`; priorities P0–P4, 0 highest)
- **Quick capture**: `bd q "<title>"` — creates and prints only the ID
- **List / search**: `bd list`, `bd search "<query>"`, `bd list -l <label>`
- **Show**: `bd show <id>`
- **Update**: `bd update <id> -s in_progress` / `--add-label` / `--remove-label` / `-a <who>` / `-p P0`
- **Comment**: `bd comments <id>` to read, `bd comments add <id> "..."` to write
- **Close / reopen**: `bd close <id>`, `bd reopen <id>`
- **Dependencies**: `bd dep add <blocked> <blocker>` (or `bd dep <blocker> --blocks <blocked>`);
  inspect with `bd dep list <id>`, `bd graph`, `bd dep cycles`
- **Ready work**: `bd ready` — open/in-progress issues with no open blockers
- **Sync to git**: `bd sync` exports to JSONL. The daemon usually handles this.

## Wayfinding operations

Used by `/wayfinder`. Runs in **beads** — it has first-class dependencies and a native
frontier query, so none of GitHub's dependency-API workarounds are needed.

- **Map**: a parent bead, `bd create "<effort>" -t epic`, holding the
  Notes / Decisions-so-far / Fog body in its description.
- **Child ticket**: `bd create "<question>" --parent <map-id> -l wayfinder:<type>` where
  `<type>` is `research` / `prototype` / `grilling` / `task`. List them with `bd children <map-id>`.
- **Blocking**: `bd dep add <child> <blocker>`. `bd` enforces it — a ticket is unblocked when
  every blocker is closed.
- **Frontier query**: `bd ready --parent <map-id> -u` — unblocked, open, unassigned
  descendants of the map; first wins.
- **Claim**: `bd update <id> --claim` — atomic (sets assignee to you and status to
  `in_progress`, fails if already claimed). The session's first write.
- **Resolve**: `bd comments add <id> "<answer>"`, then `bd close <id>`, then append a context
  pointer to the map's Decisions-so-far: `bd update <map-id> --append-notes "..."`.
