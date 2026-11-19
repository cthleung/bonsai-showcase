# Note formats

Every script reads and writes these formats. The vault is plain markdown with YAML frontmatter, so
grep is enough to find anything and nothing in it needs a database. `demo/sample-vault/` holds
made-up examples.

The vault is sorted PARA style:

- `0 Inbox/win/` and `0 Inbox/mac/` hold new notes, one folder per computer, each with one writer.
- `1 Projects/<Project>/` holds a project note, with `memory/` beside it for the project's facts.
- `2 Areas/<Area>/` holds ongoing areas. `2 Areas/Profile/` holds facts about the user, and
  `2 Areas/People/` holds one note per person who turns up in the user's chats.
- `3 Resources/chats/` holds captured chats, and only capture writes there.
- `4 Archive/` holds old notes. `4 Archive/chats-excluded/<label>/` holds chats moved out of the
  vault (someone else's, or junk).
- `Tasks/` holds one note per task, and `Templates/` holds the note templates.

## Chat note

Path: `3 Resources/chats/<source>/<YYYY>/<YYYY-MM-DD> <title> (<id8>).md`. `<id8>` is the first 8
characters of `chat_id` once anything but letters, digits, `_` and `-` is dropped. The brackets are
round because Obsidian ends a `[[...]]` link at the first `]]`, so a name ending in `]` can't be
linked. Agent sessions use the repo name (claude-code) or the year (codex) as the folder under the
source.

```yaml
---
title: "tomato blight on the lower leaves"
source: chatgpt                 # chatgpt | claude-code | codex, or another capture source
chat_id: Ab12Cd34Ef56Gh78
url: https://chatgpt.com/c/...  # left out for a local agent session
created: 2026-11-16
updated: 2026-11-16
captured: 2026-11-17
model: gpt-5
project: "[[Garden]]"           # optional
archived: true                  # optional: the chat is archived at its source. Left out when it isn't
agent: claude-code              # agent sessions only: the agent that ran it
repo: garden-planner            # claude-code sessions only
message_count: 2
capture: chatgpt-export         # chatgpt-export | specstory | codex-rollout
raw: "raw/chatgpt/Ab12Cd34Ef56Gh78.json"   # under BONSAI_STATE; for an agent session, its source file
visibility: private             # private | creative | career | public
tags: [chat]
---
```

The body opens with `# <title>` and a line saying where the chat came from. Then there's one
`## <n>. You (<date>)` heading per turn (with a `> quote` line when the user quoted the answer
back), the user's words, then `### <Source name>` and the answer. Headings inside an answer are
pushed down to `####` and below. A note is rewritten only when `updated` or `message_count`
changes, and its file name stays fixed by `chat_id`. A rewrite keeps the note's own `visibility`,
`project`, `tags` and `sorted`, so an edit made by hand survives.

The script sets `visibility`, never a model: `private` for a new note, or the repo's visibility
for an agent session (`BONSAI_SPECSTORY_DIRS` and `BONSAI_REPO_VISIBILITY`).

Two more keys mark a chat the user has looked at. `sorted: <YYYY-MM-DD>` is the day the user
checked it. `excluded: <label>` (a person's name, or `junk`) marks a chat moved out to
`4 Archive/chats-excluded/<label>/`, same file name. Nothing reads an excluded chat, and its id goes
on the capture skip list (`BONSAI_STATE/capture-skip.json`), so capture never writes it again.

## Memory note

One fact per note, at its PARA home: `1 Projects/<Project>/memory/` or `2 Areas/<Area>/`.

```yaml
---
name: water-in-the-morning
description: one line, used to decide relevance
type: project                   # user | feedback | project | reference
source: author                  # author | agent | author-approved
date: 2026-06-02                # when the ruling or fact was made
visibility: private
enforced_by: "garden-planner: AGENTS.md"   # optional: repo + repo-relative path
---
The fact. For feedback and project notes, then **Why:** and **How to apply:** lines.
```

## Digest

`0 Inbox/digest <YYYY-MM-DD>.md`, written by the night job. One checkbox per proposal:

```
- [ ] <action> | <target note> | <reason> | source: [[<chat or note>]] ^p01
    > <description> (<type>, <date>, <visibility>)
    > the text that gets written
```

Actions:

- `add`: a new memory note at the target path.
- `merge`: lines added to an existing note.
- `move`: a note moved to another folder.
- `flag`: a contradiction between a new source and an existing note, for the user to settle by hand.
- `profile`: a fact about the user, for `2 Areas/Profile/`.

The quoted lines under a box show what gets written. The `^p01` at the end is an Obsidian block id,
which Obsidian hides. A sidecar in BONSAI_STATE stores each box line exactly as written, and a tick
counts only when one line of the digest equals the stored line apart from the box character, so an
edited or pasted line never applies anything. Ticked boxes are applied at the start of the next
night run, or by `python -m dream.apply`, after a snapshot of the vault, and
`python -m dream.apply --undo` puts back what the last apply changed.

## Task

TaskNotes format: one note per task in `Tasks/`.

```yaml
---
title: "add a frost date check"
status: open
priority: normal
projects: ["[[Garden]]"]
tags: [task]
---
## Story
As the user, I want ... so that ...

## Steps
- [ ]

## Comments
```

## Writing rules

Everything the scripts write into the vault follows one set of plain-language rules: the list is
`WRITING_RULES` in `dream/propose.py`, and `dream/digest.py` checks every digest against it before
it's written. The user's own words, quoted exactly, are the one exception.
