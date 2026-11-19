"""Note templates and the vault's folder list.

The files in TEMPLATES are written into the vault's `Templates/` folder for Obsidian's Templates
plugin, so they use its placeholders ({{title}}, {{date:YYYY-MM-DD}}). The note formats follow
docs/note-format.md.
"""
from __future__ import annotations

FOLDERS = (
    "0 Inbox/win",
    "0 Inbox/mac",
    "1 Projects",
    "2 Areas",
    "3 Resources",
    "4 Archive",
    "Tasks",
    "Templates",
)

TEMPLATES_DIR = "Templates"

MEMORY = """\
---
name: "{{title}}"
description: one line, used to decide relevance
type: feedback
source: author
date: "{{date:YYYY-MM-DD}}"
visibility: private
---
The fact, in one or two lines.

**Why:** what happened or what the user said that makes this a rule.

**How to apply:** what to do differently because of it.
"""

PROJECT = """\
---
title: "{{title}}"
description: one line on what the project is
status: active
repo: ""
visibility: private
tags: [project]
---
# {{title}}

**Goal:** _the user fills this in._

**Status:** where it stands, with a date.

**Repo:** the repo link, or blank.

## Links
- Plan:
- Decision log:

## Memory
Project facts and rules go in `memory/` beside this note, one fact per note.

## Tasks
Task notes in `Tasks/` link here with `projects: ["[[{{title}}]]"]`.
"""

TASK = """\
---
title: "{{title}}"
status: open
priority: normal
projects: ["[[]]"]
tags: [task]
---
## Story
As the user, I want ... so that ...

## Steps
- [ ]

## Comments
"""

AREA = """\
---
title: "{{title}}"
description: one line on what this area covers
visibility: private
tags: [area]
---
# {{title}}

What this area covers and what doesn't belong here.

Notes in this folder are one fact each, in the memory note format.
"""

TEMPLATES = {
    "Memory.md": MEMORY,
    "Project.md": PROJECT,
    "Task.md": TASK,
    "Area.md": AREA,
}
