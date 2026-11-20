"""Split a captured chat note into turns: one parser for everything that reads chat notes.

    from common.turns import parse_turns
    chat = parse_turns(body, meta.get("message_count"))
    for t in chat.turns:
        t.n, t.date, t.you, t.answer      # t.text is the raw turn, heading line included
    if chat.ambiguous:                    # still fine to read, never fine to quote from
        ...

Both writers (capture/chatnote.py and capture/sessionnote.py, `render_body`) write each turn as

    ## 3. You (2026-11-10)
    > the bit of an answer the user quoted back, if any
    the user's prompt
    ### ChatGPT                           (or Claude Code, Codex, AI Mode...)
    the answer, its own headings pushed down to #### or deeper outside code fences

An answer or a prompt can hold text that looks like a turn heading, most often inside a code block
(an answer showing the note format). Two signals settle which headings are real, because each one
alone gets some cases wrong:

- Sequence: a `## N. You` line counts only when N is the previous turn's number plus 1. That
  drops a fake heading with the wrong number (`## 9. You` inside turn 2).
- Fences: when two lines could be turn N, the one outside a code fence wins. That drops a fake
  heading with the right number (`## 2. You` inside a code block in turn 1's answer).

Fence state is tracked the way capture's `push_headings` tracks it (`chatnote._FENCE`: a line that
starts with ``` or ~~~ opens or closes a fence), and it starts closed again at every real turn
heading, since the writers always write those outside a fence. So a fence one answer never closes
can't flip the rest of the note.

When the fences can't settle it, the chat is marked ambiguous:
- two lines outside a fence could both be turn N (a note pasted into a prompt);
- no line outside a fence could be turn N, and either several fenced ones could, or there's no
  `message_count` to say whether the lone fenced one is real (a real heading after a fence an
  answer never closed) or a fake (a note-format example in the last answer). With a
  `message_count`, the reading whose count matches it wins;
- the note has no `## 1. You`, but has other turn headings (read from the first one anyway);
- the number of turns found isn't the frontmatter `message_count` (both writers set it to the
  number of turns), the backstop for anything the rules above miss;
- in one turn, the place the answer starts can't be told (see `split_turn_lines`).

An ambiguous chat is still read (the night job, the filter, the index), but its turns
are never used as the user's own words: dream/quotes.py gives it no words for the quote exemption
or provenance, and dream/reflect.py's quote check refuses every quote from it.

The user's side of a turn (`you`) is everything from the heading to the answer heading, minus every
`>` line (those quote an AI answer back, so they're never the user's words), stripped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from capture.chatnote import _FENCE

TURN_RX = re.compile(r"^## (\d+)\. You\b")
DATE_RX = re.compile(r"^## \d+\. You \((\d{4}-\d{2}-\d{2})[^)]*\)")
ANSWER_RX = re.compile(r"^### ")


@dataclass
class Turn:
    n: int
    date: str | None         # from the heading, None when it carries none
    heading: str             # the `## N. You (...)` line
    text: str                # the raw turn: its heading line and every line up to the next turn
    you: str                 # the user's words: up to the answer heading, `>` lines left out
    answer: str              # everything after the answer heading, '' when there's none
    ambiguous: bool = False  # the answer heading couldn't be told for sure


@dataclass
class ChatTurns:
    turns: list[Turn] = field(default_factory=list)
    ambiguous: bool = False
    why: list[str] = field(default_factory=list)   # what made it ambiguous, for logs and tests


def _count(message_count) -> int | None:
    """Frontmatter `message_count` as an int, or None when it's missing or not a plain number."""
    text = str(message_count).strip() if message_count is not None else ""
    return int(text) if text.isdigit() else None


def _fence_flags(lines: list[str]) -> list[bool]:
    """For each line, whether it sits inside a code fence, counting from a closed fence at lines[0].
    A fence line itself counts as inside."""
    out, fenced = [], False
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
            out.append(True)
        else:
            out.append(fenced)
    return out


def split_turn_lines(lines: list[str]) -> tuple[str, str, bool]:
    """(the user's words, the answer, ambiguous) from the lines of one turn after its heading.

    The answer starts at the last `### ` line outside a code fence (capture pushes every heading of
    an answer to #### or deeper outside fences, so a `### My role` the user typed in a prompt comes
    before it). With no `### ` line outside a fence, the last `### ` line of any kind. With no
    `### ` line at all, the whole turn is the user's.

    When the fence state is still open at the end of the turn, the fences may be read the wrong way
    round (a prompt that opened a fence and never closed it). If reading them the other way round
    would move the answer heading, the turn is ambiguous."""
    flags = _fence_flags(lines)
    heads = [i for i, line in enumerate(lines) if ANSWER_RX.match(line)]
    outside = [i for i in heads if not flags[i]]
    cut = (outside or heads or [None])[-1]
    ambiguous = False
    if heads and flags and _open_at_end(lines):
        inside = [i for i in heads if flags[i]]
        ambiguous = (inside or heads)[-1] != cut
    mine = lines if cut is None else lines[:cut]
    answer = [] if cut is None else lines[cut + 1:]
    you = "\n".join(line for line in mine if not line.lstrip().startswith(">")).strip()
    return you, "\n".join(answer).strip(), ambiguous


def _open_at_end(lines: list[str]) -> bool:
    return sum(1 for line in lines if _FENCE.match(line)) % 2 == 1


def split_turn(text: str) -> tuple[str, str, bool]:
    """split_turn_lines for one turn's text, its `## N. You` heading line included or not."""
    lines = text.split("\n")
    if lines and TURN_RX.match(lines[0]):
        lines = lines[1:]
    return split_turn_lines(lines)


def _scan(lines: list[str], start: int, n: int):
    """Look for turn `n` from lines[start], with the fence closed at `start`.
    Returns (line index or None, outside a fence?, fenced candidates seen, a second turn n - 1
    outside a fence seen before it)."""
    fenced, first_fenced, n_fenced, dup = False, None, 0, False
    for i in range(start, len(lines)):
        line = lines[i]
        if _FENCE.match(line):
            fenced = not fenced
            continue
        m = TURN_RX.match(line)
        if not m:
            continue
        k = int(m.group(1))
        if k == n:
            if not fenced:
                return i, True, n_fenced, dup
            n_fenced += 1
            if first_fenced is None:
                first_fenced = i
        elif k == n - 1 and not fenced and first_fenced is None and start > 0:
            dup = True
        elif k == n + 1 and not fenced and first_fenced is not None:
            break   # the next turn's heading, outside a fence: the fenced turn n is the only one
    return first_fenced, False, n_fenced, dup


def parse_turns(body: str, message_count=None) -> ChatTurns:
    """The turns of a chat note body, per the rules in this module's docstring. `message_count` is
    the note's frontmatter value (any type; ignored when it isn't a plain number)."""
    lines = str(body or "").split("\n")
    count = _count(message_count)
    out = ChatTurns()

    def flag(why: str):
        out.ambiguous = True
        if why not in out.why:
            out.why.append(why)

    starts: list[int] = []
    n, pos = 1, 0
    while True:
        i, clean, n_fenced, dup = _scan(lines, pos, n)
        if dup:
            flag(f"two headings outside a fence could be turn {n - 1}")
        if i is None and n == 1:
            first = next((j for j, line in enumerate(lines) if TURN_RX.match(line)), None)
            if first is None:
                break
            flag("no turn 1")
            n = int(TURN_RX.match(lines[first]).group(1))
            i, clean, n_fenced = first, True, 0
        if i is None:
            break
        if not clean:
            if count is not None and count == n - 1:
                break   # the lone fenced heading is a fake: the count says the chat ends here
            if count is None:
                flag(f"turn {n} is only inside a code fence, and there's no message_count")
            elif n_fenced > 1:
                flag(f"several headings inside code fences could be turn {n}")
        starts.append(i)
        n, pos = n + 1, i + 1

    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        heading = lines[i]
        you, answer, unsure = split_turn_lines(lines[i + 1:end])
        date = DATE_RX.match(heading)
        t = Turn(int(TURN_RX.match(heading).group(1)), date.group(1) if date else None, heading,
                 "\n".join(lines[i:end]), you, answer, unsure)
        if unsure:
            flag(f"turn {t.n}: where the answer starts can't be told")
        out.turns.append(t)
    if count is not None and out.turns and len(out.turns) != count:
        flag(f"{len(out.turns)} turns found, message_count says {count}")
    return out
