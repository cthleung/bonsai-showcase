"""Draw the README diagrams as plain SVG files in docs/img/.

    python docs/diagrams/build.py

Needs Pillow (`pip install pillow`, not in requirements.txt: only these drawings use it). Text in an
SVG doesn't wrap by itself, so every line is measured and broken here. It's measured with Verdana,
which is wider than the fonts most screens actually use (SF, Segoe UI, Helvetica), so a line never
runs past its box on any computer. Edit the words in CREW and TRUST below and run it again.

GitHub shows an SVG in a README as an image, so these use no scripts, no web fonts and no
foreignObject, and each one has its own light background.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from PIL import ImageFont

OUT = Path(__file__).resolve().parents[1] / "img"
FONT_DIRS = [Path("/System/Library/Fonts/Supplemental"), Path("C:/Windows/Fonts"),
             Path("/usr/share/fonts/truetype/msttcorefonts")]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

INK = "#1F2328"
SOFT = "#59636E"
FAINT = "#8C959F"
PAPER = "#FBF9F4"
EDGE = "#E4DED0"

# kind: (accent, fill, tag)
KINDS = {
    "claude": ("#B4472A", "#FCEEE8", "AI · Claude"),
    "small": ("#B4472A", "#FCEEE8", "AI · a smaller Claude"),
    "cold": ("#B4472A", "#FCEEE8", "AI · Claude, a separate call"),
    "local": ("#5E4BA8", "#F1EEFA", "AI · on your own GPU"),
    "code": ("#23704F", "#E8F4EE", "Plain code · no AI"),
    "you": ("#9A5B00", "#FFF3D6", "Your call"),
    "agents": ("#245FA8", "#E8F0FB", "Your everyday agents"),
    "cloudagents": ("#245FA8", "#E8F0FB", "Agents in the cloud"),
    "storage": ("#7A6440", "#FBF5EA", "Cloud storage"),
    "person": ("#9A5B00", "#FFF3D6", "You"),
}


def _font_file(bold: bool) -> Path:
    name = "Verdana Bold.ttf" if bold else "Verdana.ttf"
    alt = "verdanab.ttf" if bold else "verdana.ttf"
    for d in FONT_DIRS:
        for n in (name, alt, alt.capitalize()):
            if (d / n).exists():
                return d / n
    raise SystemExit("Verdana not found: it's used to measure text so it never overflows")


_FONTS: dict = {}


def width(text: str, size: float, bold: bool = False) -> float:
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(str(_font_file(bold)), 200)
    return _FONTS[key].getlength(text) * size / 200


def wrap(text: str, size: float, room: float, bold: bool = False) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and width(trial, size, bold) > room:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def text(x, y, s, size, color=INK, bold=False, anchor="start", spacing=None):
    bold_attr = ' font-weight="700"' if bold else ""
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}"{bold_attr}{ls} '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def lines_at(x, y, lines, size, lead, color=INK, bold=False, anchor="start"):
    return "".join(text(x, y + i * lead, ln, size, color, bold, anchor) for i, ln in enumerate(lines))


# --- icons, drawn as paths so no font is needed -------------------------------------------------

def icon(kind: str, cx: float, cy: float) -> str:
    accent, _, _ = KINDS[kind]
    ring = f'<circle cx="{cx}" cy="{cy}" r="13" fill="#FFFFFF" stroke="{accent}" stroke-width="1.6"/>'
    if kind in ("claude", "small", "cold", "local"):
        s = 7.5
        d = (f"M{cx} {cy - s} Q{cx + 1.6} {cy - 1.6} {cx + s} {cy} Q{cx + 1.6} {cy + 1.6} {cx} {cy + s} "
             f"Q{cx - 1.6} {cy + 1.6} {cx - s} {cy} Q{cx - 1.6} {cy - 1.6} {cx} {cy - s}Z")
        return ring + f'<path d="{d}" fill="{accent}"/>'
    if kind == "code":
        return ring + (f'<path d="M{cx - 3} {cy - 5} L{cx - 7.5} {cy} L{cx - 3} {cy + 5} M{cx + 3} {cy - 5} '
                       f'L{cx + 7.5} {cy} L{cx + 3} {cy + 5}" fill="none" stroke="{accent}" '
                       f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    if kind in ("you", "person"):
        return ring + (f'<circle cx="{cx}" cy="{cy - 3.2}" r="3.4" fill="{accent}"/>'
                       f'<path d="M{cx - 6.5} {cy + 7.5} Q{cx} {cy - 1.5} {cx + 6.5} {cy + 7.5}Z" fill="{accent}"/>')
    if kind == "storage":
        return ring + (f'<path d="M{cx - 7} {cy + 4.5} H{cx + 6.5} A3.6 3.6 0 0 0 {cx + 5.5} {cy - 2.6} '
                       f'A5 5 0 0 0 {cx - 3.5} {cy - 3} A3.8 3.8 0 0 0 {cx - 7} {cy + 4.5}Z" fill="{accent}"/>')
    if kind in ("agents", "cloudagents"):
        return ring + (f'<rect x="{cx - 6.5}" y="{cy - 5}" width="13" height="8.5" rx="1.5" fill="none" '
                       f'stroke="{accent}" stroke-width="1.8"/>'
                       f'<path d="M{cx - 8.5} {cy + 6} H{cx + 8.5}" stroke="{accent}" stroke-width="1.8" '
                       f'stroke-linecap="round"/>')
    return ring


# --- a card -------------------------------------------------------------------------------------

TITLE, TAG, BODY, LEAD = 16, 11.5, 14, 19.5


def card(x, y, w, kind, title, body, min_h: float = 0) -> tuple[str, float]:
    accent, fill, tag = KINDS[kind]
    pad = 16
    room = w - pad * 2 - 34
    t_lines = wrap(title, TITLE, room, bold=True)
    b_lines = wrap(body, BODY, w - pad * 2)
    top = y + pad
    h = max(min_h, pad + len(t_lines) * 20 + 4 + 16 + 8 + len(b_lines) * LEAD + pad - 6)
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h:.1f}" rx="12" fill="{fill}" stroke="{accent}" '
           f'stroke-opacity="0.35" stroke-width="1.2"/>',
           icon(kind, x + pad + 13, top + 11),
           lines_at(x + pad + 34, top + 15, t_lines, TITLE, 20, INK, bold=True)]
    ty = top + len(t_lines) * 20 + 14
    out.append(text(x + pad + 34, ty, tag.upper(), TAG, accent, bold=True, spacing="0.6"))
    out.append(lines_at(x + pad, ty + 26, b_lines, BODY, LEAD, "#2D333B"))
    return "".join(out), h


def arrow_down(cx, y1, y2, color=FAINT) -> str:
    return (f'<path d="M{cx} {y1} V{y2 - 6}" stroke="{color}" stroke-width="1.8"/>'
            f'<path d="M{cx - 5} {y2 - 8} L{cx} {y2} L{cx + 5} {y2 - 8}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>')


def svg(w, h, title, desc, body) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h:.0f}" viewBox="0 0 {w} {h:.0f}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">'
            f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1:.0f}" rx="20" fill="{PAPER}" stroke="{EDGE}"/>'
            f'{body}</svg>\n')


LEGEND = {"you": "You"}


def legend(x, y, kinds, max_w) -> tuple[str, float]:
    out, cx, cy = [], x, y
    for k in kinds:
        label = LEGEND.get(k, KINDS[k][2])
        w = 30 + width(label, 13) + 22
        if cx + w > x + max_w:
            cx, cy = x, cy + 34
        out.append(icon(k, cx + 13, cy))
        out.append(text(cx + 32, cy + 4.5, label, 13, SOFT))
        cx += w
    return "".join(out), cy - y + 20


# --- 1. the crew --------------------------------------------------------------------------------

CREW_TITLE = "One vault, a whole crew"
CREW_SUB = ("Bonsai runs a crew of small helpers, each with one job. Some are AI models, some are plain "
            "code that checks the models' work, and you sign off on anything that changes.")
CREW = [
    ("ALL DAY", "Getting your chats in, and letting your agents find them.", [
        ("code", "Capture", "Every hour, copies new chats from Claude Code, Codex and Copilot into the vault. "
                            "ChatGPT comes in from its export. Keys and passwords are scrubbed out first."),
        ("code", "Index builder", "Re-reads the notes that changed, so search always matches the vault."),
        ("local", "Search helpers", "Two small models find notes by meaning and rank the best matches. "
                                    "They run on your computer, so search never leaves it."),
        ("code", "Gateway", "The one door your agents use: search, read a note, see what's new. "
                            "Read only, and off-limits notes stay off limits."),
        ("agents", "Claude Code, Codex, Copilot", "Look things up through the gateway while they work, and "
                                                  "leave notes in their own inbox folder."),
    ]),
    ("EVERY NIGHT", "Turning the day's chats into memory, with you deciding.", [
        ("claude", "Night proposer", "Reads what's new and suggests memory changes. Tools off, answers in a "
                                     "fixed format, and a hard dollar cap per night."),
        ("claude", "Reworder", "Rewrites any suggestion that breaks the house writing rules."),
        ("code", "Digest writer", "Turns the suggestions into a morning checklist, and checks it before "
                                  "it's written."),
        ("you", "You, in the morning", "Tick the boxes you agree with, and leave the rest."),
        ("code", "Apply", "Next night, first thing: takes a snapshot, then makes only the ticked changes. "
                          "Any change can be undone."),
    ]),
    ("WHEN YOU ASK FOR A PROFILE", "Replaying old chats to learn who you are.", [
        ("small", "Chat scorer", "Scores every chat: how much does it say about you, and is it even yours?"),
        ("claude", "Claim extractor", "Replays one month at a time and pulls out claims, each backed by "
                                      "your exact words."),
        ("code", "Quote checker", "Throws out any claim whose quote isn't word for word in a message "
                                  "you wrote."),
        ("cold", "Fact-checker", "Never sees the extractor's work, only the claims and your messages. "
                                 "Marks each one supported, unsupported or invented."),
        ("claude", "Question writer", "Turns the hunches into a few questions for you, quoting you back."),
        ("you", "You", "Answer the questions. A hunch never turns into a fact on its own."),
    ]),
]
CREW_FOOT = "Eight AI jobs in all, and none of them can change a note on its own."
CREW_DESC = ("A diagram of everything that works on the Bonsai vault, in three columns. All day: capture "
             "copies chats in, an index builder and two local search models keep search current, and a "
             "read-only gateway lets Claude Code, Codex and Copilot search the vault. Every night: Claude "
             "proposes memory changes, a reworder fixes the wording, plain code writes a morning "
             "checklist, you tick boxes, and the next night only the ticked changes are applied after a "
             "snapshot. For a profile: a smaller Claude scores chats, Claude extracts claims with quotes, "
             "plain code checks every quote word for word, a separate Claude fact-checks, Claude writes "
             "questions, and you answer them. None of the AI can change a note on its own.")


def crew() -> str:
    W, M, G = 1000, 32, 20
    colw = (W - 2 * M - 2 * G) / 3
    parts = [text(M, 62, CREW_TITLE, 30, INK, bold=True)]
    sub = wrap(CREW_SUB, 15.5, W - 2 * M)
    parts.append(lines_at(M, 92, sub, 15.5, 22, SOFT))
    y = 92 + len(sub) * 22 + 14
    leg, lh = legend(M, y + 6, ["claude", "local", "code", "you", "agents"], W - 2 * M)
    parts.append(leg)
    y += lh + 26
    bottoms = []
    for i, (head, blurb, cards) in enumerate(CREW):
        x = M + i * (colw + G)
        parts.append(f'<rect x="{x}" y="{y}" width="{colw:.1f}" height="4" rx="2" fill="#3B4A3F"/>')
        parts.append(text(x, y + 28, head, 13, "#3B4A3F", bold=True, spacing="1"))
        bl = wrap(blurb, 13.5, colw)
        parts.append(lines_at(x, y + 50, bl, 13.5, 19, SOFT))
        cy = y + 50 + len(bl) * 19 + 8
        for j, (kind, title, body) in enumerate(cards):
            if j:
                parts.append(arrow_down(x + colw / 2, cy - 22, cy - 2))
            s, h = card(x, cy, colw, kind, title, body)
            parts.append(s)
            cy += h + 24
        bottoms.append((x, cy - 24))
    vy = max(b for _, b in bottoms) + 30
    for x, b in bottoms:
        parts.append(arrow_down(x + colw / 2, b + 4, vy - 2, "#3B4A3F"))
    vh = 92
    parts.append(f'<rect x="{M}" y="{vy}" width="{W - 2 * M}" height="{vh}" rx="14" fill="#3B4A3F"/>')
    parts.append(text(M + 24, vy + 38, "The vault", 20, "#FFFFFF", bold=True))
    vt = wrap("A plain folder of markdown notes, synced by Google Drive. Chats come in through capture, "
              "your agents write only to their own inbox folders, and every other change goes through "
              "Apply.", 14.5, W - 2 * M - 48 - 120)
    parts.append(lines_at(M + 24 + 120, vy + 36, vt, 14.5, 21, "#E6ECE7"))
    fy = vy + vh + 40
    parts.append(text(W / 2, fy, CREW_FOOT, 15, "#3B4A3F", bold=True, anchor="middle"))
    return svg(W, fy + 30, CREW_TITLE, CREW_DESC, "".join(parts))


# --- 2. how a claim earns its place -------------------------------------------------------------

TRUST_TITLE = "How a claim about you earns its place"
TRUST_SUB = ("Models make up a lot about people, and their own self-checks miss it. So every claim has to "
             "get past plain code and a second, separate model before you ever see it.")
TRUST = [
    ("claude", "1. Claim extractor", "Reads one month of your chats and writes claims about you, each "
                                     "with the exact words of yours it's based on.", None),
    ("code", "2. Quote checker", "Is every quote word for word in a message you wrote, in the chat it "
                                 "names? One bad quote and the whole claim goes.", "Quote not found"),
    ("code", "3. Rule checks", "A hunch needs two or more separate chats, and can't guess at motives "
                               "or feelings. Nothing you've rejected before, and no topic you've asked "
                               "it to leave out.",
     "Breaks a rule"),
    ("cold", "4. Fact-checker", "A separate call that never saw step 1. It sees the claims and your "
                                "messages, and marks each one supported, unsupported or invented.",
     "Not supported"),
]
TRUST_END = [
    ("code", "Facts", "Written up by code, not by a model, with a link to every chat behind them."),
    ("claude", "Questions", "Hunches come back as a few questions that quote you, checked again so none "
                            "of them lead you."),
]
TRUST_YOU = ("you", "You", "Everything lands in a drafts folder for you to read. The job is never "
                           "allowed to write into your profile.")
TRUST_NOTE = "Code sets each claim's date, confidence and privacy label. A model never does."
TRUST_DESC = ("A flow diagram of the profile checks. Step 1, Claude extracts claims about you, each with "
              "your exact words. Step 2, plain code drops any claim whose quote isn't word for word in "
              "your own message. Step 3, plain code drops hunches with fewer than two chats behind them, "
              "guesses at motives or feelings, and anything on your rejected or private lists. Step 4, a "
              "separate Claude call that never saw step 1 drops anything it can't support. What survives "
              "becomes facts written up by code and questions for you, and it all goes to a drafts "
              "folder, never straight into your profile. Code, not a model, sets each claim's date, "
              "confidence and privacy label.")


def trust() -> str:
    W, M = 1000, 32
    parts = [text(M, 62, TRUST_TITLE, 30, INK, bold=True)]
    sub = wrap(TRUST_SUB, 15.5, W - 2 * M)
    parts.append(lines_at(M, 92, sub, 15.5, 22, SOFT))
    y = 92 + len(sub) * 22 + 22
    cw, cx = 560, M + 40
    ex = cx + cw + 70
    for i, (kind, title, body, drop) in enumerate(TRUST):
        if i:
            parts.append(arrow_down(cx + cw / 2, y - 26, y - 2))
        s, h = card(cx, y, cw, kind, title, body)
        parts.append(s)
        if drop:
            my = y + h / 2
            parts.append(f'<path d="M{cx + cw + 4} {my} H{ex - 8}" stroke="#B9B2A3" stroke-width="1.8" '
                         f'stroke-dasharray="5 4"/>')
            parts.append(f'<path d="M{ex - 14} {my - 5} L{ex - 6} {my} L{ex - 14} {my + 5}" fill="none" '
                         f'stroke="#B9B2A3" stroke-width="1.8" stroke-linecap="round"/>')
            parts.append(f'<rect x="{ex}" y="{my - 22}" width="{W - M - ex - 8}" height="44" rx="22" '
                         f'fill="#F2EFE8" stroke="#D9D2C3"/>')
            parts.append(text(ex + 20, my - 1, drop, 14, "#6E675A", bold=True))
            parts.append(text(ex + 20, my + 15, "dropped, and the reason kept", 12, "#8A8273"))
        y += h + 30
    pair_w = (cw - 20) / 2
    split_y = y
    parts.append(f'<path d="M{cx + cw / 2} {y - 30} V{y - 16} M{cx + pair_w / 2} {y - 16} H{cx + cw - pair_w / 2}" '
                 f'stroke="{FAINT}" stroke-width="1.8" fill="none"/>')
    hs = []
    for j, (kind, title, body) in enumerate(TRUST_END):
        x = cx + j * (pair_w + 20)
        parts.append(arrow_down(x + pair_w / 2, y - 16, y - 2))
        s, h = card(x, y, pair_w, kind, title, body)
        parts.append(s)
        hs.append(h)
    y = split_y + max(hs) + 30
    parts.append(f'<path d="M{cx + pair_w / 2} {split_y + hs[0] + 4} V{y - 16} M{cx + cw - pair_w / 2} '
                 f'{split_y + hs[1] + 4} V{y - 16} M{cx + pair_w / 2} {y - 16} H{cx + cw - pair_w / 2}" '
                 f'stroke="{FAINT}" stroke-width="1.8" fill="none"/>')
    parts.append(arrow_down(cx + cw / 2, y - 16, y - 2))
    s, h = card(cx, y, cw, *TRUST_YOU)
    parts.append(s)
    y += h + 44
    parts.append(text(W / 2, y, TRUST_NOTE, 15, "#3B4A3F", bold=True, anchor="middle"))
    return svg(W, y + 30, TRUST_TITLE, TRUST_DESC, "".join(parts))


# --- 3. the three tiers -------------------------------------------------------------------------

TIERS_TITLE = "Three tiers, one rule"
TIERS_SUB = ("Split the way Apple Intelligence splits on-device, private cloud and third-party work. The "
             "closer a tier sits to the open internet, the less of you it gets to see.")
TIERS = [
    ("EDGE", "Where you work. It reads context and leaves notes, and runs no jobs.", "#EEF3FA", "#245FA8", [
        [("person", "You", "Work the way you already do. The context comes along with you."),
         ("agents", "Phone, car, voice", "Wherever you work next picks up the same context, so nothing "
                                         "starts from scratch."),
         ("agents", "Claude Code, Codex, Copilot", "Search and read the vault while they work, and leave "
                                                   "notes in their own inbox folder.")],
    ]),
    ("PRIVATE COMPUTE", "Hardware you own: a PC today, a NAS later. Your raw history lives and gets read "
                        "here, and nowhere else.", "#ECF5EF", "#23704F", [
        [("code", "Capture", "Copies every new chat in from the tools you use, with keys and passwords "
                             "scrubbed out."),
         ("local", "Night job and profile", "Learns from your chats while you sleep, and proposes changes "
                                            "for you to approve."),
         "GATEWAY"],
        ["VAULT",
         ("local", "Search", "Finds notes by meaning, with models running on your own GPU."),
         None],
    ]),
    ("CLOUD", "Really useful, so it gets the smallest slice it needs.", "#F7F2E8", "#7A6440", [
        [("storage", "Google Drive", "Keeps the vault in sync between your computers."),
         None,
         ("cloudagents", "ChatGPT and other cloud agents", "Ask through the gateway's one door, and get "
                                                           "answers back, never your raw chats.")],
    ]),
]
GATEWAY = ("code", "Gateway", "The only door out. Your own trusted agents get search and read. Everyone "
                              "else gets one question at a time, ask(question, purpose), and a short "
                              "answer with citations.")
VAULT = ("The vault", "Plain markdown notes: the one store everything reads from.")
TIERS_FOOT = "Raw material stays on hardware you own. Answers are what travel."
TIERS_DESC = ("A diagram of Bonsai's three tiers, stacked. Edge, where you work: your phone, car or voice "
              "assistant, and Claude Code, Codex and Copilot, which search and read the vault and leave "
              "notes in their own inbox. Private compute, hardware you own: capture, the night job and "
              "profile, local search models, the vault itself, and the gateway, the only door out. Cloud: "
              "Google Drive keeps the vault in sync, and cloud agents like ChatGPT ask the gateway one "
              "question at a time and get short answers with citations, never raw chats.")


def tiers() -> str:
    W, M, G, BP = 1000, 32, 18, 18
    colw = (W - 2 * M - 2 * BP - 2 * G) / 3
    col_x = [M + BP + i * (colw + G) for i in range(3)]
    parts = [text(M, 62, TIERS_TITLE, 30, INK, bold=True)]
    sub = wrap(TIERS_SUB, 15.5, W - 2 * M)
    parts.append(lines_at(M, 92, sub, 15.5, 22, SOFT))
    y = 92 + len(sub) * 22 + 14
    leg, lh = legend(M, y + 6, ["local", "code", "agents", "storage"], W - 2 * M)
    parts.append(leg)
    y += lh + 22
    spots = {}
    for b, (name, blurb, tint, accent, rows) in enumerate(TIERS):
        band_top = y
        body = []
        bl = wrap(blurb, 13.5, col_x[2] + colw / 2 - (M + BP + 190) - 24)
        body.append(text(M + BP, y + 30, name, 13, accent, bold=True, spacing="1"))
        body.append(lines_at(M + BP + 190, y + 30, bl, 13.5, 19, SOFT))
        cy = y + 30 + max(1, len(bl)) * 19 + 8
        row_tops = []
        for r, row in enumerate(rows):
            heights = []
            for c, cell in enumerate(row):
                if isinstance(cell, tuple):
                    heights.append(card(col_x[c], cy, colw, *cell)[1])
            rh = max(heights + [104])
            row_tops.append((cy, rh))
            for c, cell in enumerate(row):
                if isinstance(cell, tuple):
                    s, _ = card(col_x[c], cy, colw, *cell, min_h=rh)
                    body.append(s)
                    spots[(b, r, c)] = (col_x[c], cy, rh)
                elif cell == "VAULT":
                    body.append(f'<rect x="{col_x[c]}" y="{cy}" width="{colw:.1f}" height="{rh:.1f}" rx="12" '
                                f'fill="#3B4A3F"/>')
                    body.append(text(col_x[c] + 18, cy + 34, VAULT[0], 18, "#FFFFFF", bold=True))
                    body.append(lines_at(col_x[c] + 18, cy + 60, wrap(VAULT[1], 14, colw - 36), 14, LEAD,
                                         "#E6ECE7"))
                    spots[(b, r, c)] = (col_x[c], cy, rh)
            cy += rh + 18
        if any(cell == "GATEWAY" for row in rows for cell in row):
            gy = row_tops[0][0]
            gh = row_tops[-1][0] + row_tops[-1][1] - gy
            s, _ = card(col_x[2], gy, colw, *GATEWAY, min_h=gh)
            body.append(s)
            spots["gateway"] = (col_x[2], gy, gh)
        band_h = cy - 18 + BP - band_top
        parts.append(f'<rect x="{M}" y="{band_top}" width="{W - 2 * M}" height="{band_h:.1f}" rx="16" '
                     f'fill="{tint}"/>')
        parts.extend(body)
        spots[("band", b)] = (band_top, band_h)
        y = band_top + band_h + 62

    def gap_mid(b):
        top, h = spots[("band", b)]
        return top + h + 31 + 5

    def link(x, y1, y2, label, both=False, left=False, mid=None):
        out = arrow_down(x, y1, y2, "#3B4A3F")
        if both:
            out += (f'<path d="M{x - 5} {y1 + 8} L{x} {y1} L{x + 5} {y1 + 8}" fill="none" stroke="#3B4A3F" '
                    f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>')
        mid = mid if mid is not None else (y1 + y2) / 2 + 5
        if left:
            return out + text(x - 14, mid, label, 13, "#3B4A3F", bold=True, anchor="end")
        return out + text(x + 14, mid, label, 13, "#3B4A3F", bold=True)

    ax, ay, ah = spots[(0, 0, 2)]
    gx, gy, gh = spots["gateway"]
    parts.append(link(ax + colw / 2, ay + ah + 4, gy - 4, "search, read, notes into their inbox", both=True,
                      left=True, mid=gap_mid(0)))
    cx_, cy_, ch_ = spots[(2, 0, 2)]
    parts.append(link(gx + colw / 2, gy + gh + 4, cy_ - 4, "a question in, a short answer out", both=True,
                      left=True, mid=gap_mid(1)))
    vx, vy, vh = spots[(1, 1, 0)]
    dx, dy, dh = spots[(2, 0, 0)]
    parts.append(link(vx + colw / 2, vy + vh + 4, dy - 4, "kept in sync", both=True, mid=gap_mid(1)))
    fy = y - 62 + 46
    parts.append(text(W / 2, fy, TIERS_FOOT, 15, "#3B4A3F", bold=True, anchor="middle"))
    return svg(W, fy + 30, TIERS_TITLE, TIERS_DESC, "".join(parts))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("crew", crew), ("trust", trust), ("tiers", tiers)):
        path = OUT / f"{name}.svg"
        path.write_text(make(), encoding="utf-8")
        print(f"wrote {path.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
