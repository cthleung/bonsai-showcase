# Bonsai

A personal context store: one plain markdown vault of notes, memories and captured AI chats that you and every agent you use can read and write, on every device, plus the nightly jobs that keep it tidy and slowly learn who you are.

This is a showcase copy of the code I run every day. The parts that only make sense for my own setup (and a couple of capture scripts that poke at other companies' private endpoints) stay in my private repo, so think of this as the core of the thing, not a product you can install and forget about. Not yet, anyway.

## Why I built it

I've been writing a novel-length project with Claude Code for a long time now, and somewhere along the way the question stopped being "can a model write" and turned into "how the fuck do i get a model trained on averages to write like me/do things my way?" The answer, as it turns out, is mostly context - years of decisions, preferences, rejected drafts and thoughts I never finished, which a model never sees unless you hand them over, every session, by hand.

And I have that context, scattered across ChatGPT, Claude Code, Codex, Copilot, Google's AI Mode and a pile of markdown notes, on two computers. Every tool remembers a little bit of me in its own silo, none of them talk to each other, and I do not like fragmented experiences that require multiple frontends. So the first goal was pretty boring and practical - one store for the important stuff (the long goals all my agents need to work towards), filled automatically, because I'm just never going to export my chats by hand.

But the bigger goal is a personal intelligence mesh for every person, and a vision of computing beyond a fixed screen. I want to be able to work by voice on the drive home, pick it up at my desk, then on the laptop later, and have every agent along the way already know what I'm doing and why. This repo is the first step toward that. It's very much a proof of concept, but it's one that I also need to use immediately.

## The idea: context follows you

Compute and agent surfaces come and go. Today it's Claude Code on a PC, tomorrow it's ChatGPT on a phone, a car's voice assistant, or a model that doesn't exist yet. Working context and personal understanding (what you're working on, how you like things done, what you've already decided) should follow the user everywhere, instead of being rebuilt from scratch in every new app.

So it's got to live somewhere you own, in a format any agent can read, and with rules about who gets to see what. Bonsai splits that into three tiers, borrowing the way Apple Intelligence splits on-device, private cloud and third-party work:

| Tier | What's there | What it does |
|---|---|---|
| **Edge** | My laptop, and the agents running on my PC (Claude Code, Codex, Copilot) | Reads context, writes notes to its own inbox folder. Runs no jobs. |
| **Private compute** | My PC today, a NAS later | Capture, the search index, the nightly dream job, reflection, and later the gateway. Raw material only ever gets processed here. |
| **Cloud** | Google Drive (where the vault syncs), cloud agents like ChatGPT | Drive stores the vault. Cloud agents get answers, never raw material. |

The edge is where you work, so it's kept light. Private compute is where your raw history lives and gets read, because years of personal chats shouldn't be shipped off to whoever asks. And the cloud is really useful, but it only gets the smallest slice it needs. The plan is a gateway on the private compute tier with one door, `ask(question, purpose)`, that searches locally and hands back a short answer with citations (like Perplexity, but for personal indexing instead of the web). Your own trusted agents get raw search and read; everyone else gets answers.

## What's in here

- **Capture** (`capture/`) turns local Claude Code, Copilot and Codex session transcripts into chat notes, plus an importer for the official ChatGPT data export. Keys and tokens are scrubbed before anything is written, and a session that's already captured is only rewritten when it actually changed.
- **The dream job** (`dream/night.py`) runs every night. It reads what's new in the inboxes and chats, asks a model (`claude -p`, tools off, a JSON schema, a hard dollar cap) for proposed memory changes, and writes a morning digest of checkboxes. **Nothing in the vault changes until you tick a box.** Applying takes a snapshot first, so you can undo any change later.
- **Reflection** (`dream/filter.py`, `dream/reflect.py`) replays your chats month by month to build a profile of you. This is probably the part I care about most, so it has rules:
  - Only your own words count as evidence, never the model's answers.
  - Every claim carries a verbatim quote, and **a script** checks the quote really is in your turn of that chat. Models invent a lot of claims about people and their self-checks don't catch it, so the checking is done by code, by a separate cold fact-checker, and by you.
  - Interpretations need two or more separate chats, can't name a motive or a feeling, and never promote themselves to facts. They come back to you as questions instead.
- **Hybrid search** (`index/`) uses SQLite FTS5 (plus a trigram table for Chinese, Japanese and Korean) and sqlite-vec vectors, with a local embedding model and reranker served by llama.cpp. The index lives outside the vault, so it's fine to delete it and rebuild it whenever.
- **Vault helpers** (`vault/`) make the PARA folders and templates, and rebuilds a one-line-per-note `Index.md` that agents can load at the start of a session.

The vault itself is just an Obsidian folder of markdown, and grep is always enough to find things. The note format is in [docs/note-format.md](docs/note-format.md).

On that note, a few rules hold everywhere. The script sets each note's `visibility`, never the model; one writer per inbox folder, so two computers never fight over a file; and the night job proposes while you approve.

## Try it

```bash
pip install -r requirements.txt
python demo/run_demo.py      # a night run on a made-up vault, with a fake model: no API key, no cost
python -m pytest
```

So the demo just copies a small sample vault to a temp folder, runs the night job against it with a canned model, and prints the morning digest it wrote.

To point it at a real vault, copy `.env.example` to `.env` and set `BONSAI_VAULT` and `BONSAI_STATE`. The night job and reflection need the `claude` CLI; search needs `llama-server` and a GPU with room for the embedding model.

## Where it's going

- The `ask` gateway on the PC, then as a connector for ChatGPT and other cloud agents.
- Moving private compute onto an always-on NAS.
- Voice: answering the week's reflection questions on the drive home.
- A personality for the thing that talks to you. Interacting with a stiff corporate machine is exhausting, and I think the world needs more kindness and care, and the first place to start in an AI world are the AI systems. It can be soulful without anyone treating it as sentient (and it should feel fun and a little silly, like how Windows XP felt back in the day).

The name comes from the bonsai philosophy I try to live by: with care and attention, even ordinary things can become extraordinary, and wonderful things are the product of many small, intentional decisions over time. Which, now that I think about it, is actually a pretty good description of a memory too.
