# Working on this repo

For every agent and every session working here. This repo is public: a scrubbed copy of a private repo, and most changes are ported from there. Several sessions on more than one computer can be working at once, and they only see each other through GitHub: issues, pull requests, pushed branches and comments.

## Before you start

1. **Look first.** `git fetch`, then check the open issues and PRs and search for your topic (`gh issue list --state all --search "<topic>"`, and the same for `gh pr list`). Two sessions fixing the same thing in parallel is the collision to avoid.
2. **One issue per piece of work.** Open one if there's none. If someone has claimed it, leave it alone.
3. **Claim it** with a comment: `Taking this: <computer>, <tool>, branch <branch>, <YYYY-MM-DD>`. Every session uses the same account, so the comment says who's on it.
4. **Push early.** Open a draft PR (`Part of #N`) once there's a first commit, so the other sessions can see the work.
5. **Drop it cleanly.** If you stop, comment saying what's on the branch.

## Branches

- **One task, one branch,** off a fresh `origin/main`: `fix/`, `feat/`, `docs/` or `port/` plus a short topic. Main only changes through PRs.
- **Parallel agents** each get their own worktree and branch, own named files or functions, and write their tests in new test files. One lead merges their branches and runs the full suite.
- **Commit and push before you switch computers.**

## Ports from the private repo

- **Apply the changes hunk by hunk** onto this repo's own versions. Never copy a file over: the shared files here were scrubbed and sometimes trimmed.
- **Keep this repo's names:** "the user", `whose: user`, the `creative` visibility, made-up fixtures.
- **Nothing private goes in:** no names, machine names or home paths, and no issue, PR or review numbers from the private repo, in code, comments, test names, commit messages or PR bodies. Test and file names describe the behavior.
- **Before every push,** run the private-terms check that the private repo keeps, on the added lines and the PR body. The list lives there on purpose.

## Pull requests

- **The body says** what changed and why, the test counts before and after, and what wasn't verified.
- **Closing keywords take one issue each:** `Closes #3. Closes #5.`
- **Commit messages** are a plain sentence about what changed, with no Co-Authored-By or Generated-with lines.
- **Tests:**
  - Run `python -m pytest` before you push. CI runs Windows and macOS; the macOS job uses uv's Python, since setup-python's builds can't load SQLite extensions.
  - Never tie a test to today's date. Use a date relative to now, or pass `now` in.
- **Merging:** a port or fix PR can be merged once CI is green and the private-terms check is clean. Anything in the owner's voice (the README, the diagrams in `docs/img/`, other docs) waits for the owner.
- **Diagrams** come from `docs/diagrams/build.py`. Edit the words there and run it again; never edit the SVGs by hand.
