"""Run the MCP server over stdio (see gateway/server.py). Two ways:

    python -m gateway [--env-file PATH]                          from the repo folder
    python -I /path/to/bonsai/gateway/__main__.py [--env-file PATH]  from anywhere (the registered form)

The second runs isolated (`-I`: no PYTHON* environment variables, no current folder or user
site-packages on the import path), so nothing in the calling project's environment can change
what runs. This file then puts the repo folder on the import path itself.
"""
import sys
from pathlib import Path

if not __package__:                       # run as a script: find the repo from this file
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
