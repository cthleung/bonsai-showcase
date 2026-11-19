"""HTTP to the local llama-server processes, never through a proxy.

urllib picks up HTTP_PROXY and the Windows system proxy by default, which would send chat text
meant for 127.0.0.1 to a proxy. This opener has an empty ProxyHandler, so it always connects
straight to the address given.
"""
from __future__ import annotations

import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def urlopen(req, timeout: float):
    return OPENER.open(req, timeout=timeout)
