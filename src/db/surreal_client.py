"""Minimal async wrapper around SurrealDB HTTP/WebSocket API.

Usage:
    from db.surreal_client import SurrealClient, get_surreal_client
    client = get_surreal_client()
    await client.query("SELECT * FROM code_chunk LIMIT 1;")
"""

from __future__ import annotations

import os
import aiohttp
import asyncio
from typing import Any, List

_SURREAL_CLIENT: "SurrealClient | None" = None


class SurrealClient:
    def __init__(self) -> None:
        url = os.getenv("SURREAL_URL", "http://surrealdb:8000")
        user = os.getenv("SURREAL_USER", "root")
        password = os.getenv("SURREAL_PASS", "root")
        self._auth = aiohttp.BasicAuth(user, password)
        self._url = f"{url}/sql"
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(auth=self._auth)
        return self._session

    async def query(self, sql: str, *, timeout: float = 30.0) -> List[Any]:
        """Run a SurrealQL query and return JSON list result."""
        session = await self._ensure_session()
        async with session.post(self._url, data=sql.encode(), timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            # Surreal returns a list of result objects per statement
            return data

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# ----------------- helper -----------------


def get_surreal_client() -> "SurrealClient":
    global _SURREAL_CLIENT
    if _SURREAL_CLIENT is None:
        _SURREAL_CLIENT = SurrealClient()
    return _SURREAL_CLIENT


# Simple sync helper for scripts / debug
def query_sync(sql: str) -> List[Any]:
    loop = asyncio.get_event_loop()
    client = get_surreal_client()
    return loop.run_until_complete(client.query(sql))

