# FILE: integrations/news_fetcher.py
# Path: integrations/news_fetcher.py
"""
Lightweight NewsData.io fetcher with TTL cache and basic throttling.
"""

from __future__ import annotations
import os
import time
import asyncio
from typing import List, Dict, Optional, Any
import aiohttp

DEFAULT_NEWSDATA_URL = os.environ.get("NEWSDATA_BASE_URL", "https://newsdata.io/api/1/news")
DEFAULT_MAX_HEADLINES = 10
DEFAULT_TTL = 120  # seconds

class _TTLCache:
    def __init__(self):
        self._store = {}

    def get(self, k: str):
        v = self._store.get(k)
        if not v:
            return None
        exp, val = v
        if time.time() > exp:
            del self._store[k]
            return None
        return val

    def set(self, k: str, v: Any, ttl: int = DEFAULT_TTL):
        self._store[k] = (time.time() + ttl, v)

class NewsFetcher:
    def __init__(self, api_key: Optional[str] = None, base_url: str = DEFAULT_NEWSDATA_URL, session: Optional[aiohttp.ClientSession] = None):
        self.api_key = api_key or os.environ.get("NEWSDATA_API_KEY")
        self.base_url = base_url
        self._session = session
        self._cache = _TTLCache()
        self._lock = asyncio.Lock()
        self._min_interval = 0.25
        self._last_request = 0.0

    async def _session(self) -> aiohttp.ClientSession:
        if self._session:
            return self._session
        self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def fetch_headlines(self, q: Optional[str] = None, language: str = "en", max_items: int = DEFAULT_MAX_HEADLINES, ttl: int = DEFAULT_TTL) -> List[Dict]:
        """
        Returns list of articles: {title, description, link, pubDate, source_id}
        """
        if not self.api_key:
            return []
        params = {"apikey": self.api_key, "language": language, "page": 0}
        if q:
            params["q"] = q
        cache_key = f"news:{q}:{language}:{max_items}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        async with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request = time.time()
            session = await self._session()
            try:
                async with session.get(self.base_url, params=params, timeout=10) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    items = []
                    for r in data.get("results", [])[:max_items]:
                        items.append({
                            "title": r.get("title"),
                            "description": r.get("description"),
                            "link": r.get("link"),
                            "pubDate": r.get("pubDate"),
                            "source_id": r.get("source_id"),
                        })
                    self._cache.set(cache_key, items, ttl=ttl)
                    return items
            except Exception:
                return []
