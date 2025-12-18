# FILE: integrations/ai_market_engine.py
# Path: integrations/ai_market_engine.py
"""
AI Market Engine integrating NewsData + Perplexity AI.
- Saves reports to ai_reports/ai_report_<ts>.json
- Sends Telegram alerts via telegram_alerts.send_ai_market_alert(report)
- Toggle run-only-when-directional with env AI_RUN_ONLY_DIRECTIONAL=1
"""

from __future__ import annotations
import os
import time
import math
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import pathlib
import requests

from integrations.news_fetcher import NewsFetcher

# defaults
DEFAULT_PERPLEXITY_MODEL = os.environ.get("PERPLEXITY_MODEL", "sonar")
DEFAULT_PERPLEXITY_SEARCH_DEPTH = os.environ.get("PERPLEXITY_SEARCH_DEPTH", "medium")
DEFAULT_WEIGHTS = {
    "htf_sr": 0.25,
    "vob": 0.20,
    "overall_sentiment": 0.20,
    "option_chain": 0.15,
    "proximity_alerts": 0.10,
    "other": 0.10,
}
DEFAULT_MAX_HEADLINES = 10
DEFAULT_NEWS_TTL = 120
AI_REPORT_DIR = os.environ.get("AI_REPORT_DIR", "ai_reports")
AI_RUN_ONLY_DIRECTIONAL = os.environ.get("AI_RUN_ONLY_DIRECTIONAL", "") == "1"  # if set -> run only on BULL/BEAR
TELEGRAM_CONFIDENCE_THRESHOLD = float(os.environ.get("AI_TELEGRAM_CONFIDENCE", "0.60"))

def _clamp(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))

def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {}
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s == 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: (float(v) / s if v > 0 else 0.0) for k, v in weights.items()}

def _sigmoid_confidence(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-6 * (abs(x) - 0.5)))

class WeightedBiasEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = _normalize_weights(weights or DEFAULT_WEIGHTS)

    def compute(self, module_biases: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        if not module_biases:
            return 0.0, {}
        contributions = {}
        w_other = self.weights.get("other", 0.0)
        for k, v in module_biases.items():
            w = self.weights.get(k, w_other)
            contributions[k] = float(v) * w
        aggregated = sum(contributions.values())
        return _clamp(aggregated), contributions

class PerplexityClient:
    def __init__(self, api_key: Optional[str], model: str = DEFAULT_PERPLEXITY_MODEL, search_depth: str = DEFAULT_PERPLEXITY_SEARCH_DEPTH):
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        self.model = model
        self.search_depth = search_depth
        self.base_url = "https://api.perplexity.ai/openai/chat/completions"

    async def ready(self) -> bool:
        return self.api_key is not None

    def _build_prompt(self, texts: List[str], hint: Optional[str] = None) -> str:
        header = "You are an expert market analyst. Analyze the following market data and return JSON: {summary: str, score: float (-1..1), reasons: [str,...]}.\n"
        if hint:
            header += f"Context: {hint}\n"
        body = "\n".join(f"- {t}" for t in texts[:12])
        return header + "\nItems:\n" + body + "\n\nRespond with valid JSON."

    def _fallback(self, texts: List[str]) -> Dict[str, Any]:
        positive = {"gain", "rise", "up", "bull", "positive", "surge", "beat", "profit"}
        negative = {"fall", "down", "drop", "bear", "negative", "crash", "decline", "loss"}
        score = 0.0
        n = 0
        reasons = []
        for t in texts:
            if not t:
                continue
            n += 1
            low = t.lower()
            p = sum(1 for w in positive if w in low)
            q = sum(1 for w in negative if w in low)
            score += (p - q)
            if p >= 1:
                reasons.append("positive headline: " + (t[:120]))
            if q >= 1:
                reasons.append("negative headline: " + (t[:120]))
        if n == 0:
            return {"summary": "", "score": 0.0, "reasons": []}
        raw = score / (n * 3.0)
        return {"summary": " | ".join([t for t in texts[:3] if t])[:800], "score": _clamp(raw), "reasons": reasons[:5]}

    async def summarize_and_score(self, texts: List[str], hint: Optional[str] = None) -> Dict[str, Any]:
        if not await self.ready():
            return self._fallback(texts)

        prompt = self._build_prompt(texts, hint)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                print(f"⚠️ Perplexity API error: {response.status_code} - {response.text}")
                return self._fallback(texts)

            resp_json = response.json()
            content = None

            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                choice = resp_json["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content")

            if not content:
                return self._fallback(texts)

            try:
                j = json.loads(content)
                if "score" in j:
                    j["score"] = float(j["score"])
                    j["score"] = _clamp(j["score"])
                return j
            except Exception:
                import re
                m = re.search(r"([+-]?\d\.\d+)", content)
                sc = float(m.group(1)) if m else 0.0
                return {"summary": content.strip()[:800], "score": _clamp(sc), "reasons": content.strip().splitlines()[:3]}

        except Exception as e:
            print(f"⚠️ Perplexity API exception: {str(e)}")
            return self._fallback(texts)

class AIMarketEngine:
    def __init__(self, news_api_key: Optional[str] = None, perplexity_api_key: Optional[str] = None, weights: Optional[Dict[str, float]] = None):
        self.news = NewsFetcher(api_key=news_api_key)
        self.perplexity = PerplexityClient(api_key=perplexity_api_key)
        self.bias_engine = WeightedBiasEngine(weights=weights or DEFAULT_WEIGHTS)
        self._news_ttl = DEFAULT_NEWS_TTL
        pathlib.Path(AI_REPORT_DIR).mkdir(parents=True, exist_ok=True)

    async def close(self):
        await self.news.close()

    async def analyze(self, overall_market: str, module_biases: Dict[str, float], market_meta: Optional[Dict[str, Any]] = None, save_report: bool = True, telegram_send: bool = True) -> Dict[str, Any]:
        overall = (overall_market or "").strip().upper()
        if AI_RUN_ONLY_DIRECTIONAL and overall not in {"BULL", "BEAR"}:
            return {"market": overall_market, "triggered": False, "reason": "AI only runs for BULL/BEAR (AI_RUN_ONLY_DIRECTIONAL=1)"}

        technical_score, contributions = self.bias_engine.compute(module_biases or {})
        query = (market_meta or {}).get("query") or "stock market"
        raw_news = await self.news.fetch_headlines(q=query, max_items=DEFAULT_MAX_HEADLINES, ttl=self._news_ttl)
        texts = [ (a.get("title") or "") + " — " + (a.get("description") or "") for a in raw_news ]

        news_result = await self.perplexity.summarize_and_score(texts, hint=f"Market hint: {overall}. Technical score: {technical_score:.3f}")
        news_score = float(news_result.get("score", 0.0))
        news_summary = news_result.get("summary", "")[:1200]
        news_reasons = news_result.get("reasons", []) or []

        meta_score = 0.0
        if market_meta:
            vol = float(market_meta.get("volatility", 0.0) or 0.0)
            vol_adj = max(-0.15, min(0.15, (vol - 0.1)))
            vol_dir = float(market_meta.get("volatility_dir", 0.0) or 0.0)
            vol_dir_adj = max(-0.1, min(0.1, 0.1 * vol_dir))
            meta_score = _clamp(vol_adj + vol_dir_adj + float(market_meta.get("volume_change", 0.0) or 0.0))

        w_tech, w_news, w_meta = 0.6, 0.3, 0.1
        combined = _clamp((technical_score * w_tech) + (news_score * w_news) + (meta_score * w_meta))

        ai_summary = ""
        ai_reasons = []
        ai_score = combined
        if await self.perplexity.ready():
            justification_text = (
                f"Overall market: {overall}\n"
                f"Technical score: {technical_score:.4f}\n"
                f"Technical contributions: {json.dumps(contributions)}\n"
                f"News score: {news_score:.4f}\n"
                f"News summary: {news_summary}\n"
                f"Market meta: {json.dumps(market_meta or {})}\n"
                "Return JSON with keys: ai_score (float -1..1), label, confidence (0..1), reasons (list).\n"
            )
            resp = await self.perplexity.summarize_and_score([justification_text], hint="Finalize market decision")
            ai_score = float(resp.get("score", combined))
            ai_summary = resp.get("summary", "")[:1500]
            if isinstance(resp.get("reasons", None), list):
                ai_reasons = resp.get("reasons")[:3]
            else:
                ai_reasons = (ai_summary or "").splitlines()[:3]
        else:
            ai_summary = news_summary
            ai_reasons = news_reasons

        if ai_score >= 0.25:
            label = "BULLISH"
        elif ai_score <= -0.25:
            label = "BEARISH"
        else:
            label = "HOLD"

        confidence = float(_sigmoid_confidence(ai_score))

        report = {
            "market": overall,
            "triggered": True,
            "technical_score": technical_score,
            "technical_contributions": contributions,
            "news_score": news_score,
            "news_summary": news_summary,
            "raw_news": raw_news,
            "meta_score": meta_score,
            "ai_score": ai_score,
            "label": label,
            "confidence": confidence,
            "ai_summary": ai_summary,
            "ai_reasons": ai_reasons,
            "recommendation": ("BUY" if label == "BULLISH" else "SELL" if label == "BEARISH" else "HOLD"),
            "timestamp": int(time.time())
        }

        # save report to file
        if save_report:
            ts = int(time.time())
            fn = pathlib.Path(AI_REPORT_DIR) / f"ai_report_{ts}.json"
            try:
                fn.write_text(json.dumps(report, indent=2))
            except Exception:
                pass

        # attempt to send to telegram if requested and confidence >= threshold
        if telegram_send and confidence >= TELEGRAM_CONFIDENCE_THRESHOLD:
            try:
                # import locally to avoid circular deps at module import time
                from telegram_alerts import send_ai_market_alert
                # send asynchronously
                await send_ai_market_alert(report, confidence_thresh=TELEGRAM_CONFIDENCE_THRESHOLD)
            except Exception:
                # don't break main flow on send failure
                pass

        return report
