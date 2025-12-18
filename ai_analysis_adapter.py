# FILE: ai_analysis_adapter.py
"""
Adapter to run AIMarketEngine from your app.
Handles AI market analysis integration.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, Any, Optional
import asyncio

# Add the parent directory to the path to find the integrations module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the AI engine
try:
    from integrations.ai_market_engine import AIMarketEngine
    AI_ENGINE_AVAILABLE = True
    print("✅ AI Market Engine imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import AI Market Engine: {e}")
    AI_ENGINE_AVAILABLE = False
    AIMarketEngine = None
    # Try alternative import path
    try:
        # For Streamlit Cloud or different structure
        from .integrations.ai_market_engine import AIMarketEngine
        AI_ENGINE_AVAILABLE = True
        print("✅ AI Market Engine imported from relative path")
    except ImportError:
        print("❌ AI Market Engine not available")
        AI_ENGINE_AVAILABLE = False

_ai_engine: Optional[AIMarketEngine] = None

def get_ai_engine(news_api_key: Optional[str] = None, perplexity_api_key: Optional[str] = None, weights: Optional[Dict[str, float]] = None) -> Optional[AIMarketEngine]:
    """
    Get or create the AI engine instance.
    """
    global _ai_engine

    if not AI_ENGINE_AVAILABLE:
        print("❌ AI Engine not available - feature disabled")
        return None

    if _ai_engine is None:
        try:
            print("🤖 Initializing AI Market Engine with Perplexity AI...")
            _ai_engine = AIMarketEngine(
                news_api_key=news_api_key,
                perplexity_api_key=perplexity_api_key,
                weights=weights
            )
            print("✅ AI Market Engine initialized successfully with Perplexity AI")
        except Exception as e:
            print(f"❌ Error initializing AI Engine: {e}")
            return None
    return _ai_engine

async def run_ai_analysis(
    overall_market: str,
    module_biases: Dict[str, float],
    market_meta: Optional[Dict[str, Any]] = None,
    news_api_key: Optional[str] = None,
    perplexity_api_key: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    save_report: bool = True,
    telegram_send: bool = True
) -> Dict[str, Any]:
    """
    Run AI market analysis.

    Args:
        overall_market: Market description (e.g., "Indian Stock Market")
        module_biases: Dictionary of module biases/scores
        market_meta: Additional market metadata
        news_api_key: News API key
        perplexity_api_key: Perplexity API key
        weights: Module weights for analysis
        save_report: Whether to save the report
        telegram_send: Whether to send to Telegram

    Returns:
        Dictionary with analysis results
    """
    if not AI_ENGINE_AVAILABLE:
        return {
            'success': False,
            'error': 'AI Engine not available. Please check if integrations.ai_market_engine is installed.',
            'sentiment': 'NEUTRAL',
            'score': 0,
            'confidence': 0,
            'reasoning': ['AI analysis feature is disabled'],
            'final_verdict': 'AI analysis unavailable. Using traditional analysis only.'
        }

    engine = get_ai_engine(
        news_api_key=news_api_key,
        perplexity_api_key=perplexity_api_key,
        weights=weights
    )
    
    if not engine:
        return {
            'success': False,
            'error': 'Failed to initialize AI Engine',
            'sentiment': 'NEUTRAL',
            'score': 0,
            'confidence': 0,
            'reasoning': ['AI engine initialization failed'],
            'final_verdict': 'AI analysis unavailable. Using traditional analysis only.'
        }
    
    try:
        print("🤖 Starting AI market analysis...")
        report = await engine.analyze(
            overall_market, 
            module_biases, 
            market_meta, 
            save_report=save_report, 
            telegram_send=telegram_send
        )
        print("✅ AI analysis completed successfully")
        return report
    except Exception as e:
        print(f"❌ AI analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'sentiment': 'NEUTRAL',
            'score': 0,
            'confidence': 0,
            'reasoning': [f'AI analysis error: {str(e)}'],
            'final_verdict': 'AI analysis failed. Please check API keys and connectivity.'
        }

async def shutdown_ai_engine():
    """
    Cleanly shutdown the AI engine.
    """
    global _ai_engine
    if _ai_engine:
        try:
            await _ai_engine.close()
            print("✅ AI Engine shutdown successfully")
        except Exception as e:
            print(f"⚠️ Error shutting down AI Engine: {e}")
        finally:
            _ai_engine = None
