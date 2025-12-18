"""
Perplexity AI Market Insights Module

This module provides real-time market insights using Perplexity AI's web research capabilities.
Users can ask questions about market conditions and get AI-powered analysis with citations.
"""

import streamlit as st
import requests
from typing import Optional, Dict, Any
from datetime import datetime
import config


class PerplexityMarketInsights:
    """
    Market Insights powered by Perplexity AI with web research capabilities
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Perplexity Market Insights

        Args:
            api_key: Perplexity API key (optional, will try to load from config)
        """
        if api_key:
            self.api_key = api_key
        else:
            # Try to load from config
            creds = config.get_perplexity_credentials()
            self.api_key = creds.get('api_key') if creds.get('enabled') else None
            self.model = creds.get('model', 'sonar')
            self.search_depth = creds.get('search_depth', 'medium')

        self.base_url = "https://api.perplexity.ai/openai/chat/completions"

    def get_market_insight(
        self,
        query: str,
        bias_data: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get market insight from Perplexity AI

        Args:
            query: User's market-related question
            bias_data: Optional bias analysis results to provide context
            current_price: Optional current market price

        Returns:
            Dictionary with insight results including answer and citations
        """
        if not self.api_key:
            return {
                'success': False,
                'error': 'Perplexity API key not configured',
                'answer': 'Please configure your Perplexity API key in settings.'
            }

        # Build context-aware query
        full_query = self._build_context_query(query, bias_data, current_price)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in Indian stock markets. Provide detailed, data-driven insights with specific market data and news."
                    },
                    {
                        "role": "user",
                        "content": full_query
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.2,
                "top_p": 0.9
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'API Error: {response.status_code}',
                    'answer': f'Perplexity API returned error: {response.text}'
                }

            resp_json = response.json()

            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                content = resp_json["choices"][0]["message"]["content"]

                return {
                    'success': True,
                    'answer': content,
                    'citations': self._extract_citations(resp_json),
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'model': self.model
                }
            else:
                return {
                    'success': False,
                    'error': 'No response from Perplexity',
                    'answer': 'Unable to get response from Perplexity AI'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': f'Error getting market insight: {str(e)}'
            }

    def _build_context_query(
        self,
        query: str,
        bias_data: Optional[Dict[str, Any]],
        current_price: Optional[float]
    ) -> str:
        """Build a context-aware query with bias data"""
        context_parts = []

        if current_price:
            context_parts.append(f"Current NIFTY 50 price: {current_price:.2f}")

        if bias_data:
            context_parts.append(f"Overall Bias: {bias_data.get('overall_bias', 'NEUTRAL')}")
            context_parts.append(f"Confidence: {bias_data.get('overall_confidence', 0)}%")
            context_parts.append(
                f"Technical Score: {bias_data.get('bullish_count', 0)} Bullish, "
                f"{bias_data.get('bearish_count', 0)} Bearish"
            )

        if context_parts:
            context = "\n".join(context_parts)
            return f"{context}\n\nQuestion: {query}"
        else:
            return query

    def _extract_citations(self, response_json: Dict[str, Any]) -> list:
        """Extract citations from Perplexity response if available"""
        # Perplexity includes citations in the response
        # This is a placeholder for citation extraction logic
        citations = []
        if "citations" in response_json:
            citations = response_json["citations"]
        return citations

    def get_quick_market_summary(self, index: str = "NIFTY 50") -> Dict[str, Any]:
        """
        Get a quick market summary for the specified index

        Args:
            index: Market index name (default: NIFTY 50)

        Returns:
            Market summary with key insights
        """
        query = f"What is the current market sentiment for {index}? Include key drivers, major news, and technical outlook for today."
        return self.get_market_insight(query)

    def analyze_bias_with_ai(
        self,
        bias_data: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Get AI analysis of the current bias setup

        Args:
            bias_data: Bias analysis results
            current_price: Current market price

        Returns:
            AI-powered bias analysis
        """
        query = (
            f"Based on the technical setup, what is the probability of a breakout? "
            f"Are there any upcoming catalysts (earnings, economic data, geopolitical events) "
            f"that could impact the market today?"
        )

        return self.get_market_insight(query, bias_data, current_price)


def render_market_insights_widget(bias_data: Optional[Dict[str, Any]] = None):
    """
    Render the market insights widget in Streamlit

    Args:
        bias_data: Optional bias analysis data to provide context
    """
    st.markdown("### 🤖 AI Market Insights (Powered by Perplexity AI)")
    st.caption("Ask questions about market conditions and get real-time AI-powered insights")

    # Initialize insights engine
    insights = PerplexityMarketInsights()

    # Check if API key is configured
    if not insights.api_key:
        st.warning("⚠️ Perplexity API key not configured")
        st.info("Add your Perplexity API key to `.streamlit/secrets.toml` to enable this feature")

        with st.expander("🔧 How to get a Perplexity API key"):
            st.markdown("""
            1. Go to [Perplexity Settings](https://www.perplexity.ai/settings/api)
            2. Generate a new API key
            3. Add it to your `.streamlit/secrets.toml`:
            ```toml
            [PERPLEXITY]
            API_KEY = "pplx-your-api-key-here"
            ```
            4. As a Pro subscriber, you get $5 monthly credits automatically!
            """)
        return

    st.success("✅ Perplexity AI Ready")

    # Quick actions
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Market Summary", use_container_width=True):
            with st.spinner("🤖 Fetching market summary..."):
                result = insights.get_quick_market_summary()
                if result['success']:
                    st.markdown("#### Market Summary")
                    st.write(result['answer'])
                else:
                    st.error(f"❌ {result['error']}")

    with col2:
        if st.button("🔍 Analyze Bias", use_container_width=True, disabled=not bias_data):
            if bias_data:
                with st.spinner("🤖 Analyzing current bias setup..."):
                    result = insights.analyze_bias_with_ai(
                        bias_data,
                        bias_data.get('current_price', 0)
                    )
                    if result['success']:
                        st.markdown("#### Bias Analysis")
                        st.write(result['answer'])
                    else:
                        st.error(f"❌ {result['error']}")

    with col3:
        if st.button("📰 Breaking News", use_container_width=True):
            with st.spinner("🤖 Fetching latest market news..."):
                result = insights.get_market_insight(
                    "What are the top 3 breaking news stories affecting Indian stock markets today?"
                )
                if result['success']:
                    st.markdown("#### Breaking News")
                    st.write(result['answer'])
                else:
                    st.error(f"❌ {result['error']}")

    st.markdown("---")

    # Custom query section
    st.markdown("#### Ask a Custom Question")

    query = st.text_area(
        "Enter your market-related question:",
        placeholder="e.g., What is the outlook for NIFTY this week? Should I be bullish or bearish?",
        help="Ask anything about market trends, technical analysis, news, or specific stocks"
    )

    if st.button("🚀 Get AI Insight", type="primary", disabled=not query):
        if query:
            with st.spinner("🤖 Researching your question..."):
                result = insights.get_market_insight(query, bias_data)

                if result['success']:
                    st.markdown("#### 🤖 AI Response")
                    st.info(result['answer'])

                    # Display metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"🕒 {result['timestamp']}")
                    with col2:
                        st.caption(f"🔧 Model: {result['model']}")

                    # Display citations if available
                    if result.get('citations'):
                        with st.expander("📚 Sources & Citations"):
                            for i, citation in enumerate(result['citations'], 1):
                                st.write(f"{i}. {citation}")
                else:
                    st.error(f"❌ {result['error']}")

    # Usage info
    with st.expander("💡 Tips for Better Insights"):
        st.markdown("""
        **Effective Questions:**
        - "What is driving the market today?"
        - "Should I be bullish or bearish on NIFTY this week?"
        - "What are the key levels to watch for NIFTY?"
        - "What earnings reports are coming up this week?"
        - "How are global markets affecting Indian stocks?"

        **Cost Optimization:**
        - You get $5 monthly credits as a Pro subscriber
        - Each query uses ~1-3 credits depending on complexity
        - Use targeted questions for better results
        - Pro tip: Combine related questions into one query
        """)


if __name__ == "__main__":
    # Test the insights engine
    insights = PerplexityMarketInsights()
    result = insights.get_quick_market_summary()
    print(result)
