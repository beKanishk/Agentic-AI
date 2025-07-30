from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from textwrap import dedent
from agno.tools.yfinance import YFinanceTools
from agno.models.google import Gemini

import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

web_search_agent = Agent(
    name="Web search financial agent",
    description = dedent("""Search the web for financial information and provide insights."""),
    model=Groq(id = "llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools(),],
    instructions=dedent("""Your task is to search the web for financial information and provide insights.
    """),
    show_tool_calls=True,
    markdown=True,
)

financial_agent = Agent(
    name="Financial AI agent",
    description=dedent("""Analyze financial data and provide insights."""),
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=dedent("""
        Use tables to display the data.
        """),
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    name="Multi AI agent",
    model = Gemini(api_key=GOOGLE_API_KEY, id="gemini-1.5-flash", search = True),
    team = [web_search_agent, financial_agent],
    instructions = ["1. Use the web search agent to find the latest financial news and updates.",
        "2. Use the financial agent to analyze stock data and provide structured insights.",
        "3. Always combine the results from both agents into a single, cohesive response.",
        "4. Follow these formatting guidelines:",
        "   - Use headings to separate sections (e.g., 'Latest News', 'Stock Analysis').",
        "   - Use tables for numerical data.",
        "   - Include clickable links for all sources.",
        "5. If the user's query is unclear, ask for clarification before proceeding.",
        "6. Always provide a summary of the findings at the end.",],
    show_tool_calls=True,
    markdown=True,
    )

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVIDIA",stream=True)
