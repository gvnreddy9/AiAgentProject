from __future__ import annotations

import datetime
import os

from anyio import Path
from langchain_ollama import ChatOllama
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(Path(__file__).with_name(".env"), override=True)

WEATHER_API_KEY = os.environ["WEATHER_API_KEY"]
WEATHER_BASE_URL = "https://api.weatherapi.com/v1"
TIMEOUT = 15

def _call_weatherapi(endpoint: str, params: dict) -> dict:
    """Shared HTTP helper for WeatherAPI.com endpoints."""
    response = requests.get(
        f"{WEATHER_BASE_URL}/{endpoint}",
        params={"key": WEATHER_API_KEY, **params},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
# The docstring is the tool description the model reads, so we need to be
# clear about when to use the tool and what the arguments mean.


@tool
def get_current_weather(location: str) -> str:
    """Get the CURRENT weather conditions for a location.

    Use this for questions about weather right now. For future days, use
    get_weather_forecast instead.

    Args:
        location: A city name, "city,country", US zip code, UK postcode,
            IATA airport code, or "latitude,longitude". Examples:
            "Seattle", "London,UK", "98101", "47.61,-122.33".
    """
    try:
        data = _call_weatherapi("current.json", {"q": location, "aqi": "no"})
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            return f"WeatherAPI could not find a location matching '{location}'."
        return f"Weather lookup failed: {exc}"
    except requests.RequestException as exc:
        return f"Weather lookup failed: {exc}"

    loc, cur = data["location"], data["current"]
    return (
        f"Current weather for {loc['name']}, {loc['region']}, {loc['country']} "
        f"(local time {loc['localtime']}):\n"
        f"- Condition: {cur['condition']['text']}\n"
        f"- Temperature: {cur['temp_c']}°C / {cur['temp_f']}°F "
        f"(feels like {cur['feelslike_c']}°C / {cur['feelslike_f']}°F)\n"
        f"- Humidity: {cur['humidity']}%\n"
        f"- Wind: {cur['wind_kph']} kph / {cur['wind_mph']} mph {cur['wind_dir']}\n"
        f"- Precipitation: {cur['precip_mm']} mm\n"
        f"- UV index: {cur['uv']}"
    )


@tool
def get_weather_forecast(location: str, days: int = 3) -> str:
    """Get a multi-day weather FORECAST for a location.

    Use this for questions about tomorrow, this weekend, or the next few days.

    Args:
        location: A city name, "city,country", US zip code, UK postcode,
            IATA airport code, or "latitude,longitude".
        days: Number of days to forecast, from 1 to 10. Free WeatherAPI
            accounts are limited to 3 days.
    """
    days = max(1, min(int(days), 10))
    try:
        data = _call_weatherapi(
            "forecast.json",
            {"q": location, "days": days, "aqi": "no", "alerts": "no"},
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            return f"WeatherAPI could not find a location matching '{location}'."
        return f"Forecast lookup failed: {exc}"
    except requests.RequestException as exc:
        return f"Forecast lookup failed: {exc}"

    loc = data["location"]
    lines = [f"{days}-day forecast for {loc['name']}, {loc['country']}:"]
    for entry in data["forecast"]["forecastday"]:
        day = entry["day"]
        lines.append(
            f"- {entry['date']}: {day['condition']['text']}, "
            f"high {day['maxtemp_c']}°C / {day['maxtemp_f']}°F, "
            f"low {day['mintemp_c']}°C / {day['mintemp_f']}°F, "
            f"{day['daily_chance_of_rain']}% chance of rain, "
            f"max wind {day['maxwind_kph']} kph"
        )
    return "\n".join(lines)


web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic",
)

TOOLS = [get_current_weather, get_weather_forecast, web_search]


# --------------------------------------------------------------------------
# Agent + memory
# --------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a helpful research assistant with live access to \
weather data and the web. Today's date is {datetime.date.today():%Y-%m-%d}.

Tool selection rules:
- Weather questions (temperature, rain, forecast, "should I bring a jacket") \
  -> use get_current_weather or get_weather_forecast. Never use web search for \
  weather; the weather tools are more accurate.
- Anything else needing current or factual information -> use tavily_search.
- You may call several tools before answering, and you may call the same tool \
  more than once for different locations.

You remember earlier turns in this conversation, so resolve follow-ups like \
"what about tomorrow?" or "and in Paris?" against what was already discussed.

Answer concisely. Cite source URLs when you used web search. If a tool returns \
an error, tell the user plainly rather than guessing at an answer."""

# InMemorySaver keeps checkpoints in this process's RAM: history survives across
# requests but NOT across restarts, and is not shared between workers. See the
# note at the bottom of this file for durable alternatives.
checkpointer = InMemorySaver()
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

agent = create_agent(
    model=llm,  #"anthropic:claude-sonnet-4-6" or "google_genai:gemini-2.5-flash"
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)



def _config(thread_id: str) -> dict:
    """Build the RunnableConfig that scopes a call to one conversation."""
    return {"configurable": {"thread_id": thread_id}}


def ask(question: str, thread_id: str) -> str:
    """Send one message to the agent and return its reply text.

    Only the new message is passed in — prior turns are restored from the
    checkpointer using thread_id.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=_config(thread_id),
    )
    return result["messages"][-1].text()


def get_history(thread_id: str) -> list[dict]:
    """Return the stored conversation as [{'role': ..., 'content': ...}, ...].

    Tool calls and tool results are filtered out — this is the user-facing
    transcript, meant for rehydrating a UI.
    """
    state = agent.get_state(_config(thread_id))
    if not state.values:
        return []

    transcript = []
    for message in state.values.get("messages", []):
        if message.type == "human":
            transcript.append({"role": "user", "content": message.text()})
        elif message.type == "ai":
            text = message.text()
            if text.strip():  # skip tool-call-only turns, which have no text
                transcript.append({"role": "assistant", "content": text})
    return transcript


def clear_thread(thread_id: str) -> None:
    """Delete all stored checkpoints for one conversation."""
    checkpointer.delete_thread(thread_id)