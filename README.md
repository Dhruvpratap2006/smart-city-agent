# 🏙️ Smart City Intelligence Agent

An AI agent built with LangChain that answers questions about any city by combining live data from multiple APIs — weather, air quality, news, nearby amenities, and a custom-built livability score.

## Features

- 🌤️ **Current Weather** — real-time temperature and conditions (OpenWeather)
- 🌫️ **Air Quality Index** — AQI level, PM2.5, and health advice (OpenWeather)
- 📰 **Latest News** — recent news for any city (Tavily)
- 🏥 **Nearby Amenities** — hospitals, pharmacies, ATMs, fuel stations, schools (Geoapify)
- 📅 **5-Day Weather Trend** — forecast analysis with the best day for outdoor activities
- 📊 **Livability Score** — a composite 0–100 score combining air quality, weather comfort, and healthcare access, using a custom weighted-scoring algorithm

## Tech Stack

- **LLM:** Mistral (`open-mistral-nemo`) via LangChain
- **UI:** Streamlit
- **APIs:** OpenWeather, Geoapify Places, Tavily Search

## How it works

The agent uses tool-calling: the LLM looks at your question and decides which tool(s) to call, runs them, and combines the results into a natural-language answer. Tool calls run automatically by default (all tools are read-only), with an optional toggle for manual approval before each call.

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirments.txt
   ```

2. Create a `.env` file in the project root with your API keys:
   ```
   MISTRAL_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   OPEN_WEATHER_API_KEY=your_key_here
   GEOAPIFY_API_KEY=your_key_here
   ```

3. Run the web UI:
   ```bash
   streamlit run smart_city_ui.py
   ```

## Example queries

- "What's the weather and air quality in Agra?"
- "Find hospitals near Delhi"
- "What's the livability score for Noida?"
- "Give me a 5-day forecast for Agra and tell me the best day to go outside"

---

Built as a learning project to explore LangChain agents, tool-calling, and combining multiple APIs into a single reasoning system.
