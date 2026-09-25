# here we are going to learn about agents 
# agents are nothing but they are 
# combinations of (LLM + Tools + some logic)


# First step => loading all the libraries
from dotenv import load_dotenv
import os
import requests # we import req when we have to send the req or want to send the req to am some url

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from rich import print

# weather tool
@tool
def get_weather_tool(city : str) -> str:
    """Get current weather for a given city name."""
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q" : city,
        "appid": api_key,
        "units": "metric" 
    }

    # send the req
    res = requests.get(url, params=params)

    if (res.status_code != 200) :
        return f"Error: could not fetch weather for {city}"

    data = res.json()

    temp = data["main"]["temp"] # taking out the data from res as openweather api send the 
    # data in json in this format only
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {temp}°C, {desc}"




# for integerating the AOI feature we need the one helper function
# as for AQI we need the lattitue and longitude of the city
# helper function to make get the lattitude and longitude
def get_coordinates(city : str) :
    """Helper: convert city name to lat/lon using OpenWeather Geocoding API."""
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q" : city,
        "limit" : 1,
        "appid": api_key
    }

    res = requests.get(url, params=params)

    if res.status_code != 200 or not res.json():
        return None, None

    data = res.json()[0]
    return data["lat"], data["lon"]



# tool which will gave the current city AQI
@tool
def get_AQI(city: str) -> str:
    """Get current air quality index (AQI) for a given city name, with health advice."""
    api_key = os.getenv("OPEN_WEATHER_API_KEY")

    lat, lon = get_coordinates(city)
    if lat is None:
        return f"Error: could not find location for {city}"

    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key
    }
    res = requests.get(url, params=params)

    if res.status_code != 200:
        return f"Error: could not fetch air quality for {city}"

    data = res.json()
    aqi_number = data["list"][0]["main"]["aqi"]
    components = data["list"][0]["components"]  # PM2.5, PM10, etc ka raw data

    aqi_info = {
        1: ("Good", "Air quality is safe. No precautions needed."),
        2: ("Fair", "Acceptable air quality. Sensitive individuals should be cautious."),
        3: ("Moderate", "Consider wearing a mask if you're sensitive to pollution."),
        4: ("Poor", "Wear a mask outdoors. Avoid prolonged outdoor activity."),
        5: ("Very Poor", "Avoid going outside if possible. Wear a mask if you must.")
    }

    label, advice = aqi_info.get(aqi_number, ("Unknown", "No data available."))
    pm25 = components.get("pm2_5", "N/A")

    return (
        f"Air Quality in {city}: {label} ({aqi_number}/5)\n"
        f"PM2.5 level: {pm25} µg/m³\n"
        f"Advice: {advice}"
    )


# news tool
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_latest_news_tool(city : str) -> str :
    """Get latest news of a city."""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=5
    )

    results = response["results"]

    if not results :
        return f"No recent news found for {city}"

    news_items = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_items.append(
            f"- {title}\n  {url}\n   {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_items   )


# this tool will gave the info of nearby cities amenities like hospital, 
# pharmacy, atm, fuel, police etc
# for this tool we do not need any api-key we can directly send the req to the url
@tool
def get_nearby_amenities_tool(city: str, amenity_type: str) -> str:
    """
    Find nearby civic amenities in a city.
    Supported amenity_type values:
      - Healthcare: 'healthcare.hospital', 'healthcare.pharmacy', 'healthcare.clinic'
      - Financial: 'service.financial.atm', 'service.financial.bank'
      - Fuel: 'service.vehicle.fuel'
      - Education: 'education.school', 'education.college'
      - Food: 'catering.restaurant', 'catering.cafe'
    """
    api_key = os.getenv("GEOAPIFY_API_KEY")

    lat, long = get_coordinates(city)
    if lat is None:
        return f"Error: could not find location for {city}"

    # Fallback normalizer in case the LLM passes short forms
    alias_map = {
        "atm": "service.financial.atm",
        "bank": "service.financial.bank",
        "financial.atm": "service.financial.atm",
        "financial.bank": "service.financial.bank",
        "banking.atm": "service.financial.atm",
        "fuel": "service.vehicle.fuel",
        "commercial.fuel": "service.vehicle.fuel",
        "hospital": "healthcare.hospital",
        "pharmacy": "healthcare.pharmacy",
    }
    amenity_category = alias_map.get(amenity_type.lower(), amenity_type)

    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": amenity_category,
        "filter": f"circle:{long},{lat},5000",
        "limit": 5,
        "apiKey": api_key
    }

    res = requests.get(url, params=params)
    if res.status_code != 200:
        return f"Error: could not fetch data for '{amenity_type}' (mapped to '{amenity_category}') in {city}"

    data = res.json()
    features = data.get("features", [])
    if not features:
        return f"No {amenity_type} found within 5km in {city}."

    results = []
    for f in features:
        props = f.get("properties", {})
        name = props.get("name") or props.get("formatted", "Unnamed facility")
        results.append(f"- {name}")

    return f"Nearby {amenity_category} in {city}:\n" + "\n".join(results)



# TOOL => this will gave the gave the upcoming days weather predections
@tool
def get_weather_trend(city : str) -> str :
    """Analyze 5-day weather forecast and suggest the best day for outdoor activities."""

    api_key = os.getenv("OPEN_WEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    res = requests.get(url, params=params)

    if res.status_code != 200 :
        return f"Error: could not fetch forecast for {city}"

    data = res.json()
    forecast_list = data.get("list", [])

    if not forecast_list:
        return f"No forecast data available for {city}"

    # now we are going to take the three hour data of weather and split that data for 3 days

    daily_data = {}
    for entry in forecast_list:
        date = entry["dt_txt"].split(" ")[0]   # sahi naam
        temp = entry["main"]["temp"]
        rain_chance = entry.get("pop", 0) * 100

        if date not in daily_data:
            daily_data[date] = {"temps": [], "rain_chances": []}

        daily_data[date]["temps"].append(temp)
        daily_data[date]["rain_chances"].append(rain_chance)

    # finding the avg temp
    daily_summary = []
    for date, values in daily_data.items():
        avg_temp = sum(values["temps"]) / len(values["temps"])
        avg_rain = sum(values["rain_chances"]) / len(values["rain_chances"])
        daily_summary.append({
            "date": date,
            "avg_temp": round(avg_temp, 1),
            "avg_rain_chance": round(avg_rain, 1)
        })


    # best day dhoondo — jaha comfortable temp ho aur kam rain chance ho
    def day_score(day):
        temp_score = 100 - abs(day["avg_temp"] - 24)  # 24°C ko ideal maana
        rain_penalty = day["avg_rain_chance"]
        return temp_score - rain_penalty

    best_day = max(daily_summary, key=day_score)

    result_lines = [f"5-Day Weather Trend for {city}:\n"]
    for day in daily_summary:
        result_lines.append(f"- {day['date']}: {day['avg_temp']}°C, {day['avg_rain_chance']}% rain chance")

    result_lines.append(f"\nBest day for outdoor activities: {best_day['date']} ({best_day['avg_temp']}°C, {best_day['avg_rain_chance']}% rain chance)")

    return "\n".join(result_lines)




# liability tool
# some helper function for this tool
# 1st helper => normalize aqi
def normalize_AQI(aqi_number : int) -> int:
    """Convert AQI (1-5 scale) to a 0-100 livability score."""
    aqi_score_map = {1: 100, 2: 80, 3: 60, 4: 35, 5: 15}
    return aqi_score_map.get(aqi_number, 50)

# 2nd helper function  => normalize temp
def normalize_temperature(temp_celsius: float) -> int:
    """Convert temperature to a comfort score. Ideal range is 20-28°C."""
    if 20 <= temp_celsius <= 28:
        return 100
    elif 15 <= temp_celsius < 20 or 28 < temp_celsius <= 33:
        return 70
    elif 10 <= temp_celsius < 15 or 33 < temp_celsius <= 38:
        return 40
    else:
        return 20

# 3rd helper function => normalize amenities
def normalize_amenities(hospital_count: int) -> int:
    """Convert number of nearby hospitals to an infrastructure score."""
    if hospital_count >= 5:
        return 100
    elif hospital_count >= 3:
        return 75
    elif hospital_count >= 1:
        return 50
    else:
        return 20



@tool
def get_livability_score_tool(city: str) -> str:
    """Calculate a composite livability score (0-100) for a city based on 
    air quality, weather comfort, and healthcare accessibility."""

    weather_api_key = os.getenv("OPEN_WEATHER_API_KEY")
    geo_api_key = os.getenv("GEOAPIFY_API_KEY")

    city_lat, city_lon = get_coordinates(city)
    if city_lat is None:
        return f"Error: could not find location for {city}"

    # --- 1. AQI data ---
    aqi_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    aqi_params = {"lat": city_lat, "lon": city_lon, "appid": weather_api_key}
    aqi_res = requests.get(aqi_url, params=aqi_params)

    if aqi_res.status_code != 200:
        return f"Error: could not fetch AQI data for {city}"

    aqi_json = aqi_res.json()
    aqi_number = aqi_json["list"][0]["main"]["aqi"]
    aqi_score = normalize_AQI(aqi_number)

    # --- 2. Weather data ---
    weather_url = "https://api.openweathermap.org/data/2.5/weather"
    weather_params = {"q": city, "appid": weather_api_key, "units": "metric"}
    weather_res = requests.get(weather_url, params=weather_params)

    if weather_res.status_code != 200:
        return f"Error: could not fetch weather data for {city}"

    weather_json = weather_res.json()
    current_temp = weather_json["main"]["temp"]
    weather_score = normalize_temperature(current_temp)

    # --- 3. Amenities data (hospitals) ---
    places_url = "https://api.geoapify.com/v2/places"
    places_params = {
        "categories": "healthcare.hospital",
        "filter": f"circle:{city_lon},{city_lat},5000",
        "limit": 5,
        "apiKey": geo_api_key
    }
    places_res = requests.get(places_url, params=places_params)

    hospital_count = 0
    if places_res.status_code == 200:
        places_json = places_res.json()
        hospital_count = len(places_json.get("features", []))

    amenities_score = normalize_amenities(hospital_count)

    # --- Weighted final score ---
    final_score = round(
        (aqi_score * 0.4) + (weather_score * 0.3) + (amenities_score * 0.3)
    )

    return (
        f"Livability Score for {city}: {final_score}/100\n\n"
        f"Breakdown:\n"
        f"- Air Quality: {aqi_score}/100 (AQI level {aqi_number}/5)\n"
        f"- Weather Comfort: {weather_score}/100 ({current_temp}°C)\n"
        f"- Healthcare Access: {amenities_score}/100 ({hospital_count} hospitals within 5km)\n"
    )




# make a llm
llm = ChatMistralAI(model="open-mistral-nemo")

# listing all the tools
tools = {
    "get_weather_tool": get_weather_tool,
    "get_AQI": get_AQI,
    "get_latest_news_tool": get_latest_news_tool,
    "get_nearby_amenities_tool": get_nearby_amenities_tool,
    "get_weather_trend": get_weather_trend,
    "get_livability_score_tool": get_livability_score_tool,
}

# bind all the tools with llm
llm_bind_tools = llm.bind_tools([
    get_weather_tool,
    get_AQI,
    get_latest_news_tool,
    get_nearby_amenities_tool,
    get_weather_trend,
    get_livability_score_tool,
])

# creating an Agent tool

# for this creating one messages history
messages = []

# in this message history we all are going to have all the system message, human message and AI message

print("============== city intelligent system =============")
print(" =================== type Exit to quit =============")

while True : 
    user_inp = input("You : ")

    if user_inp.lower() == "exit":
        break

    messages.append(HumanMessage(content= user_inp))    

    while True:
        result = llm_bind_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                confirm = input(f"Agent wants to call {tool_name} — Approve? (yes/no): ")

                if confirm.lower() == "no":
                    # denied — but LLM still needs a ToolMessage response for this tool_call
                    messages.append(ToolMessage(
                        content="Tool call denied by user.",
                        tool_call_id=tool_call['id']
                    ))
                else:
                    # approved — actually run the tool
                    tool_res = tools[tool_name].invoke(tool_call)
                    messages.append(tool_res)

            continue  # loop again so LLM can see tool results and respond
        else:
            print(result.content)
            break  # yaha break bhi zaroori hai, warna infinite loop chalta rahega