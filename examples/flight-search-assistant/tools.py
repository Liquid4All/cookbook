import csv
import os
import random
from typing import Any, Dict, List

# Path to the flights data CSV
FLIGHTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "sample_flights.csv")

# Type aliases for clarity
Airport = Dict[str, str]
FlightData = Dict[str, Any]
FlightSearchResult = Dict[str, Any]

# Airport database
AIRPORTS: Dict[str, Airport] = {
    "JFK": {
        "name": "John F. Kennedy International",
        "city": "New York",
        "country": "USA",
    },
    "LAX": {
        "name": "Los Angeles International",
        "city": "Los Angeles",
        "country": "USA",
    },
    "LHR": {"name": "London Heathrow", "city": "London", "country": "UK"},
    "CDG": {"name": "Charles de Gaulle", "city": "Paris", "country": "France"},
    "BEG": {"name": "Belgrade Nikola Tesla", "city": "Belgrade", "country": "Serbia"},
    "DXB": {"name": "Dubai International", "city": "Dubai", "country": "UAE"},
    "FRA": {"name": "Frankfurt Airport", "city": "Frankfurt", "country": "Germany"},
    "AMS": {
        "name": "Amsterdam Schiphol",
        "city": "Amsterdam",
        "country": "Netherlands",
    },
    "IST": {"name": "Istanbul Airport", "city": "Istanbul", "country": "Turkey"},
    "BCN": {"name": "Barcelona-El Prat", "city": "Barcelona", "country": "Spain"},
    "BOS": {"name": "Boston Logan International", "city": "Boston", "country": "USA"},
}

CITY_TO_AIRPORT: Dict[str, str] = {
    "New York": "JFK",
    "Los Angeles": "LAX",
    "London": "LHR",
    "Paris": "CDG",
    "Belgrade": "BEG",
    "Dubai": "DXB",
    "Frankfurt": "FRA",
    "Amsterdam": "AMS",
    "Istanbul": "IST",
    "Barcelona": "BCN",
    "Boston": "BOS",
}


def _to_airport_code(location: str) -> str:
    """
    Convert city name to airport code if needed.

    Args:
        location: City name or airport code

    Returns:
        Airport code in uppercase
    """
    location = location.strip()
    if location.upper() in AIRPORTS:
        return location.upper()
    for city, code in CITY_TO_AIRPORT.items():
        if city.lower() in location.lower():
            return code
    return location.upper()


def search_flights(
    departure: str, destination: str, date: str, max_results: int = 5
) -> FlightSearchResult:
    """
    Search for flights between two airports from the CSV database.

    Args:
        departure: Departure airport code or city name
        destination: Destination airport code or city name
        date: Departure date in YYYY-MM-DD format
        max_results: Maximum number of flights to return

    Returns:
        Dictionary containing flight search results with departure/destination info and list of flights
    """
    departure = _to_airport_code(departure)
    destination = _to_airport_code(destination)

    if departure not in AIRPORTS or destination not in AIRPORTS:
        return {
            "error": f"Unknown airport code. Departure: {departure}, Destination: {destination}",
            "available_airports": list(AIRPORTS.keys()),
        }

    flights: List[FlightData] = []

    with open(FLIGHTS_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["departure_airport"] == departure
                and row["destination_airport"] == destination
                and row["date"] == date
            ):
                layover_airports = (
                    row["layover_airports"].split(";")
                    if row["layover_airports"]
                    else []
                )

                flights.append(
                    {
                        "flight_number": row["flight_number"],
                        "airline": row["airline"],
                        "departure_airport": row["departure_airport"],
                        "departure_city": row["departure_city"],
                        "destination_airport": row["destination_airport"],
                        "destination_city": row["destination_city"],
                        "departure_time": row["departure_time"],
                        "arrival_time": row["arrival_time"],
                        "date": row["date"],
                        "distance_km": int(row["distance_km"]),
                        "duration": row["duration"],
                        "duration_minutes": int(row["duration_minutes"]),
                        "layovers": int(row["layovers"]),
                        "layover_airports": layover_airports,
                        "price_usd": float(row["price_usd"]),
                        "available_seats": int(row["available_seats"]),
                    }
                )

    flights.sort(key=lambda x: x["price_usd"])
    flights = flights[:max_results]

    return {
        "departure": AIRPORTS[departure],
        "destination": AIRPORTS[destination],
        "search_date": date,
        "total_flights_found": len(flights),
        "flights": flights,
    }


def book_flight(flight_number: str, date: str) -> Dict[str, Any]:
    """
    Book a flight by its flight number and date.

    Args:
        flight_number: The flight number to book
        date: The date of the flight in YYYY-MM-DD format

    Returns:
        Dictionary containing booking confirmation or error
    """
    # Search for the flight in the CSV
    with open(FLIGHTS_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["flight_number"] == flight_number and row["date"] == date:
                if int(row["available_seats"]) > 0:
                    return {
                        "status": "confirmed",
                        "booking_reference": f"BK{random.randint(100000, 999999)}",
                        "flight_number": flight_number,
                        "date": date,
                        "airline": row["airline"],
                        "departure_airport": row["departure_airport"],
                        "departure_city": row["departure_city"],
                        "destination_airport": row["destination_airport"],
                        "destination_city": row["destination_city"],
                        "departure_time": row["departure_time"],
                        "arrival_time": row["arrival_time"],
                        "price_usd": float(row["price_usd"]),
                        "message": "Flight booked successfully!",
                    }
                else:
                    return {
                        "status": "failed",
                        "flight_number": flight_number,
                        "date": date,
                        "error": "No available seats on this flight",
                    }

    return {
        "status": "failed",
        "flight_number": flight_number,
        "date": date,
        "error": "Flight not found",
    }


# Available functions mapping
tool_registry: Dict[str, Any] = {
    "search_flights": search_flights,
    "book_flight": book_flight,
}

# Tool definitions
tool_schemas: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two airports on a specific date. Returns detailed information for each flight including: flight number, airline, departure/arrival times, duration, number of layovers, price in USD, and available seats. Use this to compare flight options and prices before booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "Departure airport code (e.g., 'JFK', 'LAX', 'BEG') or city name",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport code (e.g., 'CDG', 'LHR', 'DXB') or city name",
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of flights to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["departure", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight by its flight number for a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {
                        "type": "string",
                        "description": "The flight number to book (e.g., 'AA123', 'LH456')",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date of the flight in YYYY-MM-DD format",
                    },
                },
                "required": ["flight_number", "date"],
            },
        },
    },
]
