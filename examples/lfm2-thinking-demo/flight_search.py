#!/usr/bin/env python3
"""
Flight Search Assistant with CLI
Supports both OpenAI models and local models via llama-server
"""

from openai import OpenAI
import json
from datetime import datetime, timedelta
import random
import argparse
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Union

# Type aliases for clarity
Airport = Dict[str, str]
FlightData = Dict[str, Any]
FlightSearchResult = Dict[str, Any]
FlightDetails = Dict[str, Any]
Messages = List[Dict[str, Any]]

# Airport database
AIRPORTS: Dict[str, Airport] = {
    "JFK": {"name": "John F. Kennedy International", "city": "New York", "country": "USA"},
    "LAX": {"name": "Los Angeles International", "city": "Los Angeles", "country": "USA"},
    "LHR": {"name": "London Heathrow", "city": "London", "country": "UK"},
    "CDG": {"name": "Charles de Gaulle", "city": "Paris", "country": "France"},
    "BEG": {"name": "Belgrade Nikola Tesla", "city": "Belgrade", "country": "Serbia"},
    "DXB": {"name": "Dubai International", "city": "Dubai", "country": "UAE"},
    "FRA": {"name": "Frankfurt Airport", "city": "Frankfurt", "country": "Germany"},
    "AMS": {"name": "Amsterdam Schiphol", "city": "Amsterdam", "country": "Netherlands"},
    "IST": {"name": "Istanbul Airport", "city": "Istanbul", "country": "Turkey"},
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
}

# Tool definitions
tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two airports on a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "Departure airport code (e.g., 'JFK', 'LAX', 'BEG') or city name"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport code (e.g., 'CDG', 'LHR', 'DXB') or city name"
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of flights to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["departure", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_details",
            "description": "Get detailed information about a specific flight by flight number",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {
                        "type": "string",
                        "description": "Flight number (e.g., 'AA123', 'LH456')"
                    },
                    "date": {
                        "type": "string",
                        "description": "Flight date in YYYY-MM-DD format"
                    }
                },
                "required": ["flight_number", "date"]
            }
        }
    }
]


def normalize_location(location: str) -> str:
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


def calculate_distance(departure: str, destination: str) -> int:
    """
    Calculate approximate distance between airports in kilometers.
    
    Args:
        departure: Departure airport code
        destination: Destination airport code
        
    Returns:
        Distance in kilometers
    """
    distances: Dict[Tuple[str, str], int] = {
        ("JFK", "LHR"): 3451, ("JFK", "CDG"): 3625, ("LAX", "LHR"): 5456,
        ("BEG", "LHR"): 1056, ("BEG", "CDG"): 862, ("BEG", "JFK"): 4647,
        ("DXB", "JFK"): 6847, ("DXB", "LHR"): 3414, ("IST", "BEG"): 522,
        ("FRA", "BEG"): 740,
    }
    key: Tuple[str, str] = (departure, destination)
    reverse_key: Tuple[str, str] = (destination, departure)
    return distances.get(key, distances.get(reverse_key, 2000))


def search_flights(
    departure: str, 
    destination: str, 
    date: str, 
    max_results: int = 5
) -> FlightSearchResult:
    """
    Search for flights between two airports.
    
    Args:
        departure: Departure airport code or city name
        destination: Destination airport code or city name
        date: Departure date in YYYY-MM-DD format
        max_results: Maximum number of flights to return
        
    Returns:
        Dictionary containing flight search results with departure/destination info and list of flights
    """
    departure = normalize_location(departure)
    destination = normalize_location(destination)
    
    if departure not in AIRPORTS or destination not in AIRPORTS:
        return {
            "error": f"Unknown airport code. Departure: {departure}, Destination: {destination}",
            "available_airports": list(AIRPORTS.keys())
        }
    
    distance: int = calculate_distance(departure, destination)
    airlines: List[str] = ["AA", "BA", "LH", "AF", "EK", "TK", "JU"]
    flights: List[FlightData] = []
    
    for i in range(min(max_results, 5)):
        airline: str = random.choice(airlines)
        flight_num: str = f"{airline}{random.randint(100, 999)}"
        base_duration_hours: float = distance / 550
        duration_minutes: int = int(base_duration_hours * 60) + random.randint(-30, 30)
        hours: int = duration_minutes // 60
        minutes: int = duration_minutes % 60
        
        num_layovers: int
        if distance > 4000:
            num_layovers = random.choice([1, 2])
        elif distance > 2000:
            num_layovers = random.choice([0, 1])
        else:
            num_layovers = 0
        
        layover_cities: List[str] = []
        if num_layovers > 0:
            possible_hubs: List[str] = ["FRA", "AMS", "IST", "DXB"]
            layover_cities = random.sample(
                [code for code in possible_hubs if code not in [departure, destination]], 
                num_layovers
            )
        
        departure_hour: int = 6 + (i * 3)
        departure_time: str = f"{departure_hour:02d}:{random.randint(0, 59):02d}"
        arrival_hour: int = (departure_hour + hours + (num_layovers * 2)) % 24
        arrival_time: str = f"{arrival_hour:02d}:{random.randint(0, 59):02d}"
        base_price: float = distance * 0.15 + (num_layovers * 50)
        price: float = round(base_price + random.uniform(-100, 100), 2)
        
        flights.append({
            "flight_number": flight_num,
            "airline": airline,
            "departure_airport": departure,
            "departure_city": AIRPORTS[departure]["city"],
            "destination_airport": destination,
            "destination_city": AIRPORTS[destination]["city"],
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "date": date,
            "distance_km": distance,
            "duration": f"{hours}h {minutes}m",
            "duration_minutes": duration_minutes,
            "layovers": num_layovers,
            "layover_airports": [AIRPORTS[code]["city"] for code in layover_cities],
            "price_usd": price,
            "available_seats": random.randint(5, 150)
        })
    
    flights.sort(key=lambda x: x["price_usd"])
    
    return {
        "departure": AIRPORTS[departure],
        "destination": AIRPORTS[destination],
        "search_date": date,
        "total_flights_found": len(flights),
        "flights": flights
    }


def get_flight_details(flight_number: str, date: str) -> FlightDetails:
    """
    Get detailed information about a specific flight.
    
    Args:
        flight_number: Flight number (e.g., 'AA123')
        date: Flight date in YYYY-MM-DD format
        
    Returns:
        Dictionary containing detailed flight information
    """
    airline_code: str = ''.join(filter(str.isalpha, flight_number))
    
    flight_info: FlightDetails = {
        "flight_number": flight_number,
        "airline": airline_code,
        "date": date,
        "aircraft": random.choice(["Boeing 737-800", "Airbus A320", "Boeing 787-9", "Airbus A350"]),
        "status": random.choice(["On Time", "Delayed 15 min", "Boarding", "Departed"]),
        "departure": {
            "airport": "JFK",
            "terminal": random.choice(["1", "4", "8"]),
            "gate": f"{random.choice(['A', 'B', 'C'])}{random.randint(1, 30)}",
            "scheduled_time": "14:30",
            "actual_time": "14:30"
        },
        "arrival": {
            "airport": "LHR",
            "terminal": random.choice(["2", "3", "5"]),
            "gate": f"{random.choice(['A', 'B', 'C'])}{random.randint(1, 30)}",
            "scheduled_time": "03:45",
            "estimated_time": "03:45"
        },
        "baggage_claim": random.randint(1, 12),
        "amenities": {
            "wifi": True,
            "entertainment": True,
            "power_outlets": True,
            "meal_service": True
        }
    }
    
    return flight_info


# Available functions mapping
available_functions: Dict[str, Any] = {
    "search_flights": search_flights,
    "get_flight_details": get_flight_details
}


def setup_client(args: argparse.Namespace) -> Tuple[OpenAI, str]:
    """
    Setup OpenAI client based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (OpenAI client instance, model name string)
    """
    if args.model in ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"]:
        # OpenAI model
        api_key: Optional[str] = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
            sys.exit(1)
        
        client: OpenAI = OpenAI(api_key=api_key)
        model_name: str = args.model
        print(f"🌐 Using OpenAI model: {model_name}")
        
    else:
        # Local model via llama-server
        base_url: str = f"http://{args.host}:{args.port}/v1"
        client = OpenAI(
            base_url=base_url,
            api_key="not-needed"
        )
        model_name = args.model
        print(f"🖥️  Using local model: {model_name}")
        print(f"📍 Endpoint: {base_url}")
    
    return client, model_name


def run_conversation(
    client: OpenAI, 
    model_name: str, 
    user_message: str, 
    verbose: bool = False
) -> Messages:
    """
    Run a complete conversation with tool calling.
    
    Args:
        client: OpenAI client instance
        model_name: Name of the model to use
        user_message: User's input message
        verbose: Whether to show detailed output
        
    Returns:
        List of message dictionaries representing the conversation history
    """
    messages: Messages = [
        {
            "role": "system", 
            "content": "You are a helpful flight search assistant. You can help users find flights, compare options, and get flight details. Always provide clear, organized information about flights including prices, duration, and layovers."
        },
        {"role": "user", "content": user_message}
    ]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"User: {user_message}")
        print(f"{'='*80}\n")
    
    # First API call
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7
        )
    except Exception as e:
        print(f"❌ Error calling model: {e}")
        return messages
    
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    
    # Check if the model wants to call tools
    if assistant_message.tool_calls:
        if verbose:
            print("🔧 Tool Calls Requested\n")
        
        # Handle multiple tool calls
        for tool_call in assistant_message.tool_calls:
            function_name: str = tool_call.function.name
            function_args: Dict[str, Any] = json.loads(tool_call.function.arguments)
            
            if verbose:
                print(f"  Function: {function_name}")
                print(f"  Arguments: {json.dumps(function_args, indent=4)}")
            
            # Execute the function
            function_to_call = available_functions.get(function_name)
            if not function_to_call:
                print(f"⚠️  Unknown function: {function_name}")
                continue
                
            function_response: Union[FlightSearchResult, FlightDetails] = function_to_call(**function_args)
            
            if verbose:
                if 'flights' in function_response:
                    print(f"  ✓ Found {function_response.get('total_flights_found', 0)} flights\n")
                else:
                    print(f"  ✓ Retrieved flight details\n")
            
            # Add the function result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response)
            })
        
        # Get final response from the model with tool results
        try:
            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools
            )
            
            if verbose:
                print("💬 Assistant Response:\n")
            print(final_response.choices[0].message.content)
            
        except Exception as e:
            print(f"❌ Error getting final response: {e}")
        
    else:
        # No tool calls needed
        if verbose:
            print("💬 Assistant Response (No Tools Needed):\n")
        print(assistant_message.content)
    
    return messages


def interactive_mode(client: OpenAI, model_name: str) -> None:
    """
    Run in interactive mode with continuous conversation.
    
    Args:
        client: OpenAI client instance
        model_name: Name of the model to use
    """
    print("\n" + "="*80)
    print("✈️  Flight Search Assistant - Interactive Mode")
    print("="*80)
    print("Type 'exit' or 'quit' to end the conversation\n")
    
    while True:
        try:
            user_input: str = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print()
            run_conversation(client, model_name, user_input, verbose=True)
            print("\n" + "-"*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main() -> None:
    """
    Main entry point for the CLI application.
    """
    parser = argparse.ArgumentParser(
        description="Flight Search Assistant with OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use OpenAI GPT-4
  python flight-search.py --model gpt-4 --prompt "Find flights from NYC to London on 2026-03-15"
  
  # Use local model on default port (8080)
  python flight-search.py --model lfm-1.2 --prompt "Search flights from Belgrade to Paris"
  
  # Use local model on custom port
  python flight-search.py --model lfm-1.2 --port 8081 --prompt "Flight to Dubai"
  
  # Interactive mode with OpenAI
  python flight-search.py --model gpt-4 --interactive
  
  # Interactive mode with local model
  python flight-search.py --model lfm-1.2 --interactive
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model to use (e.g., 'gpt-4', 'gpt-4o', 'lfm-1.2'). Default: lfm-1.2"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for local llama-server (default: 8080)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for local llama-server (default: localhost)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="User prompt/query for flight search"
    )
    
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    args: argparse.Namespace = parser.parse_args()
    
    # Setup client
    client: OpenAI
    model_name: str
    client, model_name = setup_client(args)
    
    # Run in interactive or single-query mode
    if args.interactive:
        interactive_mode(client, model_name)
    elif args.prompt:
        run_conversation(client, model_name, args.prompt, verbose=args.verbose or True)
    else:
        print("Error: Either --prompt or --interactive is required")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()