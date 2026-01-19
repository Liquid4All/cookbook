#!/usr/bin/env python3
"""
Flight Search Assistant with CLI
Supports both OpenAI models and local models via llama-server
"""

from openai import OpenAI
import csv
import json
from datetime import datetime, timedelta
import random
import argparse
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Union

# Path to the flights data CSV
FLIGHTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "sample_flights.csv")

# Type aliases for clarity
Airport = Dict[str, str]
FlightData = Dict[str, Any]
FlightSearchResult = Dict[str, Any]
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
    "BCN": {"name": "Barcelona-El Prat", "city": "Barcelona", "country": "Spain"},
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
            "name": "book_flight",
            "description": "Book a flight by its flight number for a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {
                        "type": "string",
                        "description": "The flight number to book (e.g., 'AA123', 'LH456')"
                    },
                    "date": {
                        "type": "string",
                        "description": "The date of the flight in YYYY-MM-DD format"
                    }
                },
                "required": ["flight_number", "date"]
            }
        }
    },
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


def search_flights(
    departure: str,
    destination: str,
    date: str,
    max_results: int = 5
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
    departure = normalize_location(departure)
    destination = normalize_location(destination)

    if departure not in AIRPORTS or destination not in AIRPORTS:
        return {
            "error": f"Unknown airport code. Departure: {departure}, Destination: {destination}",
            "available_airports": list(AIRPORTS.keys())
        }

    flights: List[FlightData] = []

    with open(FLIGHTS_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["departure_airport"] == departure and
                row["destination_airport"] == destination and
                row["date"] == date):

                layover_airports = row["layover_airports"].split(";") if row["layover_airports"] else []

                flights.append({
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
                    "available_seats": int(row["available_seats"])
                })

    flights.sort(key=lambda x: x["price_usd"])
    flights = flights[:max_results]

    return {
        "departure": AIRPORTS[departure],
        "destination": AIRPORTS[destination],
        "search_date": date,
        "total_flights_found": len(flights),
        "flights": flights
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
                        "message": "Flight booked successfully!"
                    }
                else:
                    return {
                        "status": "failed",
                        "flight_number": flight_number,
                        "date": date,
                        "error": "No available seats on this flight"
                    }

    return {
        "status": "failed",
        "flight_number": flight_number,
        "date": date,
        "error": "Flight not found"
    }


# Available functions mapping
available_functions: Dict[str, Any] = {
    "search_flights": search_flights,
    "book_flight": book_flight
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
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable.")
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
    verbose: bool = False,
    max_iterations: int = 5
) -> Messages:
    """
    Run a complete conversation with tool calling.

    Args:
        client: OpenAI client instance
        model_name: Name of the model to use
        user_message: User's input message
        verbose: Whether to show detailed output
        max_iterations: Maximum number of iterations (model call -> tool execution -> model call)

    Returns:
        List of message dictionaries representing the conversation history
    """
    messages: Messages = [
        {
            "role": "system",
            # "content": "You are a helpful flight search assistant. You can help users find flights, compare options, and get flight details. Always provide clear, organized information about flights including prices, duration, and layovers."
            "content": """You are an AI assistant with access to a set of tools.
When a user asks a question, determine if a tool should be called to help answer.
If a tool is needed, respond with a tool call using the following format:
Each tool function call should use json-like syntax, e.g., {"name": "speak", "arguments": {"name": "Hello"}}.
If no tool is needed, answer the user directly.
Always use the most relevant tool(s) for the user request.
If a tool returns an error, explain the error to the user.
Be concise and helpful.
force json schema."""
        },
        {"role": "user", "content": user_message}
    ]

    if verbose:
        print(f"\n{'='*80}")
        print(f"User: {user_message}")
        print(f"{'='*80}\n")

    for iteration in range(max_iterations):
        # API call
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                # temperature=0.1,
                # extra_body={
                #     # "top_k": 20,
                #     # "repetition_penalty": 1.05
                # }
            )
        except Exception as e:
            print(f"❌ Error calling model: {e}")
            return messages

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # breakpoint()

        # Check if the model wants to call tools
        if not assistant_message.tool_calls:
            # No tool calls - print final response and exit loop
            if verbose:
                print("💬 Assistant Response:\n")
            print(assistant_message.content)
            break

        # Handle tool calls
        if verbose:
            print(f"🔧 Tool Calls Requested (iteration {iteration + 1}/{max_iterations})\n")

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

            function_response: FlightSearchResult = function_to_call(**function_args)

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
  python flight_search.py --model gpt-4 --prompt "Find flights from NYC to London on 2026-03-15"
  
  # Use local model on default port (8080)
  python flight_search.py --port 8080 --prompt "Search flights from Belgrade to Paris"
  
  # Interactive mode with OpenAI
  python flight_search.py --model gpt-4 --interactive
  
  # Interactive mode with local model
  python flight_search.py --port 8080 --interactive
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        # default="gpt-4",
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