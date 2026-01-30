from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .connection_manager import ConnectionManager

router = APIRouter()


def create_functions_router(manager: ConnectionManager) -> APIRouter:
    """Create and configure the functions API router."""

    @router.get("/functions.json")
    async def get_functions():
        """
        Get all available cockpit functions in JSON format.
        Fetches definitions from the frontend via WebSocket.
        """
        if not manager.active_connections:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "No active cockpit connections. Please open the UI in a browser first.",
                },
            )

        ws = manager.active_connections[0]

        try:
            functions = await manager.send_rpc_request(ws, "system.getFunctions", {})
            return JSONResponse(content={"functions": functions})
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)},
            )

    @router.get("/debug/get-functions-matching/{query}")
    async def debug_get_functions_matching(query: str):
        """
        Debug endpoint: Search for functions matching a query string.
        Prints results to terminal and returns JSON response.

        Args:
            query: Search string to match against function names (case-insensitive)
        """
        if not manager.active_connections:
            print("\n[DEBUG] No active connections. Please open the UI in a browser first.\n")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "No active cockpit connections. Please open the UI in a browser first.",
                },
            )

        ws = manager.active_connections[0]

        try:
            # Fetch all functions from the frontend
            all_functions = await manager.send_rpc_request(ws, "system.getFunctions", {})

            # Filter functions whose name contains the query (case-insensitive)
            query_lower = query.lower()
            matching = [f for f in all_functions if query_lower in f["name"].lower()]

            # Print to terminal
            print("\n" + "=" * 80)
            print(f"[DEBUG] Function search results for query: '{query}'")
            print("=" * 80)
            print(f"Total matches: {len(matching)}")
            print()

            if matching:
                print("First 5 matching functions:\n")
                for i, func in enumerate(matching[:5], 1):
                    print(f"{i}. {func['name']}")
                    print(f"   Description: {func['description']}")
                    print(f"   Parameters: {func['parameters']}")
                    print()
            else:
                print("No functions matched the query.")

            print("=" * 80 + "\n")

            # Return JSON response
            return JSONResponse(
                content={
                    "status": "success",
                    "query": query,
                    "total_matches": len(matching),
                    "first_5": matching[:5],
                    "all_matches": matching,
                }
            )

        except Exception as e:
            error_msg = f"Error fetching functions: {str(e)}"
            print(f"\n[DEBUG ERROR] {error_msg}\n")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": error_msg},
            )

    return router
