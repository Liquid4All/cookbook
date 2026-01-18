import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .connection_manager import ConnectionManager

router = APIRouter()


def create_checklist_router(manager: ConnectionManager) -> APIRouter:
    """Create and configure the checklist testing router."""

    @router.get("/checklist/quick-check")
    async def quick_check():
        """
        Quick system check - tests a few basic functions:
        - Open and close front-right window
        - Set climate temperature
        - Get system state
        """
        if not manager.active_connections:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "No active cockpit connections"},
            )

        ws = manager.active_connections[0]
        results = []

        try:
            # Test 1: Open front-right window
            results.append(
                {
                    "test": "Open front-right window",
                    "result": await manager.send_rpc_request(ws, "carWindows.set", {"id": "fr", "open": True}),
                }
            )
            await asyncio.sleep(0.5)

            # Test 2: Check window state
            window_state = await manager.send_rpc_request(ws, "carWindows.get", {"id": "fr"})
            results.append(
                {
                    "test": "Verify window opened",
                    "result": window_state,
                    "passed": window_state,
                }
            )

            # Test 3: Close window
            results.append(
                {
                    "test": "Close front-right window",
                    "result": await manager.send_rpc_request(ws, "carWindows.set", {"id": "fr", "open": False}),
                }
            )
            await asyncio.sleep(0.5)

            # Test 4: Set climate temperature
            results.append(
                {
                    "test": "Set temperature to 22°C",
                    "result": await manager.send_rpc_request(ws, "climate.setTarget", {"temperature": 22}),
                }
            )
            await asyncio.sleep(0.5)

            # Test 5: Verify climate state
            climate_state = await manager.send_rpc_request(ws, "climate.get", {})
            results.append(
                {
                    "test": "Verify temperature set",
                    "result": climate_state,
                    "passed": climate_state.get("targetTemp") == 22,
                }
            )

            # Test 6: Get full system state
            state = await manager.send_rpc_request(ws, "system.getState", {})
            results.append(
                {
                    "test": "Get system state",
                    "result": "success" if state else "failed",
                    "state_keys": list(state.keys()) if state else [],
                }
            )

            return {"status": "success", "tests_run": len(results), "results": results}

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "partial_results": results},
            )

    @router.get("/checklist/full-system")
    async def full_system_check():
        """
        Comprehensive system test - exercises all major functions:
        - All window controls
        - Media player controls
        - Climate controls
        - Navigation controls
        """
        if not manager.active_connections:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "No active cockpit connections"},
            )

        ws = manager.active_connections[0]
        results = []

        try:
            # === WINDOWS TESTS ===
            # Test opening all windows
            results.append(
                {
                    "test": "Open all windows",
                    "result": await manager.send_rpc_request(ws, "carWindows.openAll", {}),
                }
            )
            await asyncio.sleep(0.5)

            # Verify all windows open
            windows = await manager.send_rpc_request(ws, "carWindows.get", {})
            results.append(
                {
                    "test": "Verify all windows open",
                    "result": windows,
                    "passed": all(windows.values()),
                }
            )

            # Close all windows
            results.append(
                {
                    "test": "Close all windows",
                    "result": await manager.send_rpc_request(ws, "carWindows.closeAll", {}),
                }
            )
            await asyncio.sleep(0.5)

            # Test individual window toggle
            results.append(
                {
                    "test": "Toggle rear-left window",
                    "result": await manager.send_rpc_request(ws, "carWindows.toggle", {"id": "rl"}),
                }
            )
            await asyncio.sleep(0.3)

            # === MEDIA TESTS ===
            # Get initial media state
            media_state = await manager.send_rpc_request(ws, "media.get", {})
            results.append(
                {
                    "test": "Get media state",
                    "result": f"Track: {media_state['track']['title']}",
                    "passed": "track" in media_state,
                }
            )

            # Play media
            results.append(
                {
                    "test": "Play media",
                    "result": await manager.send_rpc_request(ws, "media.play", {}),
                }
            )
            await asyncio.sleep(0.5)

            # Verify playing
            media_state = await manager.send_rpc_request(ws, "media.get", {})
            results.append(
                {
                    "test": "Verify media playing",
                    "result": media_state.get("isPlaying"),
                    "passed": media_state.get("isPlaying"),
                }
            )

            # Next track
            results.append(
                {
                    "test": "Next track",
                    "result": await manager.send_rpc_request(ws, "media.next", {}),
                }
            )
            await asyncio.sleep(0.3)

            # Previous track
            results.append(
                {
                    "test": "Previous track",
                    "result": await manager.send_rpc_request(ws, "media.previous", {}),
                }
            )
            await asyncio.sleep(0.3)

            # Pause
            results.append(
                {
                    "test": "Pause media",
                    "result": await manager.send_rpc_request(ws, "media.pause", {}),
                }
            )
            await asyncio.sleep(0.3)

            # === CLIMATE TESTS ===
            # Set target temperature
            results.append(
                {
                    "test": "Set temperature to 24°C",
                    "result": await manager.send_rpc_request(ws, "climate.setTarget", {"temperature": 24}),
                }
            )
            await asyncio.sleep(0.3)

            # Set fan level
            results.append(
                {
                    "test": "Set fan to level 3",
                    "result": await manager.send_rpc_request(ws, "climate.setFan", {"level": 3}),
                }
            )
            await asyncio.sleep(0.3)

            # Verify climate state
            climate = await manager.send_rpc_request(ws, "climate.get", {})
            results.append(
                {
                    "test": "Verify climate settings",
                    "result": climate,
                    "passed": climate.get("targetTemp") == 24 and climate.get("fanLevel") == 3,
                }
            )

            # Reset to defaults
            await manager.send_rpc_request(ws, "climate.setTarget", {"temperature": 23})
            await manager.send_rpc_request(ws, "climate.setFan", {"level": 2})

            # === NAVIGATION TESTS ===
            # Set destination
            results.append(
                {
                    "test": "Set navigation destination",
                    "result": await manager.send_rpc_request(ws, "navigation.setDestination", {}),
                }
            )
            await asyncio.sleep(0.3)

            # Get navigation state
            nav_state = await manager.send_rpc_request(ws, "navigation.get", {})
            results.append(
                {
                    "test": "Verify route generated",
                    "result": f"Route has {nav_state.get('totalSteps', 0)} steps",
                    "passed": nav_state.get("totalSteps", 0) > 0,
                }
            )

            # Start navigation
            results.append(
                {
                    "test": "Start navigation",
                    "result": await manager.send_rpc_request(ws, "navigation.start", {}),
                }
            )
            await asyncio.sleep(1.0)

            # Pause navigation
            results.append(
                {
                    "test": "Pause navigation",
                    "result": await manager.send_rpc_request(ws, "navigation.pause", {}),
                }
            )
            await asyncio.sleep(0.3)

            # Clear navigation
            results.append(
                {
                    "test": "Clear navigation",
                    "result": await manager.send_rpc_request(ws, "navigation.clear", {}),
                }
            )
            await asyncio.sleep(0.3)

            # === FINAL STATE CHECK ===
            final_state = await manager.send_rpc_request(ws, "system.getState", {})
            results.append(
                {
                    "test": "Final system state check",
                    "result": "success",
                    "state": final_state,
                }
            )

            passed_tests = sum(1 for r in results if r.get("passed", True))
            total_tests = len(results)

            return {
                "status": "success",
                "summary": f"{passed_tests}/{total_tests} tests passed",
                "tests_run": total_tests,
                "tests_passed": passed_tests,
                "results": results,
            }

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "partial_results": results},
            )

    return router
