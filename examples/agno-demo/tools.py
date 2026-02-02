from datetime import datetime

from agno.tools import tool


@tool
def get_current_datetime() -> dict:
    """Get the current date and time.

    Returns:
        dict: A dictionary containing:
            - date: Current date in YYYY-MM-DD format
            - time: Current time in HH:MM:SS format
            - weekday: Full name of the current day (e.g., Monday)
    """
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }

if __name__ == '__main__':
    print(get_current_datetime)