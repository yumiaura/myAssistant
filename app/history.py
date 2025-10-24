"""History management for storing and retrieving interactions."""

from pathlib import Path
from typing import List

from app.config import HISTORY_FILE


def save_interaction(query: str, response: str):
    """Save query and response to history file."""
    history_path = Path(HISTORY_FILE)
    
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(f"Q: {query}\n")
        f.write(f"A: {response}\n")
        f.write("---\n")


def get_recent_history(lines: int = 10) -> List[str]:
    """Get recent history lines."""
    history_path = Path(HISTORY_FILE)
    
    if not history_path.exists():
        return []
    
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:] if line.strip()]
    
    except FileNotFoundError:
        return []
