"""Template rendering for prompts."""

import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from app.config import PROMPT_TEMPLATE


def render_prompt(context: dict) -> str:
    """Render the PROMPT.j2 template with given context."""
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(PROMPT_TEMPLATE)
    return template.render(**context)
