"""ADK discovers `root_agent` from the package's `agent` module."""
from . import agent
from .agent import root_agent

__all__ = ["agent", "root_agent"]
