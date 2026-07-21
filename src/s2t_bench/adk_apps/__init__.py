"""Dedicated agents directory for ADK discovery.

ADK's get_fast_api_app(agents_dir=...) treats every subdirectory here as one
agent. Keeping this folder separate from the rest of the package means ADK never
tries to load engines/ or benchmark/ (which aren't agents) and choke.
Each subpackage just re-exports an existing root_agent.
"""
