"""Test that every user-facing MCP tool has a corresponding playground UI entry.

Parses tool registrations from the MCP server source, playground tool keys from
tools.html, and API route handlers from profile-server.py. Ensures nothing is
exposed via MCP without a way to test it in the playground.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MCP_SERVER = REPO / "mcp" / "local-models-server.py"
TOOLS_HTML = REPO / "app" / "tools.html"
PROFILE_SERVER = REPO / "app" / "profile-server.py"

# MCP tools that are infrastructure/plumbing and intentionally excluded
# from the playground (no user-visible UX makes sense for these).
EXCLUDED_MCP_TOOLS = {
    "local_models_status",   # informational, no interactive test
    "local_candidates",      # parallel multi-model query, no single-model UX
    "local_dispatch",        # background job start (async plumbing)
    "local_collect",         # background job collect (async plumbing)
    "local_similarity_search",  # search tool, no standalone playground card
}

# Mapping from MCP tool function name -> playground tool key.
# Most follow the pattern local_X -> X, but some differ.
MCP_TO_PLAYGROUND = {
    "local_generate": {"code", "general", "unfiltered"},
    "local_review": {"review"},
    "local_vision": {"vision"},
    "local_computer_use": {"computer_use"},
    "local_image": {"image_gen"},
    "local_image_edit": {"image_edit"},
    "local_transcribe": {"transcribe"},
    "local_speak": {"speak"},
    "local_translate": {"translate"},
    "local_summarize": {"summarize"},
    "local_embed": {"embed"},
    "local_video": {"video"},
}


def _parse_mcp_tools():
    """Extract @mcp.tool() function names from the MCP server source."""
    src = MCP_SERVER.read_text()
    # Pattern: @mcp.tool() followed by async def <name>(
    return set(re.findall(
        r"@mcp\.tool\(\)\s+async\s+def\s+(\w+)\s*\(", src))


def _parse_playground_tools():
    """Extract tool keys from the TOOLS object in tools.html."""
    src = TOOLS_HTML.read_text()
    # Match the start of the TOOLS block, then extract keys
    match = re.search(r"const TOOLS\s*=\s*\{(.+?)\};", src, re.DOTALL)
    assert match, "Could not find TOOLS object in tools.html"
    block = match.group(1)
    # Keys are bare identifiers followed by colon or { at start of line/after comma
    return set(re.findall(r"(\w+)\s*:\s*\{", block))


def _parse_api_routes():
    """Extract tool names handled by /api/test in profile-server.py."""
    src = PROFILE_SERVER.read_text()
    # Match: tool == "X" or tool in ("X", "Y")
    single = set(re.findall(r'tool\s*==\s*["\'](\w+)["\']', src))
    multi = re.findall(r'tool\s+in\s*\(([^)]+)\)', src)
    for group in multi:
        single.update(re.findall(r'["\'](\w+)["\']', group))
    return single


class TestPlaygroundCoverage:
    def test_every_mcp_tool_has_playground_entry(self):
        """Every user-facing MCP tool must have a playground UI card."""
        mcp_tools = _parse_mcp_tools()
        playground_tools = _parse_playground_tools()

        missing = []
        for tool in sorted(mcp_tools - EXCLUDED_MCP_TOOLS):
            expected_keys = MCP_TO_PLAYGROUND.get(tool)
            if expected_keys is None:
                missing.append(f"{tool} (no mapping in MCP_TO_PLAYGROUND)")
                continue
            for key in expected_keys:
                if key not in playground_tools:
                    missing.append(f"{tool} -> playground key '{key}'")

        assert not missing, (
            f"MCP tools missing playground UI:\n  " + "\n  ".join(missing))

    def test_every_playground_tool_has_api_route(self):
        """Every playground UI card must have a backend API handler."""
        playground_tools = _parse_playground_tools()
        api_routes = _parse_api_routes()

        missing = playground_tools - api_routes
        assert not missing, (
            f"Playground tools missing API route in profile-server.py: {missing}")

    def test_every_mcp_tool_is_mapped_or_excluded(self):
        """Every MCP tool must appear in either the mapping or exclusion list."""
        mcp_tools = _parse_mcp_tools()
        mapped = set(MCP_TO_PLAYGROUND.keys())
        accounted = mapped | EXCLUDED_MCP_TOOLS

        unmapped = mcp_tools - accounted
        assert not unmapped, (
            f"MCP tools not mapped or excluded: {unmapped}\n"
            "Add to MCP_TO_PLAYGROUND or EXCLUDED_MCP_TOOLS in test_playground_coverage.py")

    def test_playground_tools_not_empty(self):
        """Sanity check: parsers found tools."""
        assert len(_parse_mcp_tools()) >= 10
        assert len(_parse_playground_tools()) >= 10
        assert len(_parse_api_routes()) >= 10
