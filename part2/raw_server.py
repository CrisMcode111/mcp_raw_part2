"""
Custom MCP Server - Raw Dessert Tools (Nut-free / No-bake)

This file implements a REAL MCP server exposing 2 custom tools:
1) raw.clean_citation   -> extract concepts + generate structured "insight"
2) raw.texture_advice   -> return actionable texture advice based on concepts

Run:
  python raw_server.py
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---- Custom tool logic (your original functions, improved) ----

def clean_citation(text: str) -> Dict[str, Any]:
    """
    Custom MCP tool:
    - Extract technical keywords from an input text
    - Return a structured insight block + extracted keywords
    """
    vocab = ["hydration", "viscosity", "gel", "fiber", "emulsion", "stabilize", "thicken"]
    keywords: List[str] = [kw for kw in vocab if kw in (text or "").lower()]

    cleaned = (
        f"Insight from: {text}\n"
        "- Hydration score: 8/10 (simulated)\n"
        "- Viscosity: medium-high (simulated)\n"
        "- Suggested structure: fiber + gel synergy\n"
        "- Nut-free raw note: prefer chia/psyllium + fruit pectin sources\n"
    )

    return {"cleaned": cleaned, "keywords": keywords}


def texture_advice(keywords: List[str]) -> Dict[str, str]:
    """
    Custom MCP tool:
    - Given extracted keywords, return practical raw dessert texture fixes
    """
    kws = set([k.lower() for k in (keywords or [])])

    advice = ["Raw dessert guidance:"]

    if "hydration" in kws:
        advice.append("- If too runny: add chia gel or a small psyllium dose; rest 10–15 min.")
        advice.append("- If too dense: add 1–2 tbsp fruit purée or a splash of water, blend again.")
    if "viscosity" in kws or "thicken" in kws:
        advice.append("- For thickness: coconut flour (tiny doses) or oat fiber; blend + rest.")
    if "gel" in kws or "stabilize" in kws:
        advice.append("- For stable mousse/cream: chilled coconut cream + psyllium/chia micro-doses.")
    if "fiber" in kws:
        advice.append("- Fiber balances wetness: add carrot pulp, apple fiber, or oat fiber gradually.")
    if "emulsion" in kws:
        advice.append("- For emulsion: blend longer + add fat slowly (cocoa butter/coconut oil).")

    # Always add a general rule
    advice.append("- General: chill 2–4 hours to set texture before judging final firmness.")

    return {"advice": "\n".join(advice)}


# ---- MCP Server wrapper (real server, not a mock dict) ----
# The exact import path can differ by bootcamp template.
# Try Option A first. If it errors, use Option B below.

def main() -> None:
    # Option A (common in MCP mini-project templates)
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
        mcp = FastMCP("raw-dessert-mcp")

        @mcp.tool(name="raw.clean_citation", description="Extract raw dessert technical concepts from text.")
        def tool_clean_citation(text: str) -> Dict[str, Any]:
            return clean_citation(text)

        @mcp.tool(name="raw.texture_advice", description="Give raw dessert texture advice based on keywords.")
        def tool_texture_advice(keywords: List[str]) -> Dict[str, str]:
            return texture_advice(keywords)

        mcp.run()
        return

    except Exception:
        pass

    # Option B (fallback if the template uses a different server API)
    try:
        from mcp.server import Server  # type: ignore
        from mcp.types import Tool  # type: ignore

        server = Server("raw-dessert-mcp")

        @server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="raw.clean_citation",
                    description="Extract raw dessert technical concepts from text.",
                    inputSchema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                ),
                Tool(
                    name="raw.texture_advice",
                    description="Give raw dessert texture advice based on keywords.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["keywords"],
                    },
                ),
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
            if name == "raw.clean_citation":
                return clean_citation(arguments.get("text", ""))
            if name == "raw.texture_advice":
                return texture_advice(arguments.get("keywords", []))
            raise ValueError(f"Unknown tool: {name}")

        server.run()
        return

    except Exception as e:
        raise RuntimeError(
            "Could not start MCP server. "
            "Your environment/template likely uses a different MCP import path.\n"
            "Fix: open agent_client.py and raw_server.py imports to match your bootcamp skeleton.\n"
            f"Error: {e}"
        )


if __name__ == "__main__":
    main()
