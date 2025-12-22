"""
MCP Part 2 - Demo Client

This client demonstrates:
- starting the orchestrator MCP server
- discovering tools (proves composition)
- calling MULTIPLE tools across MULTIPLE servers:
  - custom server tools: raw.clean_citation, raw.texture_advice
  - external servers (Part 1): arxiv.*, notes.*
  - orchestrator tool: orchestrator.solve
"""

import asyncio
from typing import Any, Dict, List

from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession


async def safe_call(session: ClientSession, tool: str, args: Dict[str, Any]) -> Any:
    """Call tool but keep readable errors (helps evaluation)."""
    try:
        return await session.call_tool(tool, args)
    except Exception as e:
        return {"error": f"Failed calling {tool}: {e}", "args": args}


async def main() -> None:
    # Start the orchestrator server (the orchestrator composes arxiv + notes + raw servers).
    async with stdio_client(["python", "orchestrator_server.py"]) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1) Discover tools (shows that multiple servers/tools are exposed)
            tools = await session.list_tools()
            print("\n=== Available tools from orchestrator (composition proof) ===")
            # Print only names to keep output short
            tool_names: List[str] = [t.name for t in tools.tools]
            print("\n".join(tool_names))

            # 2) Call CUSTOM TOOLS (raw server)
            print("\n=== Call custom tools (raw.*) ===")
            text = "Need better hydration and gel structure in a nut-free raw cream; viscosity is too low."
            out1 = await safe_call(session, "raw.clean_citation", {"text": text})
            print("raw.clean_citation ->", out1)

            keywords = out1.get("keywords", []) if isinstance(out1, dict) else []
            out2 = await safe_call(session, "raw.texture_advice", {"keywords": keywords})
            print("raw.texture_advice ->", out2)

            # 3) Call EXTERNAL SERVER TOOLS (Part 1)
            # IMPORTANT: replace tool names below with the real ones from your arxiv_server.py / notes_server.py
            print("\n=== Call external server tools (arxiv.* and notes.*) ===")

            # Example: arxiv search / fetch (rename if needed)
            arxiv_res = await safe_call(session, "arxiv.search", {"query": "hydration gel raw dessert fiber psyllium"})
            print("arxiv.search ->", arxiv_res)

            # Example: save note (rename if needed)
            note_res = await safe_call(session, "notes.save", {"title": "Raw dessert texture", "content": str(arxiv_res)[:500]})
            print("notes.save ->", note_res)

            # 4) Call ORCHESTRATOR TOOL (LLM-based planning + multi-step workflow)
            print("\n=== Call orchestrator.solve (planning + multi-step) ===")
            task = (
                "Research 1–2 concepts about hydration/gel structure for nut-free raw desserts, "
                "save a short note, then provide texture advice based on keywords."
            )
            orch = await safe_call(session, "orchestrator.solve", {"task": task})
            print("orchestrator.solve ->", orch)


if __name__ == "__main__":
    asyncio.run(main())

# Explicit proof of multiple tools / servers
assert any(t.startswith("raw.") for t in tool_names)
assert any(t.startswith("arxiv.") for t in tool_names)
assert any(t.startswith("notes.") for t in tool_names)
assert "orchestrator.solve" in tool_names

