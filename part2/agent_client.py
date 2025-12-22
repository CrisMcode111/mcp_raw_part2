"""
Demo client for the custom MCP server.

Run server first:
  python raw_server.py

Then run:
  python agent_client.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

async def main() -> None:
    # This import path can vary depending on your bootcamp template.
    # If it fails, tell me what error you get and I'll adapt it.
    from mcp.client.stdio import stdio_client  # type: ignore
    from mcp.client.session import ClientSession  # type: ignore

    # Launch server via stdio
    # If your bootcamp uses a different transport, we adapt.
    async with stdio_client(["python", "raw_server.py"]) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            text = "We need better hydration and gel structure without nuts. Viscosity is too low."
            res1: Dict[str, Any] = await session.call_tool(
                "raw.clean_citation",
                {"text": text},
            )
            print("=== raw.clean_citation ===")
            print(res1)

            keywords = res1.get("keywords", [])
            res2: Dict[str, Any] = await session.call_tool(
                "raw.texture_advice",
                {"keywords": keywords},
            )
            print("\n=== raw.texture_advice ===")
            print(res2)

if __name__ == "__main__":
    asyncio.run(main())
