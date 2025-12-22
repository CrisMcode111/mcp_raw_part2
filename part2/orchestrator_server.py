"""
ORCHESTRATOR MCP SERVER (Part 2)

- Composes external MCP servers from Part 1 (arxiv_server.py, notes_server.py, raw_server.py)
- Adds LLM-based planning (GroqCloud or Ollama)
- Exposes a non-trivial tool that uses multiple tools (research -> notes -> raw advice)
"""

from __future__ import annotations
import os
import json
import asyncio
from typing import Any, Dict, List, Tuple

# ---- MCP imports (these match the stdio_client style you already used) ----
from mcp.server.fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

mcp = FastMCP("part2-orchestrator")


# ---------- LLM (Groq or Ollama) ----------

async def llm_plan(prompt: str) -> Dict[str, Any]:
    """
    Returns a plan in JSON form:
    {
      "steps": [
        {"tool": "arxiv.search", "args": {...}},
        {"tool": "notes.save", "args": {...}},
        {"tool": "raw.texture_advice", "args": {...}}
      ]
    }
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        # Minimal Groq example (requires: pip install groq)
        from groq import AsyncGroq  # type: ignore
        api_key = os.environ["GROQ_API_KEY"]
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        client = AsyncGroq(api_key=api_key)
        system = (
            "You are an MCP planner. Output ONLY valid JSON with a 'steps' array. "
            "Each step must be {tool: string, args: object}. No markdown."
        )
        user = f"""
Available tools:
- arxiv.search(query: str)
- notes.save(title: str, content: str)
- raw.clean_citation(text: str)
- raw.texture_advice(keywords: list[str])

Task:
{prompt}

Return JSON plan only.
"""

        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return json.loads(text)

    # Default: Ollama (requires local Ollama running: http://localhost:11434)
    # pip install httpx
    import httpx  # type: ignore

    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    system = (
        "You are an MCP planner. Output ONLY valid JSON with a 'steps' array. "
        "Each step must be {tool: string, args: object}. No markdown."
    )
    user = f"""
Available tools:
- arxiv.search(query: str)
- notes.save(title: str, content: str)
- raw.clean_citation(text: str)
- raw.texture_advice(keywords: list[str])

Task:
{prompt}

Return JSON plan only.
"""
    payload = {
        "model": ollama_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("http://localhost:11434/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return json.loads(data["message"]["content"])


# ---------- Compose external MCP servers (stdio) ----------

class ToolRouter:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.contexts: List[Tuple[Any, Any, Any]] = []  # keep async context managers alive

    async def start(self):
        # Start external servers as subprocesses using stdio transport.
        # IMPORTANT: these scripts must be MCP servers themselves.
        for name, cmd in [
            ("arxiv", ["python", "arxiv_server.py"]),
            ("notes", ["python", "notes_server.py"]),
            ("raw",   ["python", "raw_server.py"]),
        ]:
            cm = stdio_client(cmd)
            read, write = await cm.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            self.sessions[name] = session
            self.contexts.append((cm, session, (read, write)))

    async def stop(self):
        # Close sessions / contexts
        for cm, session, _ in reversed(self.contexts):
            await session.__aexit__(None, None, None)
            await cm.__aexit__(None, None, None)

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        # route "arxiv.xxx" -> arxiv session, "notes.xxx" -> notes session, "raw.xxx" -> raw session
        prefix = tool_name.split(".", 1)[0]
        if prefix not in self.sessions:
            raise ValueError(f"Unknown tool prefix: {prefix}")
        return await self.sessions[prefix].call_tool(tool_name, args)


router = ToolRouter()


# ---------- Non-trivial tool (uses multiple servers + LLM plan) ----------

@mcp.tool(
    name="orchestrator.solve",
    description="LLM-planned multi-step workflow: research -> note -> raw texture advice."
)
async def orchestrator_solve(task: str) -> Dict[str, Any]:
    """
    This tool demonstrates:
    - LLM planning
    - composition (calls tools across arxiv + notes + raw servers)
    - non-trivial execution chain
    """
    plan = await llm_plan(task)

    results: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {}

    for i, step in enumerate(plan.get("steps", []), start=1):
        tool = step["tool"]
        args = step.get("args", {})

        # Allow small variable injection from previous steps if needed
        # (keep it simple but shows "agentic" behavior)
        if isinstance(args, dict):
            for k, v in list(args.items()):
                if v == "$last":
                    args[k] = results[-1]["output"]

        out = await router.call(tool, args)
        results.append({"step": i, "tool": tool, "args": args, "output": out})

        # Keep a few useful values
        context["last_output"] = out

    return {"plan": plan, "results": results}


async def _main():
    await router.start()
    try:
        mcp.run()
    finally:
        await router.stop()


if __name__ == "__main__":
    asyncio.run(_main())

