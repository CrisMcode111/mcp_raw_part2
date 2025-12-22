"""
MCP MINI PROJECT - PART 2 (ALL-IN-ONE)

This single file includes:
1) A CUSTOM MCP SERVER with 2 non-trivial tools (domain-specific raw dessert tools)
2) Composition with EXTERNAL SERVERS from Part 1 via stdio (arxiv_server.py, notes_server.py)
3) LLM-based planning support (GroqCloud OR Ollama)
4) A demo client mode that calls multiple tools across multiple servers

Run:
  # Start server (orchestrator)
  python agent_client.py --server

  # Run demo (starts server + calls multiple tools)
  python agent_client.py --demo

Environment:
  # Ollama (default)
  set LLM_PROVIDER=ollama
  set OLLAMA_MODEL=llama3.1

  # Groq
  set LLM_PROVIDER=groq
  set GROQ_API_KEY=...
  set GROQ_MODEL=llama-3.1-8b-instant
"""

from __future__ import annotations
import os
import json
import asyncio
import argparse
from typing import Any, Dict, List, Tuple

# --- MCP imports (same family as stdio_client) ---
from mcp.server.fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession


# =========================
# 1) CUSTOM TOOLS (non-trivial, domain-specific)
# =========================

def raw_clean_citation(text: str) -> Dict[str, Any]:
    """
    Extract technical keywords + return structured insight for nut-free raw desserts.
    """
    vocab = ["hydration", "viscosity", "gel", "fiber", "emulsion", "stabilize", "thicken"]
    kws = [kw for kw in vocab if kw in (text or "").lower()]

    insight = (
        f"Insight from: {text}\n"
        "- Structure diagnosis: needs gel+fiber balance\n"
        "- Nut-free binders: chia/psyllium + fruit pectin sources\n"
        "- Viscosity strategy: micro-dose thickeners + resting time\n"
    )
    return {"cleaned": insight, "keywords": kws}


def raw_texture_advice(keywords: List[str]) -> Dict[str, str]:
    """
    Actionable texture fixes based on extracted keywords.
    """
    kws = set([k.lower() for k in (keywords or [])])
    lines = ["Raw dessert guidance:"]

    if "hydration" in kws:
        lines += [
            "- Too runny: add chia gel OR tiny psyllium dose; rest 10–15 min.",
            "- Too dense: add 1–2 tbsp fruit purée/water; blend again.",
        ]
    if "viscosity" in kws or "thicken" in kws:
        lines += ["- For thickness: coconut flour/oat fiber in tiny doses; rest after blending."]
    if "gel" in kws or "stabilize" in kws:
        lines += ["- For stable mousse: chill + add binder micro-dose; set 2–4h in fridge."]
    if "fiber" in kws:
        lines += ["- Fiber balance: add apple fiber/oat fiber gradually to bind water."]
    if "emulsion" in kws:
        lines += ["- Emulsion: add fat slowly (cocoa butter/coconut oil) while blending longer."]

    lines += ["- General: chill before judging final texture (setting needs time)."]
    return {"advice": "\n".join(lines)}


# =========================
# 2) LLM PLANNER (Groq or Ollama)
# =========================

async def llm_plan(task: str) -> Dict[str, Any]:
    """
    Return plan JSON:
      {"steps":[{"tool":"raw.clean_citation","args":{"text":"..."}}, ...]}
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    system = (
        "You are an MCP planner. Output ONLY valid JSON with a 'steps' array. "
        "Each step must be {tool: string, args: object}. No markdown."
    )

    tools_desc = """
Available tools:
- raw.clean_citation(text: str)
- raw.texture_advice(keywords: list[str])
- arxiv.search(query: str)            # from Part 1 server (if present)
- notes.save(title: str, content: str) # from Part 1 server (if present)
"""

    user = f"""{tools_desc}

Task:
{task}

Return JSON plan only.
"""

    if provider == "groq":
        # pip install groq
        from groq import AsyncGroq  # type: ignore
        api_key = os.environ["GROQ_API_KEY"]
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        client = AsyncGroq(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)

    # Default: Ollama
    # pip install httpx
    import httpx  # type: ignore
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("http://localhost:11434/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return json.loads(data["message"]["content"])


# =========================
# 3) COMPOSITION: connect to external servers (Part 1) via stdio
# =========================

class Router:
    def __init__(self) -> None:
        self.sessions: Dict[str, ClientSession] = {}
        self.contexts: List[Tuple[Any, ClientSession]] = []

    async def start(self) -> None:
        # These should be MCP servers from Part 1 (arxiv_server.py, notes_server.py).
        # If they are not MCP servers, calls will fail gracefully.
        for prefix, cmd in [
            ("arxiv", ["python", "arxiv_server.py"]),
            ("notes", ["python", "notes_server.py"]),
        ]:
            try:
                cm = stdio_client(cmd)
                read, write = await cm.__aenter__()
                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()
                self.sessions[prefix] = session
                self.contexts.append((cm, session))
            except Exception:
                # External server not available; that's ok for demo/robustness
                continue

    async def stop(self) -> None:
        for cm, session in reversed(self.contexts):
            await session.__aexit__(None, None, None)
            await cm.__aexit__(None, None, None)

    async def call_external(self, tool: str, args: Dict[str, Any]) -> Any:
        prefix = tool.split(".", 1)[0]
        if prefix not in self.sessions:
            return {"warning": f"External server '{prefix}' not available", "tool": tool, "args": args}
        return await self.sessions[prefix].call_tool(tool, args)


router = Router()


# =========================
# 4) CUSTOM MCP SERVER (Orchestrator)
# =========================

mcp = FastMCP("part2-orchestrator")


@mcp.tool(name="raw.clean_citation", description="Custom tool: extract raw dessert technical concepts from text.")
def tool_raw_clean_citation(text: str) -> Dict[str, Any]:
    return raw_clean_citation(text)


@mcp.tool(name="raw.texture_advice", description="Custom tool: give actionable texture advice based on keywords.")
def tool_raw_texture_advice(keywords: List[str]) -> Dict[str, str]:
    return raw_texture_advice(keywords)


@mcp.tool(
    name="orchestrator.solve",
    description="LLM-planned multi-step workflow calling tools across custom + external servers."
)
async def tool_orchestrator_solve(task: str) -> Dict[str, Any]:
    plan = await llm_plan(task)
    results: List[Dict[str, Any]] = []

    for i, step in enumerate(plan.get("steps", []), start=1):
        tool = step.get("tool")
        args = step.get("args", {}) or {}

        # route: raw.* are local tools; arxiv.* / notes.* are external tools
        if tool.startswith("raw."):
            if tool == "raw.clean_citation":
                out = raw_clean_citation(args.get("text", ""))
            elif tool == "raw.texture_advice":
                out = raw_texture_advice(args.get("keywords", []))
            else:
                out = {"error": f"Unknown local tool {tool}"}
        else:
            out = await router.call_external(tool, args)

        results.append({"step": i, "tool": tool, "args": args, "output": out})

    return {"plan": plan, "results": results}


async def run_server() -> None:
    await router.start()
    try:
        mcp.run()
    finally:
        await router.stop()


# =========================
# 5) DEMO CLIENT (calls multiple tools)
# =========================

async def run_demo() -> None:
    async with stdio_client(["python", "agent_client.py", "--server"]) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # prove multiple tools exist
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("\n=== Tools exposed by custom MCP server ===")
            print("\n".join(names))

            # call multiple tools
            print("\n=== Call custom tools ===")
            text = "Need hydration and gel structure; viscosity too low in nut-free raw cream."
            a = await session.call_tool("raw.clean_citation", {"text": text})
            print("raw.clean_citation ->", a)

            b = await session.call_tool("raw.texture_advice", {"keywords": a.get("keywords", [])})
            print("raw.texture_advice ->", b)

            print("\n=== Call orchestrator tool (planning + multi-step) ===")
            task = (
                "Use raw.clean_citation on this sentence, then raw.texture_advice on extracted keywords. "
                "If available, also call arxiv.search for 'hydration gel psyllium' and save a note."
            )
            c = await session.call_tool("orchestrator.solve", {"task": task})
            print("orchestrator.solve ->", c)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Run the custom MCP orchestrator server")
    parser.add_argument("--demo", action="store_true", help="Run demo client calling multiple tools")
    args = parser.parse_args()

    if args.server:
        asyncio.run(run_server())
        return
    if args.demo:
        asyncio.run(run_demo())
        return

    print("Usage: python agent_client.py --server | --demo")


if __name__ == "__main__":
    main()

