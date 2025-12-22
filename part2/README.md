# MCP Part 2 — Orchestrator (Composition + LLM Planning)

This project implements a **custom MCP orchestrator server** that:
- **composes external MCP servers** from Part 1: `arxiv_server.py`, `notes_server.py`, and `raw_server.py`
- adds **LLM-based planning** (GroqCloud or Ollama)
- exposes a **non-trivial tool**: `orchestrator.solve` (multi-step workflow)

## Files
- `orchestrator_server.py` — main MCP server (composition + planner)
- `arxiv_server.py` — external MCP server (Part 1)
- `notes_server.py` — external MCP server (Part 1)
- `raw_server.py` — custom MCP server with tools `raw.clean_citation`, `raw.texture_advice`
- `agent_client.py` — demo client that calls the orchestrator

## Run (Ollama)
1) Start Ollama locally
2) Set env:
   - `LLM_PROVIDER=ollama`
   - `OLLAMA_MODEL=llama3.1`
3) Run:
```bash
python agent_client.py

