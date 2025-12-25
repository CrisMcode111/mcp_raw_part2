# MCP Raw – Part 2  
### Orchestrator (Composition + LLM Planning)

This repository contains an **academic project developed as part of an AI training program**.

The goal of the assignment was to explore the Model Context Protocol (MCP) and demonstrate how an MCP client and a custom orchestrator server can interact with multiple MCP servers and tools.

This project is **educational in nature** and is not intended as a production-ready system.

---

## Overview

This project implements a **custom MCP orchestrator server** that:

- composes multiple external MCP servers developed in Part 1  
  (`arxiv_server.py`, `notes_server.py`, `raw_server.py`)
- integrates **LLM-based planning** (GroqCloud or Ollama)
- exposes a non-trivial orchestration tool:  
  `orchestrator.solve` (multi-step workflow execution)

The orchestrator coordinates tool calls across servers based on a high-level user goal.

---

## Files

- `orchestrator_server.py` — main MCP orchestrator server (composition + planning)
- `arxiv_server.py` — external MCP server (Part 1)
- `notes_server.py` — external MCP server (Part 1)
- `raw_server.py` — custom MCP server with tools  
  (`raw.clean_citation`, `raw.texture_advice`)
- `agent_client.py` — demo client invoking the orchestrator

---

## Running the demo (Ollama)

1. Start Ollama locally
2. Set environment variables:

```
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1
```
---

## Run the client

python agent_client.py

## Scope & Limitations

* The project focuses on orchestration and planning logic, not on production deployment.

* Error handling, security, and scalability concerns are intentionally minimal.

* The implementation strictly follows the scope of the academic assignment.
