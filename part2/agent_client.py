import asyncio
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

async def main():
    async with stdio_client(["python", "orchestrator_server.py"]) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            task = (
                "Find 1–2 scientific concepts about hydration/gel structure for nut-free raw desserts, "
                "save a short note, then give texture advice keywords-based."
            )
            res = await session.call_tool("orchestrator.solve", {"task": task})
            print(res)

if __name__ == "__main__":
    asyncio.run(main())
