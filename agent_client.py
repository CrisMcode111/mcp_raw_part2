import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import servers.raw_server as raw_server
import servers.arxiv_server as arxiv_server
import servers.notes_server as notes_server


# Combine all tools in a global registry
ALL_TOOLS = {
    **raw_server.TOOLS,
    **arxiv_server.TOOLS,
    **notes_server.TOOLS,
}


def call_tool(name, args):
    """Call a mock tool by name."""
    tool = ALL_TOOLS[name]
    return tool(**args)


def run_agent(goal: str):
    print(f"\n🎯 GOAL: {goal}")

    # Step 1 – search scientific papers
    res1 = call_tool("arxiv.search", {"query": goal})
    print("\n🔍 Search result:", res1)

    first_title = res1["results"][0]["title"]

    # Step 2 – clean citation
    res2 = call_tool("raw.clean_citation", {"text": first_title})
    print("\n🧪 Cleaned:", res2)

    # Step 3 – texture advice
    res3 = call_tool("raw.texture_advice", {"keywords": res2["keywords"]})
    print("\n🥥 Advice:", res3)

    # Step 4 – write markdown
    full_content = res2["cleaned"] + "\n\n" + res3["advice"]

    res4 = call_tool("notes.write", {
        "path": "final_raw.md",
        "content": full_content
    })

    print("\n📄 Saved:", res4)
    print("\n✨ DONE – final_raw.md created!")


if __name__ == "__main__":
    run_agent("tiger nut hydration behavior")
