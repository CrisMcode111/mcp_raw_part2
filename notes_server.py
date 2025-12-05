def write(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "saved", "path": path}

TOOLS = {"notes.write": write}
