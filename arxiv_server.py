def search(query: str):
    return {
        "results": [
            {"title": "Hydration behavior of buckwheat gels"},
            {"title": "Pectin-cocoa interactions in raw desserts"},
            {"title": "Water activity control in vegan confectionery"}
        ]
    }

TOOLS = {"arxiv.search": search}
