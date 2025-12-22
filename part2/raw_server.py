# MOCK RAW SERVER — fully valid for MCP mini-project

def clean_citation(text: str):
    """Simulated MCP tool: extract scientific concepts"""
    keywords = [kw for kw in ["hydration", "viscosity", "gel", "fiber"]
                if kw in text.lower()]
    cleaned = (
        f"Insight from: {text}\n"
        "- Hydration score: 8/10\n"
        "- Viscosity: medium-high\n"
        "- Good for no-nut raw structure\n"
    )
    return {"cleaned": cleaned, "keywords": keywords}


def texture_advice(keywords):
    advice = "Raw dessert guidance:\n"
    if "hydration" in keywords:
        advice += "- Add chia/psyllium to stabilize gels.\n"
    if "viscosity" in keywords:
        advice += "- Use coconut flour / gums for thickness.\n"
    advice += "- Fruit purées add elasticity.\n"
    return {"advice": advice}


TOOLS = {
    "raw.clean_citation": clean_citation,
    "raw.texture_advice": texture_advice
}
