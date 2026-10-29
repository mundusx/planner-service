"""Explicit artifact capabilities. Unknown formats never become inference work."""
from typing import Any, TypedDict
import re

IMAGE_FORMATS = {"png", "jpeg", "webp", "bmp", "tiff", "gif", "avif"}


class ArtifactState(TypedDict, total=False):
    request: dict[str, Any]
    response: dict[str, Any]


def artifact_plan(request: dict[str, Any]) -> dict[str, Any]:
    from langgraph.graph import END, StateGraph

    def capability(state: ArtifactState) -> ArtifactState:
        item = state["request"]
        fmt, prompt = item.get("format"), item.get("prompt", "")
        response: dict[str, Any] = {"planner_provider": "langgraph", "format": fmt,
                                    "execution_owner": "artifact_api", "file_saved": False}
        if fmt not in IMAGE_FORMATS | {"sb3"}:
            response.update(action="unsupported", reason="No generator/validator for this file type")
        elif fmt == "sb3" and (len(prompt) > 600 or re.search(
                r"\b(game|platformer|score|collision|multiplayer|levels?|variables?|clones?|sound|music|keyboard|mouse|jump|fly|dance|rotate|spin|say|talk)\b", prompt, re.I)):
            response.update(action="unsupported", reason="Scratch currently supports one generated sprite with optional walking/bouncing motion; this request needs a coding generator")
        elif fmt != "sb3" and re.search(r"\b(animated|animation|video|moving gif)\b", prompt, re.I):
            response.update(action="unsupported", reason="This route generates still images; animation needs its own generator")
        else:
            motion = "walk" if re.search(r"\b(walk\w*|mov\w*|bounc\w*)\b", prompt, re.I) else "static"
            response.update(action="generate", input_kind="image", source_format="png",
                            operation="scratch_sprite" if fmt == "sb3" else "encode_image",
                            motion=motion if fmt == "sb3" else None,
                            validation="scratch_structure_and_assets" if fmt == "sb3" else "decode_output")
        return {"response": response}

    graph = StateGraph(ArtifactState)
    graph.add_node("check_artifact_capability", capability)
    graph.set_entry_point("check_artifact_capability")
    graph.add_edge("check_artifact_capability", END)
    return graph.compile().invoke({"request": request})["response"]
