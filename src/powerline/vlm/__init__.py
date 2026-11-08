from .base import VLMBackend, parse_verdict


def build_vlm(cfg: dict, backend: str | None = None) -> VLMBackend:
    name = backend or cfg["vlm"]["backend"]
    if name == "claude":
        from .claude import ClaudeBackend
        return ClaudeBackend(cfg["vlm"]["claude"])
    if name == "qwen":
        from .qwen import QwenVLBackend
        return QwenVLBackend(cfg["vlm"]["qwen"])
    if name == "mock":
        from .mock import MockBackend
        return MockBackend()
    raise ValueError(f"unknown VLM backend: {name}")


__all__ = ["VLMBackend", "build_vlm", "parse_verdict"]
