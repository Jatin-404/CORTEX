__all__ = ["KBGraphAgent"]


def __getattr__(name: str):
    if name == "KBGraphAgent":
        from cortex.agent.runner import KBGraphAgent

        return KBGraphAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
