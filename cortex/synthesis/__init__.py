from cortex.synthesis.result import SourceCitation, SynthesisResult

__all__ = ["KBSynthesizer", "SourceCitation", "SynthesisResult"]


def __getattr__(name: str):
    if name == "KBSynthesizer":
        from cortex.synthesis.synthesizer import KBSynthesizer

        return KBSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
