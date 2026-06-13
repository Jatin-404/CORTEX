from enum import StrEnum


class SourceType(StrEnum):
    """Discriminator for the unified Qdrant collection."""

    HANDBOOK_MARKDOWN = "handbook_markdown"
    SEMANTIC_LAYER = "semantic_layer"
    ENTITY_NOTES = "entity_notes"
    PDF = "pdf"
    DOCX = "docx"


class ConfidentialityLevel(StrEnum):
    GLOBAL = "global"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


class ChunkRole(StrEnum):
    CHILD = "child"    # embedded + retrieved
    PARENT = "parent"  # context payload only (not indexed separately)
