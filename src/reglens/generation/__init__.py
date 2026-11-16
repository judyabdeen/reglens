"""Generation (assignment section 2.5): the grounded prompt, the answer object, citation and figure checks."""

from reglens.generation.answer import EXTRACTIVE, LLM_ONLY, RAG, Answer, Source
from reglens.generation.citations import citation_coverage, extract_labels, invalid_labels
from reglens.generation.grounding import unverified_figures
from reglens.generation.prompts import ABSTAIN_MARKER, SYSTEM_PROMPT, build_user_prompt

__all__ = [
    "ABSTAIN_MARKER",
    "Answer",
    "EXTRACTIVE",
    "LLM_ONLY",
    "RAG",
    "SYSTEM_PROMPT",
    "Source",
    "build_user_prompt",
    "citation_coverage",
    "extract_labels",
    "invalid_labels",
    "unverified_figures",
]
