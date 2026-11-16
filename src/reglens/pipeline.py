"""The RAG pipeline end to end (assignment Step 6): question in, grounded and cited answer out.

    question -> [condense follow-up] -> [glossary / paraphrases / HyDE] -> hybrid retrieval -> [re-rank]
             -> context assembly (word budget, numbered sources) -> prompt -> language model -> citation and figure checks

Every stage is timed, and every failure of the language model degrades to showing the retrieved passages instead of
an error page.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace

from reglens.generation.answer import EXTRACTIVE, LLM_ONLY, RAG, Answer, Source
from reglens.generation.citations import citation_coverage, extract_labels, invalid_labels
from reglens.generation.grounding import unverified_figures
from reglens.generation.prompts import ABSTAIN_MARKER, LLM_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt, source_header
from reglens.llm.base import LLMClient, LLMError, LLMRequest
from reglens.retrieval.context import DEFAULT_MAX_WORDS, assemble_context
from reglens.retrieval.query_transform import QueryPlan, QueryTransformer, TransformOptions, Turn
from reglens.retrieval.rerank import RerankError
from reglens.retrieval.retriever import RetrievalOptions, Retriever

MAX_QUESTION_CHARS = 2000
NO_PASSAGES = "No passage in the indexed documents matches this question."
ABSTAIN_MAX_WORDS = 60  # a refusal is one sentence; a longer text with citations after the marker is an answer
UNAVAILABLE = "unavailable"  # Answer.error_kind when no language model is configured at all


@dataclass(frozen=True)
class AskOptions:
    retrieval: RetrievalOptions = field(default_factory=RetrievalOptions)
    transform: TransformOptions = field(default_factory=TransformOptions)
    max_context_words: int = DEFAULT_MAX_WORDS
    max_output_tokens: int = 2048
    use_llm: bool = True  # False: retrieval only


def _clean_question(question: str) -> str:
    question = " ".join(question.split())
    if not question:
        raise ValueError("the question is empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"the question is too long ({len(question)} characters; the limit is {MAX_QUESTION_CHARS})")
    return question


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None, *, llm_unavailable: str | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.llm_unavailable = llm_unavailable if llm is None else None
        self._transformer = QueryTransformer(llm)

    # ------------------------------------------------------------------------------------------ RAG
    def ask(self, question: str, history: Sequence[Turn] = (), options: AskOptions | None = None) -> Answer:
        if isinstance(history, AskOptions):
            raise TypeError("the second argument of ask() is the conversation history; pass the settings as options=AskOptions(...)")
        options = options or AskOptions()
        started = time.perf_counter()
        question = _clean_question(question)
        timings: dict[str, float] = {}

        mark = time.perf_counter()
        plan = self._transformer.plan(question, history, options.transform)
        timings["transform"] = time.perf_counter() - mark

        mark = time.perf_counter()
        notes: list[str] = []
        try:
            results = self.retriever.retrieve(plan, options.retrieval)
        except RerankError as exc:  # the cross-encoder could not be loaded or failed: keep the first-stage order
            notes.append(f"The re-ranking model could not be used ({exc}); the passages are in the order of the first search.")
            results = self.retriever.retrieve(plan, replace(options.retrieval, rerank=False))
        context = assemble_context(results, max_words=options.max_context_words, max_items=max(options.retrieval.top_k, 1))
        timings["retrieve"] = time.perf_counter() - mark

        sources = tuple(Source.from_item(item) for item in context.items)
        base = dict(
            question=question,
            standalone_question=plan.standalone,
            plan_notes=plan.notes,
            options=_options_dict(options),
        )
        if context.dropped:
            notes.append(f"{len(context.dropped)} retrieved passage(s) did not fit the context budget and were left out")

        if not sources:
            timings["total"] = time.perf_counter() - started
            return Answer(**base, mode=RAG, text=NO_PASSAGES, abstained=True, timings=timings, notes=tuple(notes))

        if not options.use_llm or self.llm is None:
            reason = self.llm_unavailable if options.use_llm else None
            text = "The language model was not used; the most relevant passages are shown below, best first."
            if reason:
                notes.append(reason)
            timings["total"] = time.perf_counter() - started
            return Answer(
                **base, mode=EXTRACTIVE, text=text, sources=sources, error=reason,
                error_kind=UNAVAILABLE if reason else None, timings=timings, notes=tuple(notes),
            )

        mark = time.perf_counter()
        request = LLMRequest.single(SYSTEM_PROMPT, build_user_prompt(context.items, plan.standalone), max_output_tokens=options.max_output_tokens)
        try:
            response = self.llm.generate(request)
        except LLMError as exc:
            timings["generate"] = time.perf_counter() - mark
            timings["total"] = time.perf_counter() - started
            notes.append(f"The language model could not be used ({exc.kind}); the retrieved passages are shown instead.")
            text = "The language model could not answer, so the most relevant passages are shown below, best first."
            return Answer(
                **base, mode=EXTRACTIVE, text=text, sources=sources, error=str(exc), error_kind=exc.kind,
                timings=timings, notes=tuple(notes),
            )
        timings["generate"] = time.perf_counter() - mark

        answer_text = response.text.strip()
        abstained = answer_text.upper().startswith(ABSTAIN_MARKER)
        if abstained:
            rest = answer_text[len(ABSTAIN_MARKER) :].strip()
            if len(rest.split()) > ABSTAIN_MAX_WORDS and extract_labels(rest):
                # the model began with the refusal marker and then answered from the sources anyway: show the answer, not a refusal
                abstained = False
                answer_text = rest
                notes.append("The model began by saying the documents do not answer the question, then answered from them. Read it with care.")
            else:
                answer_text = rest or "The indexed documents do not answer this question."
        labels = extract_labels(answer_text)
        valid = [s.label for s in sources]
        invalid = invalid_labels(labels, valid)
        coverage, uncited = citation_coverage(answer_text)
        figures = unverified_figures(
            answer_text, (f"{source_header(i.retrieved.metadata, i.retrieved.id)}\n{i.retrieved.text}" for i in context.items), plan.standalone
        )
        if invalid:
            notes.append(f"The answer cites {', '.join(invalid)}, which do not exist among the sources: treat those citations as unreliable.")
        if response.truncated:
            notes.append("The answer was cut off by the output limit.")
        if response.fell_back_from:
            notes.append(f"{response.fell_back_from} was unavailable, so {response.model} answered.")
        cited_sources = tuple(
            Source.from_item(item, cited=item.label in labels) for item in context.items
        )
        timings["total"] = time.perf_counter() - started
        return Answer(
            **base,
            mode=RAG,
            text=answer_text,
            abstained=abstained,
            sources=cited_sources,
            cited=tuple(label for label in labels if label in valid),
            invalid_citations=tuple(invalid),
            unverified_figures=tuple(figures),
            citation_coverage=coverage,
            uncited_sentences=tuple(uncited),
            provider=response.provider,
            model=response.model,
            fell_back_from=response.fell_back_from,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.output_tokens,
            truncated=response.truncated,
            timings=timings,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------------------------------ baseline
    def ask_llm_only(self, question: str, max_output_tokens: int = 2048) -> Answer:
        """The same model without retrieval: the "LLM alone" side of the comparison in assignment section 1.1."""
        started = time.perf_counter()
        question = _clean_question(question)
        base = dict(question=question, standalone_question=question, mode=LLM_ONLY)
        if self.llm is None:
            reason = self.llm_unavailable or "no language model is configured"
            return Answer(**base, text="The language model is not available.", error=reason, error_kind=UNAVAILABLE, notes=(reason,))
        try:
            response = self.llm.generate(LLMRequest.single(LLM_ONLY_SYSTEM_PROMPT, question, max_output_tokens=max_output_tokens))
        except LLMError as exc:
            return Answer(
                **base, text="The language model could not answer.", error=str(exc), error_kind=exc.kind,
                notes=(f"The language model could not be used ({exc.kind}).",), timings={"total": time.perf_counter() - started},
            )
        return Answer(
            **base,
            text=response.text.strip(),
            provider=response.provider,
            model=response.model,
            fell_back_from=response.fell_back_from,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.output_tokens,
            truncated=response.truncated,
            timings={"generate": response.latency_s, "total": time.perf_counter() - started},
        )


def _options_dict(options: AskOptions) -> dict:
    data = asdict(options)
    data["retrieval"]["where"] = dict(options.retrieval.where) if options.retrieval.where else None
    return data


__all__ = ["AskOptions", "MAX_QUESTION_CHARS", "NO_PASSAGES", "QueryPlan", "RAGPipeline", "UNAVAILABLE"]
