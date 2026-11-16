"""A hand-written glossary that maps everyday wording to the vocabulary of the regulations.

People ask about "a data leak" or "the regulator"; the GDPR speaks of a "personal data breach" and a "supervisory
authority". A keyword search cannot bridge that gap, and even a good embedding model can miss it. Each rule below
recognises everyday wording in a question and appends the official terms to the **keyword** query (the dense query is
left alone). The appended words are terminology only - never answers such as deadlines or amounts - so the glossary
cannot leak the answer into the search.

Every term is checked against the real corpus by ``scripts/check_glossary.py``, and every rule has examples that its
pattern must match (and a counter-example that it must not) in ``tests/test_retrieval_text.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    terms: str  # official vocabulary appended to the keyword query
    examples: tuple[str, ...]  # phrases the pattern must match
    counter_examples: tuple[str, ...] = ()  # phrases it must not match


def rule(name: str, pattern: str, terms: str, examples: tuple[str, ...], counter: tuple[str, ...] = ()) -> Rule:
    return Rule(name, re.compile(pattern, re.IGNORECASE), terms, examples, counter)


GLOSSARY: tuple[Rule, ...] = (
    # ---- GDPR and EDPB / WP29 guidance
    rule("breach", r"\b(data|security)\s+(leak|breach|incident)s?\b|\bhack(ed|ing)?\b|\bleaked\b",
         "personal data breach notification supervisory authority communication data subjects",
         ("we had a data leak", "our database was hacked", "a security incident exposed emails"), ("breach of contract damages",)),
    rule("regulator", r"\b(regulator|dpa|data protection authority|privacy authority|watchdog)s?\b",
         "supervisory authority", ("who is the regulator", "report to the DPA", "the data protection authority")),
    rule("erasure", r"\bright to be forgotten\b|\b(delete|deleting|deletion|erase|erasing)\b.*\b(data|information|account|records?)\b|\b(data|information|account)\b.*\b(delete|deletion|erase)\b",
         "erasure right to erasure right to be forgotten", ("delete a customer's data", "right to be forgotten", "account deletion request"), ("delete the file",)),
    rule("access", r"\b(dsar|subject access|access request|access to (their|my|his|her) (own )?data)\b",
         "right of access data subject copy of the personal data", ("we received a DSAR", "a subject access request", "access request from a user")),
    rule("consent", r"\bconsent\b|\bopt[- ]in\b|\bcookie banner\b",
         "consent freely given specific informed unambiguous withdraw", ("do we need consent", "opt-in checkbox")),
    rule("dpia", r"\bdpias?\b|\bimpact assessment\b|\bprivacy assessment\b",
         "data protection impact assessment high risk processing", ("do we need a DPIA", "privacy assessment for the new feature")),
    rule("dpo", r"\bdpos?\b|\bdata protection officer\b|\bprivacy officer\b",
         "data protection officer designation tasks", ("must we appoint a DPO", "privacy officer duties")),
    rule("adm", r"\bautomated (decision|decisions|decision[- ]making)\b|\bprofiling\b|\balgorithmic decision",
         "automated individual decision-making profiling solely automated legal effects human intervention",
         ("automated decisions about customers", "profiling of users", "algorithmic decisions on loans")),
    rule("processor", r"\b(vendor|subprocessor|sub-processor|service provider|data processor|processor)s?\b",
         "processor controller processing contract sub-processor", ("our cloud vendor processes data", "engage a sub-processor")),
    rule("transfer", r"\b(transfer|transfers|send|sending|export)\b.*\b(outside|abroad|third countr|overseas|us servers|united states|india)|\binternational transfers?\b|\bcross[- ]border\b",
         "transfers personal data third countries adequacy decision standard contractual clauses safeguards",
         ("transfer data outside the EU", "sending customer data abroad", "international transfers to the US")),
    rule("legitimate_interest", r"\blegitimate interests?\b", "legitimate interests balancing test necessary", ("is legitimate interest enough",)),
    rule("anonymisation", r"\banonymi[sz](e|ed|ation|ing)\b|\bpseudonymi[sz]", "anonymisation anonymous data pseudonymisation identifiable", ("is anonymised data personal data",)),
    rule("children", r"\b(child|children|minor|minors|kids|teenager)s?\b", "children child parental consent age", ("users who are children", "services for minors")),
    rule("by_design", r"\bprivacy by (design|default)\b|\bdata protection by (design|default)\b", "data protection by design and by default technical organisational measures", ("privacy by design requirements",)),
    rule("retention", r"\bretention\b|\bhow long\b.*\b(keep|store|retain)\b|\bstorage limitation\b", "storage limitation retention period no longer necessary", ("data retention rules", "how long can we keep logs")),
    rule("fines", r"\b(fine|fines|penalt(y|ies)|sanction|sanctions)\b", "administrative fines penalties infringements", ("what are the fines", "penalties for non-compliance")),
    rule("web_scraping", r"\bscrap(e|ed|ing)\b|\bcrawl(ed|ing|er)?\b", "web scraping publicly available personal data training data legitimate interest", ("scraping websites for training data",)),
    # ---- EU AI Act and Commission guidance
    rule("chatbot", r"\bchat[- ]?bots?\b|\bvirtual assistants?\b|\bconversational (ai|agent|system)s?\b", "AI systems interacting directly with natural persons transparency obligations disclosed",
         ("our customer chatbot", "a virtual assistant for support")),
    rule("hiring", r"\b(resume|r[eé]sum[eé]|cv)s?\b|\brecruit(ing|ment|er)?\b|\bhiring\b|\bjob (candidates?|applicants?)\b|\bcandidate screening\b",
         "employment workers management recruitment selection high-risk Annex III", ("a resume screening feature", "tools for hiring", "ranking job applicants")),
    rule("high_risk", r"\bhigh[- ]risk\b", "high-risk AI systems classification Annex III requirements provider", ("is it a high-risk system",)),
    rule("deepfake", r"\bdeep[- ]?fakes?\b|\bsynthetic (media|content)\b|\bai[- ]generated (image|video|audio|content|text)s?\b", "deep fake AI-generated synthetic content marking disclosure labelling", ("labelling deepfakes", "AI-generated images")),
    rule("emotion", r"\bemotion (recognition|detection|analysis)\b|\bsentiment of (employees|workers|students)\b", "emotion recognition systems workplace education prohibited", ("emotion recognition in call centres",)),
    rule("biometric", r"\b(facial|face) recognition\b|\bbiometric\b|\bfingerprint\b", "biometric identification remote real-time facial images", ("facial recognition at the entrance", "biometric login")),
    rule("social_scoring", r"\bsocial (scoring|credit)\b|\bscore (people|citizens|customers)\b", "social scoring evaluation classification natural persons prohibited practices", ("a social scoring system",)),
    rule("gpai", r"\b(foundation|frontier|base) models?\b|\bllms?\b|\blarge language models?\b|\bgen(erative)? ?ai\b|\bgenai\b", "general-purpose AI models systemic risk generative AI obligations providers", ("we fine-tune a large language model", "GenAI features", "an LLM-based assistant")),
    rule("ai_act_dates", r"\b(delay\w*|postpon\w+|deadlines?|apply from|enters? into (force|application)|come into effect|when does|when do|start to apply)\b.*\b(ai act|ai regulation|omnibus)\b|\b(ai act|omnibus)\b.*\b(delay\w*|postpon\w+|deadlines?|date of application|apply from|entry into|obligations apply|apply)\b",
         "date of application entry into force amending regulation Digital Omnibus", ("when do the AI Act obligations apply", "was the AI Act delayed", "AI Act application deadline")),
    rule("omnibus", r"\bomnibus\b|\b2026/1744\b|\bsimplification package\b", "Digital Omnibus amending Regulation 2024/1689 2026/1744", ("what did the digital omnibus change",)),
    rule("literacy", r"\bai literacy\b|\btrain(ing)? (our |the )?(staff|employees)\b", "AI literacy staff sufficient level", ("AI literacy training for staff",)),
    # ---- United States
    rule("hipaa", r"\bhipaa\b|\bphi\b|\bprotected health information\b|\b(patient|medical|health) (data|records?|information)\b", "protected health information covered entity business associate", ("is this HIPAA covered", "storing patient records")),
    rule("baa", r"\bbusiness associates?\b|\bbaa\b", "business associate agreement covered entity subcontractor", ("do we need a BAA",)),
    rule("ccpa", r"\bccpa\b|\bcpra\b|\bcalifornia\b", "California Consumer Privacy Act consumer business personal information", ("do we have to follow the CCPA", "California customers")),
    rule("opt_out", r"\bopt[- ]?out\b|\bdo not sell\b|\bsell(ing)? (my|their|customer) (data|information)\b|\bsale of personal\b", "opt-out sale sharing personal information right to opt-out", ("opt out of the sale", "do not sell my information")),
    rule("ftc", r"\bftc\b|\bfederal trade commission\b|\bdeceptive\b", "Federal Trade Commission unfair deceptive acts practices", ("what does the FTC say", "deceptive AI claims")),
    # ---- frameworks and international instruments
    rule("nist_rmf", r"\bnist\b|\bai rmf\b|\brisk management framework\b", "AI Risk Management Framework govern map measure manage trustworthy", ("what is the NIST AI RMF",)),
    rule("adversarial", r"\bprompt injection\b|\bjailbreak\w*\b|\bpoison(ing|ed)?\b|\badversarial\b|\bevasion attacks?\b|\bmodel (theft|extraction|inversion)\b", "adversarial machine learning attacks poisoning evasion privacy prompt injection mitigations",
         ("defend against prompt injection", "training data poisoning", "adversarial examples")),
    rule("bias", r"\bbias(ed)?\b|\bfairness\b|\bdiscriminat\w*\b", "bias discrimination fairness systemic statistical computational human", ("how do we test for bias", "algorithmic discrimination")),
    rule("hallucination", r"\bhallucinat\w+\b|\bconfabulat\w+\b|\bmake[s]? things up\b", "confabulation hallucination generative AI risk information integrity", ("the model hallucinates", "confabulation risks")),
    rule("oecd", r"\boecd\b", "OECD Recommendation on Artificial Intelligence principles trustworthy AI", ("what are the OECD AI principles",)),
    rule("unesco", r"\bunesco\b|\bai ethics\b", "ethics of artificial intelligence UNESCO recommendation values principles", ("UNESCO ai ethics",)),
    rule("coe", r"\bcouncil of europe\b|\bframework convention\b", "Council of Europe Framework Convention artificial intelligence human rights democracy rule of law", ("the Council of Europe treaty",)),
    rule("uk", r"\b(uk|u\.k\.|britain|british|ico)\b", "Information Commissioner's Office UK GDPR", ("UK guidance on AI", "what does the ICO say")),
)


def expand(question: str) -> tuple[str, list[tuple[str, str]]]:
    """The question's keyword query: the question itself followed by the terms of every rule that fires.

    Returns the keyword query and the list of (rule name, terms) that were applied, for display in the interface.
    """
    applied = [(r.name, r.terms) for r in GLOSSARY if r.pattern.search(question)]
    if not applied:
        return question, []
    extra: list[str] = []
    for _, terms in applied:
        extra.extend(word for word in terms.split() if word not in extra)
    return f"{question} {' '.join(extra)}", applied
