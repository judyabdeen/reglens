# RegLens evaluation

Run on 2026-11-16. Index: 3813 passages from 34 documents (chroma). Embedding model: BAAI/bge-small-en-v1.5. Language model: gemini/gemini-3.1-flash-lite.
Gold set: 27 questions (23 answerable from the 34 indexed documents, 4 that the documents cannot answer).

## 1. Retrieval: is the evidence among the retrieved passages?

hit@k = the question has a relevant passage among the first k; evidence@8 = share of the expected evidence found in the first 8; from gold docs@8 = share of the first 8 passages that come from a document that holds the answer.

### Search method

| Configuration | hit@1 | hit@3 | hit@5 | hit@8 | MRR | evidence@8 | from gold docs@8 | s/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Keywords only (BM25) | 30% | 48% | 52% | 70% | 0.403 | 67% | 77% | 0.00 |
| Meaning only (embeddings) | 35% | 52% | 65% | 70% | 0.461 | 70% | 62% | 0.03 |
| Hybrid: meaning + keywords (RRF) | 30% | 57% | 57% | 78% | 0.455 | 78% | 73% | 0.04 |

### Re-ranking

| Configuration | hit@1 | hit@3 | hit@5 | hit@8 | MRR | evidence@8 | from gold docs@8 | s/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid: meaning + keywords (RRF) | 30% | 57% | 57% | 78% | 0.455 | 78% | 73% | 0.04 |
| Hybrid + cross-encoder re-ranking | 43% | 74% | 74% | 74% | 0.591 | 74% | 74% | 1.95 |

### Query transformation

| Configuration | hit@1 | hit@3 | hit@5 | hit@8 | MRR | evidence@8 | from gold docs@8 | s/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid: meaning + keywords (RRF) | 30% | 57% | 57% | 78% | 0.455 | 78% | 73% | 0.04 |
| Hybrid without the glossary | 35% | 52% | 57% | 70% | 0.470 | 67% | 67% | 0.04 |
| Hybrid + 2 paraphrases (multi-query) | 39% | 57% | 70% | 83% | 0.523 | 83% | 78% | 24.01 |
| Hybrid + hypothetical answer (HyDE) | 26% | 61% | 65% | 83% | 0.459 | 83% | 78% | 35.74 |

### Footnote passages

| Configuration | hit@1 | hit@3 | hit@5 | hit@8 | MRR | evidence@8 | from gold docs@8 | s/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid: meaning + keywords (RRF) | 30% | 57% | 57% | 78% | 0.455 | 78% | 73% | 0.04 |
| Hybrid without footnote passages | 30% | 57% | 57% | 78% | 0.454 | 78% | 75% | 0.07 |

### The default (hybrid) by type of question

| Question type | n | hit@1 | hit@5 | hit@8 | MRR |
|---|---:|---:|---:|---:|---:|
| definition | 3 | 33% | 33% | 33% | 0.404 |
| multi-document | 2 | 0% | 0% | 0% | 0.000 |
| number | 8 | 25% | 62% | 100% | 0.473 |
| rule | 6 | 33% | 67% | 100% | 0.521 |
| scope | 1 | 100% | 100% | 100% | 1.000 |
| temporal | 3 | 33% | 67% | 67% | 0.444 |

## 2. Answers: correctness, faithfulness, trade-offs

| Configuration | correct | abstained when it should | grounded | citation coverage | invented citations | unverified figures | s/answer | tokens in/out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Language model alone (no retrieval) | 18/23 (78%) | not asked | n/a | n/a | 0 answers | 0 answers | 25.1 | 39 / 219 |
| RegLens: hybrid retrieval + grounded prompt | 21/23 (91%) | 3/4 (75%) | 65% | 94% | 0 answers | 0 answers | 13.3 | 3461 / 240 |
| RegLens + re-ranking | 20/23 (87%) | 3/4 (75%) | 45% | 90% | 0 answers | 0 answers | 15.3 | 3476 / 243 |
| RegLens + 2 paraphrases | 21/23 (91%) | 3/4 (75%) | 61% | 92% | 0 answers | 0 answers | 20.9 | 3480 / 239 |

*grounded* = every sentence cited, no invented citation, no figure missing from the sources. This is a strict test: a heading, a lead-in or a closing disclaimer without a citation makes an answer fail it, so it understates grounding and is best used to compare configurations, not as a score on its own. *tokens* are the average prompt and answer tokens per question.

*Re-scored on 2026-11-16: the saved answers were scored again with corrected checks (no model was asked again; every configuration with the same checks). The corrections: two fact patterns that rejected correct answers were widened, and the figure check now reads each source as the model saw it, heading included (title, date, section). 5 correctness verdict(s) changed: rag q02 wrong to right, rag q12 wrong to right, rag_rerank q02 wrong to right, rag_multiquery q02 wrong to right, rag_multiquery q12 wrong to right; 20 answer(s) had their figure check changed.*

## 3. The questions where an out-of-date model should fail

**q04, Language model alone (no retrieval)** - does NOT state the expected facts.

> For high-risk AI systems listed in **Annex III**, the obligations under the EU AI Act generally start to apply **36 months** after the Act enters into force. The Regulation entered into force on **August 1, 2024**. Therefore, the obligations for Annex III systems will become applicable on **August 2, 2027**. **Key exceptions to note:** * **Systems already regulated:** If an Annex III system is also subject to specific EU harmonization legislation (listed in Annex I, such as medical devices or toys), the obligations apply from **August 2, 2027**. * **Governance and Penalties:** Certain provisions regarding governance, penalties, and the establishment of the AI Office became applicable earlie...

**q04, RegLens: hybrid retrieval + grounded prompt** - states the expected facts.

> The obligations for high-risk AI systems classified pursuant to Article 6(2) and Annex III of the EU AI Act apply as from 2 December 2027 [S1][S3]. * While the original AI Act (Regulation (EU) 2024/1689) initially set the date of application for these obligations as 2 August 2026, this date was postponed by the "AI Omnibus" (Regulation (EU) 2026/1744) [S1][S3]. * The postponement was enacted to address challenges regarding the delayed availability of standards, common specifications, and national competent authorities, aiming to avoid a significant increase in implementation costs [S3]. * Operators of high-risk AI systems placed on the market or put into service before 2 August 2026 are onl...

**q04, RegLens + re-ranking** - states the expected facts.

> The obligations for high-risk AI systems classified under Article 6(2) and Annex III of the EU AI Act apply from **2 December 2027** [S1][S2]. This date was established by the "AI Omnibus" (Regulation (EU) 2026/1744), which postponed the initial application date of 2 August 2026 due to challenges regarding the availability of standards and the establishment of national competent authorities [S1][S2]. Please note the following regarding existing systems: * **Significant Changes:** For high-risk systems placed on the market or put into service before 2 August 2026, the Act only applies if those systems undergo significant design changes on or after that date [S1][S7]. * **Public Authorities:*...

**q04, RegLens + 2 paraphrases** - states the expected facts.

> The obligations for AI systems classified as high-risk pursuant to Article 6(2) and Annex III of the EU AI Act apply as of 2 December 2027 [S1][S3]. Supporting points: * While the original version of the AI Act set the date of application for these obligations as 2 August 2026, the adoption of the Digital Omnibus on AI (Regulation (EU) 2026/1744) postponed this date to 2 December 2027 [S1][S3]. * This delay was introduced because the late availability of standards, common specifications, alternative guidance, and the delayed establishment of national competent authorities created implementation challenges [S3]. * Please note that this information relies on Regulation (EU) 2026/1744, which c...

**q05, Language model alone (no retrieval)** - does NOT state the expected facts.

> Under the EU AI Act (Article 111(2)), high-risk rules for AI systems that are safety components of products covered by Union harmonisation legislation listed in **Annex I, Section A** will apply from **August 2, 2027** (36 months after the entry into force of the Act). However, there is an important condition: these rules will only apply from this date **if** the relevant sector-specific legislation listed in Annex I is updated to incorporate the AI Act's high-risk requirements by August 2, 2027. *(Note: For AI systems classified as high-risk under Annex III, the application date is earlier, on August 2, 2026).*

**q05, RegLens: hybrid retrieval + grounded prompt** - states the expected facts.

> The AI Act's high-risk rules for AI systems classified under Article 6(1)—which includes AI systems that are safety components of products covered by Union harmonisation legislation listed in Annex I—apply from **2 August 2028** [S7]. Please note that this date reflects a postponement introduced by the AI Omnibus regulation; the original date for these requirements was 2 August 2027 [S7]. Guidance documents relied upon here are drafts and not adopted law [S5][S7].

**q05, RegLens + re-ranking** - does NOT state the expected facts.

> The provided sources do not specify the exact calendar date from which the AI Act's high-risk rules apply to AI systems that are safety components of products covered by Annex I, though they confirm the Act entered into force on 1 August 2024 [S7] and that an amendment entered into force on 27 July 2026 [S2][S5].

**q05, RegLens + 2 paraphrases** - states the expected facts.

> The AI Act’s high-risk rules for AI systems classified under Article 6(1)—which includes AI systems that are safety components of products covered by Union harmonisation legislation listed in Annex I—apply from **2 August 2028** [S5]. Please note that this date reflects a postponement enacted by the Digital Omnibus on AI [S5]. Source S5 is a draft document and not yet law, though it cites the entry into force of the Digital Omnibus (Regulation (EU) 2026/1744) [S3][S5].

**q06, Language model alone (no retrieval)** - does NOT state the expected facts.

> Under the final text of the EU AI Act, **Article 4** mandates that providers and deployers of AI systems ensure a sufficient level of **AI literacy** among their staff and other persons dealing with the operation and use of AI systems on their behalf. The specific requirements include: 1. **Scope of Knowledge:** The literacy measures must be tailored to the specific context of the AI system, considering the knowledge, experience, education, and training of the individuals involved. 2. **Responsibility:** Providers and deployers must consider the context in which the AI systems are to be used and the persons on whom the systems may have an impact. 3. **Content:** Literacy programs should ena...

**q06, RegLens: hybrid retrieval + grounded prompt** - states the expected facts.

> Following the entry into force of Regulation (EU) 2026/1744 (the "Digital Omnibus on AI") on 2026-07-27, Article 4 of the AI Act has been replaced [S1][S3]. The current requirements under Article 4 are as follows: * **Objective:** Providers and deployers must take measures to support the development of AI literacy for their staff and other persons dealing with the operation and use of AI systems on their behalf [S3]. * **Factors for consideration:** When taking these measures, providers and deployers must account for the individuals' technical knowledge, experience, education, and training, the context in which the AI systems are used, and the persons or groups of persons on whom the AI sys...

**q06, RegLens + re-ranking** - states the expected facts.

> Following the entry into force of the Digital Omnibus on AI (Regulation (EU) 2026/1744) on 2026-07-27, Article 4 of the AI Act has been amended to reflect the following requirements: * **Objective:** Providers and deployers must take measures to support the development of AI literacy for their staff and other persons dealing with the operation and use of AI systems on their behalf [S3]. * **Contextual approach:** Measures should account for the technical knowledge, experience, education, and training of these individuals, as well as the context in which the AI systems are used and the persons or groups on whom the systems are used [S3]. * **No guarantee required:** The obligation does not r...

**q06, RegLens + 2 paraphrases** - states the expected facts.

> As of July 27, 2026, Article 4 of Regulation (EU) 2024/1689 is replaced by the version set forth in the Digital Omnibus on AI (Regulation (EU) 2026/1744) [S4][S7]. The current requirements for providers and deployers are as follows: * **Support for AI Literacy:** Providers and deployers must take measures to support the development of AI literacy for their staff and other persons acting on their behalf who deal with the operation and use of AI systems [S4]. * **Contextual Factors:** These measures must account for the technical knowledge, experience, education, and training of the individuals involved, as well as the context in which the AI systems are used and the persons or groups on whom...

## 4. Where RegLens was wrong

- **q19** (rule): missed expected fact group(s) [1, 2, 3]. Answer: The provided sources do not list nine distinct criteria; rather, they state that to determine if a processing operation is "likely to result in a high risk," controllers should consider a set of criteria to provide a more concrete assessment [S2]. While the sources mention that "the following nine...
- **q22** (multi-document): missed expected fact group(s) [0]. Answer: The provided sources cover the California Consumer Privacy Act (CCPA) and its requirements regarding the right to delete personal information, but they do not contain information regarding the GDPR. Under the CCPA, consumers have the right to request that a business delete any personal information...
- **q26** (out-of-scope): answered a question the documents cannot answer. Answer: The provided sources cover general requirements for data protection by design, processor audits, and security measures for AI and data processing systems, but they do not address the specific setup or implementation of SOC 2 Type II audit controls. Regarding general security and audit obligations:...

## Limits of this evaluation

- The gold set has 27 questions written by the project author from the indexed documents. It is large enough to show
  clear differences between configurations and too small to estimate small ones; read a difference of one question as noise.
- *Correctness* is checked by matching the facts a correct answer must state (dates, amounts, named concepts) with
  regular expressions. It rewards an answer that states them and cannot see a wrong statement that sits beside them;
  the judge scores (if present) and the list of failures below are the check on that.
- *Faithfulness* here is what the pipeline can verify without a language model: every claim carries a citation, no cited
  label is invented, and no figure is missing from the sources. It does not prove that a cited passage really supports
  its sentence.
- The judge is the same family of model that wrote the answers, which tends to favour its own phrasing.
- Language-model answers vary from run to run; one run per configuration was made.

