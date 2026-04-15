"""
RoboQ quiz-generation pipeline.

Phase 1: Logic & Reasoning Upgrade.
A semantic router classifies each document as LOGIC_BASED or MEMORIZATION_BASED
and dispatches chunks to one of two sub-chains:

  Branch A (memorization): legacy refine chain on gpt-4o-mini.
  Branch B (logic):        LCEL pipeline on gpt-4o — procedural abstraction
                           followed by CoT + few-shot generation with
                           Pydantic-enforced structured output.

The public entry point `generate_questions(file_path, pagenum)` keeps the
exact same signature and return shape as before so app.py is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from typing import Literal

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MEMORIZATION_MODEL = "gpt-4o-mini"
LOGIC_MODEL = "gpt-4o"
ROUTER_MODEL = "gpt-4o-mini"

ROUTER_SNIPPET_CHARS = 300


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    category: Literal["LOGIC_BASED", "MEMORIZATION_BASED"] = Field(
        description=(
            "LOGIC_BASED if the text requires calculation, procedural rules, "
            "or formulas (math, physics, CS algorithms). MEMORIZATION_BASED "
            "if it is historical facts, definitions, or descriptive content."
        )
    )


class LogicQuestion(BaseModel):
    reasoning: str = Field(
        description=(
            "Step-by-step logical reasoning about how a student might "
            "misunderstand the extracted concept, and the calculation steps "
            "for the correct answer."
        )
    )
    text: str = Field(description="The actual multiple choice question text.")
    options: list[str] = Field(description="Exactly four multiple choice options.")
    answer: str = Field(
        description="The correct option, matching exactly one item in the options list."
    )


class QuizOutput(BaseModel):
    questions: list[LogicQuestion]


# ---------------------------------------------------------------------------
# Helpers shared by both branches
# ---------------------------------------------------------------------------

def clean_label(text: str) -> str:
    """Strip leading 'A.', 'B)', etc. from an option or answer string."""
    return re.sub(r"^[A-E][.)]\s*", "", text.strip())


def _normalize_question(q: dict) -> dict:
    """Clean labels and shuffle options so the UI receives a stable shape."""
    if "options" in q and isinstance(q["options"], list):
        q["options"] = [clean_label(o) for o in q["options"]]
        random.shuffle(q["options"])
    if "answer" in q:
        q["answer"] = clean_label(q["answer"])
    # reasoning is logged and stripped so the frontend contract is unchanged
    if "reasoning" in q:
        print(f"[logic CoT] {q.pop('reasoning')}")
    return q


# ---------------------------------------------------------------------------
# Semantic router
# ---------------------------------------------------------------------------

def build_router():
    llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a classifier that decides how a study passage should "
                "be turned into quiz questions. Read the excerpt and return "
                "LOGIC_BASED when answering questions about it requires "
                "calculation, applying formulas, following a procedure, or "
                "reasoning through a proof (math, physics, chemistry "
                "problem-solving, CS algorithms). Return MEMORIZATION_BASED "
                "when correct answers depend mainly on recalling facts, "
                "dates, names, vocabulary, or descriptive content (history, "
                "biology terminology, literature).",
            ),
            ("human", "Excerpt:\n---\n{snippet}\n---"),
        ]
    )
    return prompt | llm.with_structured_output(RouteDecision)


# ---------------------------------------------------------------------------
# Branch A: memorization (legacy refine chain)
# ---------------------------------------------------------------------------

MEMORIZATION_PROMPT_TEMPLATE = """
You are an expert at creating AP-style multiple choice questions based on
class materials and teacher handouts. Your goal is to prepare a student for
their exam. Ask multiple choice questions about the text below:

------------
{text}
------------

Create multiple choice questions that will prepare the student for their
tests. Do not refer to the given text in the question, because the student
will not have access to it. Provide the questions as a JSON list where each
entry has a 'text' field (question text), an 'options' field (array of four
choices), and an 'answer' field (the correct choice). Do not label the
options or answers with letters.

QUESTIONS:
"""


def build_memorization_chain():
    llm = ChatOpenAI(temperature=0.3, model=MEMORIZATION_MODEL)
    prompt = PromptTemplate(
        template=MEMORIZATION_PROMPT_TEMPLATE, input_variables=["text"]
    )
    return load_summarize_chain(
        llm=llm, chain_type="refine", question_prompt=prompt
    )


def _parse_memorization_output(output_text: str) -> list[dict]:
    """Robustly parse the refine chain's free-form JSON list output."""
    cleaned = output_text.replace("```json", "").replace("```", "").strip()
    cleaned = cleaned.replace("(\\)", "(\\\\)")
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]

    individual = re.split(r"},\s+{", cleaned)
    individual = [q if q.strip().startswith("{") else "{" + q for q in individual]
    individual = [q if q.strip().endswith("}") else q + "}" for q in individual]

    results: list[dict] = []
    for raw in individual:
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError as e:
            print(f"[memorization] JSON parse error: {e} on {raw[:120]}...")
    return results


async def run_memorization_branch(chunk: str, chain) -> list[dict]:
    doc = [Document(page_content=chunk)]
    result = await chain.ainvoke(doc)
    parsed = _parse_memorization_output(result["output_text"])
    return [_normalize_question(q) for q in parsed]


# ---------------------------------------------------------------------------
# Branch B: logic (LCEL, procedural abstraction + CoT/ICL + structured output)
# ---------------------------------------------------------------------------

ABSTRACTION_SYSTEM = (
    "You are a domain expert in mathematics, physics, and quantitative "
    "reasoning. Given a study passage, extract the underlying formal "
    "content a student must master: named theorems, definitions, formulas "
    "with variable meanings, procedural rules, and any edge cases or "
    "common pitfalls. Output a concise bullet list — no prose, no quiz "
    "questions. If the passage contains no formal content, output the "
    "single line: NONE."
)

GENERATION_SYSTEM = """You are an expert author of rigorous multiple choice \
questions for logic- and calculation-heavy subjects (math, physics, \
algorithms). You will be given (a) a list of formal rules already extracted \
from a passage, and (b) the original passage. Produce multiple choice \
questions that test whether a student can APPLY those rules, not just \
recognize them. Questions must stand alone — do not reference "the text" or \
"the passage". Each question must have exactly four options and exactly one \
correct answer that appears verbatim in the options list. Distractors must \
encode realistic student mistakes (off-by-one, sign error, wrong formula, \
confused variable). For every question, first think through the solution \
step by step in the `reasoning` field before writing `text`, `options`, and \
`answer`.

Here are three examples of the quality bar you must meet.

=== EXAMPLE 1 ===
reasoning: "A common mistake is forgetting the chain rule. d/dx[sin(3x)] = \
cos(3x) * 3 = 3cos(3x). A student who forgets the chain rule writes \
cos(3x). A student who also mishandles the coefficient writes cos(x) or \
3sin(3x). The correct answer is 3cos(3x)."
text: "What is the derivative of f(x) = sin(3x) with respect to x?"
options: ["3cos(3x)", "cos(3x)", "cos(x)", "3sin(3x)"]
answer: "3cos(3x)"

=== EXAMPLE 2 ===
reasoning: "Kinetic energy KE = 1/2 m v^2. With m = 4 kg and v = 5 m/s, \
KE = 0.5 * 4 * 25 = 50 J. Students who forget the 1/2 factor get 100 J. \
Students who forget to square v get 10 J. Students who use v^2 but forget \
the 1/2 and swap units get 20 J."
text: "A 4 kg object moves at 5 m/s. What is its kinetic energy?"
options: ["50 J", "100 J", "10 J", "20 J"]
answer: "50 J"

=== EXAMPLE 3 ===
reasoning: "Binary search on a sorted array of n elements runs in \
O(log n). Students confuse it with linear search O(n), with sorting \
O(n log n), or with nested loops O(n^2). The correct worst-case time \
complexity is O(log n)."
text: "What is the worst-case time complexity of binary search on a \
sorted array of n elements?"
options: ["O(log n)", "O(n)", "O(n log n)", "O(n^2)"]
answer: "O(log n)"
=== END EXAMPLES ===

Generate between 2 and 4 questions per call, depending on how much \
quiz-worthy material the passage contains."""


def build_logic_chain():
    logic_llm = ChatOpenAI(temperature=0.2, model=LOGIC_MODEL)

    abstraction_prompt = ChatPromptTemplate.from_messages(
        [("system", ABSTRACTION_SYSTEM), ("human", "{text}")]
    )
    abstraction_subchain = abstraction_prompt | logic_llm | StrOutputParser()

    generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATION_SYSTEM),
            (
                "human",
                "EXTRACTED RULES:\n{rules}\n\nORIGINAL PASSAGE:\n{text}",
            ),
        ]
    )
    generation_subchain = generation_prompt | logic_llm.with_structured_output(
        QuizOutput
    )

    # Full LCEL pipeline: attach rules, then generate structured quiz.
    return (
        RunnablePassthrough.assign(rules=abstraction_subchain)
        | generation_subchain
    )


def _is_token_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "context_length_exceeded" in msg
        or "maximum context length" in msg
        or "token" in msg and "limit" in msg
    )


async def run_logic_branch(
    chunk: str, logic_chain, memorization_chain
) -> list[dict]:
    try:
        result: QuizOutput = await logic_chain.ainvoke({"text": chunk})
    except Exception as exc:
        # Token-limit failures (and any other logic-branch failure) fall back
        # to the memorization branch so we still return *something* for this
        # chunk rather than dropping it.
        if _is_token_limit_error(exc):
            print(f"[logic] token limit hit — falling back to memorization: {exc}")
        else:
            print(f"[logic] failure — falling back to memorization: {exc}")
        return await run_memorization_branch(chunk, memorization_chain)

    return [_normalize_question(q.model_dump()) for q in result.questions]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def _generate_for_document_async(
    chunks: list[str], category: str
) -> list[dict]:
    memorization_chain = build_memorization_chain()

    if category == "LOGIC_BASED":
        logic_chain = build_logic_chain()
        tasks = [
            run_logic_branch(c, logic_chain, memorization_chain) for c in chunks
        ]
    else:
        tasks = [run_memorization_branch(c, memorization_chain) for c in chunks]

    # gpt-4o + CoT is slow per-call; fan out chunks concurrently.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    questions: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[pipeline] chunk failed entirely, skipping: {r}")
            continue
        questions.extend(r)
    return questions


def _collect_chunks(data, pagenum) -> tuple[list[str], str]:
    """Page-filter the loaded PDF and split into token-sized chunks.

    Returns (chunks, full_snippet_for_router).
    """
    splitter = TokenTextSplitter(
        model_name=MEMORIZATION_MODEL, chunk_size=300, chunk_overlap=50
    )

    chunks: list[str] = []
    accum = ""
    full_text_for_router = ""

    for page in data:
        if -1 not in pagenum and page.metadata["page"] not in pagenum:
            continue

        if len(full_text_for_router) < ROUTER_SNIPPET_CHARS:
            full_text_for_router += page.page_content

        accum += page.page_content
        if len(accum) > 1000:
            chunks.extend(splitter.split_text(accum))
            accum = ""

    if len(accum) > 100:
        chunks.extend(splitter.split_text(accum))

    return chunks, full_text_for_router[:ROUTER_SNIPPET_CHARS]


def generate_questions(file_path: str, pagenum) -> list[dict]:
    """Public entry point — signature preserved for app.py."""
    loader = PyPDFLoader(file_path)
    data = loader.load()

    chunks, router_snippet = _collect_chunks(data, pagenum)
    if not chunks:
        print("[pipeline] no eligible chunks for requested pages")
        return []

    router = build_router()
    try:
        decision: RouteDecision = router.invoke({"snippet": router_snippet})
        category = decision.category
    except Exception as exc:
        print(f"[router] classification failed, defaulting to memorization: {exc}")
        category = "MEMORIZATION_BASED"

    print(f"[router] category = {category} (chunks={len(chunks)})")

    questions = asyncio.run(_generate_for_document_async(chunks, category))
    print(f"[pipeline] generated {len(questions)} questions")
    return questions
