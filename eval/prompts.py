"""Prompt templates and generation settings for every eval task.

One fixed prompt per task, used for every model at every stage. Base models get the
text as-is (few-shot completion style); chat models get the same text wrapped in the
chat template by run_eval.py. The only variable across rows of the results table is
the model.

Changing anything in this file changes the benchmark: earlier rows in results/table.md
stop being comparable, and every run has to be regenerated (not just re-scored).
"""

# ---------------------------------------------------------------- domain_qa ---
# Closed-book short-answer QA. The header scopes the domain and asks for a bare answer, because
# scorers.qa_correct only reads the first line and the few-shot answers set the expected length.
QA_HEADER = (
    "You are a reference assistant for structural and civil engineering documents "
    "(USACE, FHWA, NIST, FEMA, NASA). Answer with the value, term or name only. "
    "No explanation.\n\n"
)


def qa_prompt(question: str, fewshot: list[dict]) -> str:
    """Header, the 3 few-shot pairs from eval/tasks/fewshot.jsonl, then the open "A:".

    The few-shot items are real corpus questions held out of domain_qa.jsonl (make_tasks.py),
    so they show the answer style without leaking any scored item. The prompt ends right after
    "A:" so a base model continues with the answer, and GEN stops it at the newline."""
    shots = "".join(f"Q: {s['question']}\nA: {s['answer']}\n\n" for s in fewshot)
    return f"{QA_HEADER}{shots}Q: {question}\nA:"


# -------------------------------------------------------------------- vocab ---
# Hand-written, textbook-level terms that show the target length and style: one sentence, the
# mechanism rather than a synonym. make_tasks.py removes these terms from vocab.jsonl, since a
# term used as a few-shot example would have its answer in the prompt.
VOCAB_SHOTS = [
    {
        "term": "load factor",
        "definition": "A multiplier applied to a nominal load in strength design to account "
        "for uncertainty in the magnitude of that load.",
    },
    {
        "term": "plastic hinge",
        "definition": "A region of a member where the full plastic moment has developed and "
        "rotation continues at approximately constant moment.",
    },
    {
        "term": "collector",
        "definition": "A diaphragm element that transfers diaphragm shear to a vertical "
        "element of the seismic force-resisting system.",
    },
]


def vocab_prompt(term: str) -> str:
    """Instruction, the 3 VOCAB_SHOTS, then "Term: <term>\\nDefinition:" for the model to finish.
    Closed-book: the source passage is not shown. The judge compares the output with the
    definition extracted from that passage (judge.vocab_rubric)."""
    shots = "".join(f"Term: {s['term']}\nDefinition: {s['definition']}\n\n" for s in VOCAB_SHOTS)
    return (
        "Define each structural engineering term in one precise sentence.\n\n"
        f"{shots}Term: {term}\nDefinition:"
    )


# ------------------------------------------------- grounded and adversarial ---
# Shared by both tasks, so the model can't tell from the prompt whether the answer is present:
# grounded items contain it, adversarial items don't. The abstain sentence must stay in sync
# with ABSTAIN_PHRASE below (scorers.abstained looks for it verbatim, lowercased).
GROUNDED_INSTRUCTIONS = (
    "Answer the question using only the passages below. After each claim, cite the id of "
    "the passage that supports it in square brackets, for example [doc:p12:c0]. "
    "If the passages do not contain the answer, reply exactly: Not in the provided passages."
)

# A one-line format example, with no passages of its own. The earlier version was a full worked
# example (two synthetic passages plus a Question/Answer), and it made Ministral 3 8B Instruct refuse
# nearly everything: 130/131 grounded answered "Not in the provided passages." (Mistral's API gave
# the same refusals, so it was the prompt, not the serving path). Without the fake passages the
# model answers and still declines unanswerable questions (API test on items not used for the
# first test: 12/12 grounded answered, 10/10 adversarial declined), while base models still see the citation
# format. The id "doc-a:p3:c0" never appears in a real context, so a model that copies it fails
# scorers.citations_valid.
GROUNDED_FORMAT = (
    'Example of the answer format (unrelated to the passages below): "For unusual load conditions '
    'the minimum factor of safety against sliding is 1.3 [doc-a:p3:c0]."'
)


def grounded_prompt(question: str, context: list[dict]) -> str:
    """Instructions, the one-line format example, the item's passages as "[chunk_id]\\ntext"
    blocks, then the question and an open "Answer:".

    `context` is the item's list of {chunk_id, text}: 4 passages for grounded (gold, two
    neighbours from the same document, one distractor from another document, shuffled by
    make_tasks.py), 3 for adversarial (none of which contains the answer)."""
    passages = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in context)
    return (
        f"{GROUNDED_INSTRUCTIONS}\n{GROUNDED_FORMAT}\n\n{passages}\n\nQuestion: {question}\nAnswer:"
    )


# ------------------------------------------------------- generation settings ---
# Greedy everywhere; seed fixed in run_eval. Stops keep base models from rambling into
# a new "Q:" block; harmless for chat models.
#   domain_qa / vocab: one line is the whole answer, so stop at the first newline. 48 tokens
#     fits any short answer; 80 fits one sentence of definition.
#   grounded / adversarial: answers can run to a few sentences, so allow more tokens and stop
#     only when the model starts a new "Question:" or a new "[passage]" block, i.e. starts
#     imitating the prompt instead of answering.
GEN = {
    "domain_qa": {"max_tokens": 48, "stop": ["\n"]},
    "vocab": {"max_tokens": 80, "stop": ["\n"]},
    "grounded": {"max_tokens": 300, "stop": ["\n\nQuestion:", "\n\n["]},
    "adversarial": {"max_tokens": 200, "stop": ["\n\nQuestion:", "\n\n["]},
}

# Overrides for chat models (run_eval --chat) on the one-line tasks. Ministral 3 Instruct opens
# with a header echoing the prompt's labels ("**Term: x**\n**Definition:** ..."), so the "\n" stop
# ended 273/303 vocab answers before the definition. For chat the stop is dropped, the budget
# covers the header, and run_eval keeps the first line with content (scorers.answer_line).
CHAT_GEN = {
    "domain_qa": {"max_tokens": 64, "stop": []},
    "vocab": {"max_tokens": 112, "stop": []},
}

# Lowercased form of the abstain sentence in GROUNDED_INSTRUCTIONS; scorers.abstained() checks
# for it before calling the judge.
ABSTAIN_PHRASE = "not in the provided passages"
