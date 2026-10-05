# Full-passage read rubrics

The rubrics a reader applies to each record of a read packet (`data/scripts/sft_audit.py
read-packets`). The reader sees the full source passage(s), the question (or term) and the
answer, and returns one verdict per record:

- **defect**: drop. The record would teach something wrong.
- **minor**: keep. The record is correct and faithful, with a style flaw only.
- **ok**: keep. No flaw.

Verdicts go in `data/sft/read_filter.jsonl` with the fingerprint of the content read
(`sft_common.fingerprint`). `sft_assemble.py` drops defects and leaves out any record that must be
read and has no verdict for its current content.

## Closed-book

A record is a **defect** if the question and answer, as written, would teach something the passage
doesn't support, even when the answer value itself appears in the passage:

- **wrong_answer / unsupported:** the answer contradicts the passage, or the passage doesn't state it.
- **overclaim:** the question uses a stronger word than the passage.
  - Stronger words: must, required, mandated, prescribed, specified, maximum, minimum, allowable,
    standard.
  - The passage only says: should, recommended, typical, usually, may, one study found, or it is a
    figure caption.
- **misframing / boundary:** the question attaches the fact to a different situation, structure
  type, condition or direction. That includes a persona that moves a building provision to "a
  bridge project", and "less than 2 to 1" asked as "does not exceed 2 to 1".
- **ambiguous_question / not_standalone:**
  - several correct answers exist;
  - the question asks for a term or definition the passage doesn't give;
  - a document is credited with something it only cites;
  - the question doesn't name the document, edition or condition that another source would answer
    differently;
  - the question refers to something the student can't see.
- **worked_example_value:** the value is an input or result of one design example.
- **partial_answer:** a range is given as one end, an open range is closed, or a dropped
  alternative changes the rule.
- **garbled:** an extraction artifact makes the record wrong or unpinnable ("a C′", "T 1619").

A record is **minor** (kept) only if it is faithful and the flaw is style alone:

- a persona voice that doesn't change scope or meaning;
- awkward but accurate wording;
- a correct but low-value lookup (a section number or citation year the passage gives for exactly
  the thing asked);
- a harmless notation slip.

## Definition

A record is a **defect** if:

- **unfaithful:** it contradicts how the passage uses the term;
- **wrong_sense:** it defines a different meaning;
- **unsupported_addition:** it adds a specific claim the passage doesn't support that is doubtful or
  wrong;
- **circular:** it defines the term with itself;
- **not_a_term:** the "term" is a fragment, a bare symbol or a proper name.

A record is **minor** if it is correct but:

- too vague;
- more than one sentence;
- opens with a pronoun ("It is ...");
- generic (a common word with no domain sense).

## Grounded

A record is a **defect** if:

- **unsupported_claim:** a claim isn't stated in any passage;
- **wrong_citation:** a cited passage doesn't support the claim next to it;
- **missing_citation:** a sentence has no citation;
- **wrong_answer:** it contradicts the passages or answers a different question;
- **incomplete:** it drops a condition or caveat the passages give, so it misleads;
- **question_not_answerable:** the passages don't answer the question.

A record is **minor** if it is:

- verbose;
- restates the question;
- points to a figure instead of saying what it shows;
- slightly incomplete;
- stacks its citations.

## Abstain

A record is a **defect** if:

- **answer_present:** one of the four passages answers the question, fully or partly, including by
  simple arithmetic on its values or by stating the gold in other words;
- **partially_answerable:** a careful reader would answer rather than decline;
- **malformed:** the question is incoherent.

A record is **minor** if it is:

- **trivia:** a document detail (a year, a cost, a page);
- **ambiguous:** a vague question;
- **off_topic:** passages so unrelated that declining is trivially easy.
