# domain_qa review rubric

Every domain_qa item is reviewed against its source passage before any model generates on it (rule
9). The reviewer sees the question, the gold answer, its `answer_type` and the passage, and returns
**keep** or **reject**, with a one-line reason for a reject. Reject only for a concrete defect from
the list below, **never for difficulty**: the task is closed-book recall of facts from these
documents, and a low base rate is the point. When unsure, keep and flag; the lead re-checks every
flagged item.

## How the item will be scored (`eval/scorers.py`)

- **numeric** (the gold contains exactly one number): the first number in the model's answer line
  must be within 2% of the gold's number. Units are not compared, and number words don't count
  ("five" is not 5). A hyphen glued to a letter or digit is not a minus sign ("FEMA P-361" is 361).
- **exact** (anything else): lowercased, hyphens turned into spaces, punctuation except `. % / -`
  and the articles a/an/the removed. Then the model's first line must equal the gold, or contain
  it as a whole phrase.

## Reject when

1. **Wrong gold.** The passage doesn't state it, states something else, or the gold doesn't answer
   the question as asked. Examples: a start year given for a "minimum duration"; a period in seconds
   called a magnitude.
2. **Not the only reasonable answer.** Another short answer is also correct from the documents or
   standard practice. Examples: an edition-dependent clause number when the question names no
   edition; "typical" values with ranges.
3. **A correct answer would score wrong.** Examples:
   - two units or a fraction symbol ("1½ inches (38 mm)");
   - ranges or alternatives ("20 or 30 years");
   - a list answer;
   - a bare abbreviation, or a long descriptive phrase unlikely to be matched verbatim;
   - "unity" against 1.0;
   - a quantity with no unit in the question while the documents use two unit systems;
   - an id scored numerically where a natural correct form carries another number first.
4. **A wrong answer would score right.** The gold's single number isn't the fact: "Type 2",
   "Grade 50" when the question is about something else, or a gold whose first number is
   incidental.
5. **General knowledge.** An engineer would answer it without these documents: 5% damping, 0.85
   f'c, 270 ksi strand, k = 5.0 for unstiffened webs, standard Proctor percentages, textbook
   definitions, or a fact that follows from the question's own wording.
6. **Trivia.**
   - The document's own numbering: its chapter, figure or equation numbers, or "section 3 of this
     manual".
   - Publication years or authors.
   - Dates of datasets.
   - Which document cites which.

   Clause numbers of a named external standard (AASHTO LRFD, ASCE 7 with edition) are fine.
7. **One worked example's value.** A value computed, chosen or assumed for one example, case study
   or analysis, framed as a rule (a knowledge factor chosen for one example; one girder's
   stress).
8. **Misframed.** The question says "required", "minimum", "maximum" or "permitted" where the
   passage only describes, suggests, reports or calls it typical.
9. **Not standalone.** It refers to something the reader can't see (the figure, the table, the
   example, "this study") or doesn't name the structure, code or quantity well enough to have one
   answer.
10. **Layout locator.** The answer is a page, table, figure, equation, exhibit or plate number or a
    page count, or the question asks where in a document something appears. That is layout, not
    knowledge; it changes between editions, and it can leak from metadata.
    - `make_tasks.py --task-version 2` already removes these by rule (`qa_rules.is_locator`), so
      reject any that slip through.
    - "According to Table 3.4.1-1, what is the load factor for ...?" with the answer 1.25 is not a
      locator: it asks for a value.
    - Document ids and article or section numbers are not locators either; they are
      identifiers, kept and capped.

## Tag every kept item's answer kind

Kept items carry a regex-assigned `answer_kind` (`qa_rules.answer_kind`); correct it when it's
wrong. Stage 3 reports qa_acc per kind, so a wrong tag moves an item into the wrong column.

- **number:** a value with or without a unit (0.75, 24 ksi, 10:1).
- **identifier:** a document id or an article, section or clause number (EM 1110-2-2906, FEMA
  P-695, AASHTO LRFD 6.10.9.2).
- **term:** a named method, component, phenomenon or defined phrase (Bruun rule, judicious neglect).
- **other:** anything else (a range, a short descriptive phrase).

Identifiers are capped at 20% of the set. The assembler enforces the cap (the surplus goes to
`held_back.jsonl`), so reviewers only tag.

## Output

One JSON line per item: `{"id", "verdict": "keep" | "reject", "reason", "flag": true | false,
"answer_kind"}`. The reason is required for a reject and for a flag; `answer_kind` (the corrected
tag) for a keep. Tag corrections go into `eval/tasks/answer_kinds.jsonl` as `{"match": <the full
question>, "kind"}`.

The lead re-checks every flagged item and a fixed blind sample of 40 keeps and rejects, checking
the tag as well as the verdict, and records the result in `notes/decisions.md`: n checked, n
overturned, n tags corrected. Rejects go into `eval/tasks/rejects.jsonl` as
`{"task": "domain_qa", "match": <the full question>, "reason", "reviewed": "<date> <review name>"}`.
`make_tasks.py` drops them after sampling, so a reject never pulls in an unreviewed replacement.
