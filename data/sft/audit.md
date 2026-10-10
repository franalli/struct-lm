# SFT set: full-passage audit

Records read against their full source passage(s) (not a human read): `data/scripts/sft_audit.py`, verdicts in `data/sft/audit.jsonl`. Rates are count/n (share; Wilson 95% interval).

## Round 1

The set as first frozen (commit 3c86431): 40 records per synthetic format, 30 from Mistral Large 3 and 10 from Medium 3.5, disjoint from the 50 in review.md.

| format | defect | minor | ok |
|---|---|---|---|
| abstain | 0/40 (0%; 0%-9%) | 17/40 (42%; 29%-58%) | 23/40 (57%; 42%-71%) |
| closed_book | 8/40 (20%; 10%-35%) | 13/40 (32%; 20%-48%) | 19/40 (48%; 33%-63%) |
| definition | 3/40 (8%; 3%-20%) | 9/40 (22%; 12%-38%) | 28/40 (70%; 55%-82%) |
| grounded | 9/40 (22%; 12%-38%) | 9/40 (22%; 12%-38%) | 22/40 (55%; 40%-69%) |
| multi_step | 17/40 (42%; 29%-58%) | 15/40 (38%; 24%-53%) | 8/40 (20%; 10%-35%) |
| all | 37/200 (18%; 14%-24%) | 63/200 (32%; 25%-38%) | 100/200 (50%; 43%-57%) |

| teacher | defect | minor |
|---|---|---|
| mistral-large-2512 | 30/150 (20%; 14%-27%) | 44/150 (29%; 23%-37%) |
| mistral-medium-2604 | 7/50 (14%; 7%-26%) | 19/50 (38%; 26%-52%) |

| origin | defect | minor |
|---|---|---|
| eval_seen | 9/74 (12%; 7%-22%) | 21/74 (28%; 19%-40%) |
| ordinary | 28/126 (22%; 16%-30%) | 42/126 (33%; 26%-42%) |

Categories (defect and minor):

- abstain: ambiguous_question (5)
- abstain: off_topic_passages (10)
- abstain: trivia_question (3)
- closed_book: ambiguous_question (3)
- closed_book: awkward_phrasing (10)
- closed_book: notation (2)
- closed_book: partial_answer (1)
- closed_book: trivia (6)
- closed_book: unsupported (4)
- closed_book: worked_example_value (1)
- closed_book: wrong_answer (2)
- definition: awkward_phrasing (5)
- definition: generic_term (2)
- definition: too_vague (1)
- definition: unfaithful (3)
- definition: unsupported_addition (3)
- grounded: awkward_question (1)
- grounded: bundled_citations (3)
- grounded: figure_reference (1)
- grounded: incomplete (2)
- grounded: loose_paraphrase (4)
- grounded: missing_citation (5)
- grounded: question_not_answerable (1)
- grounded: slightly_incomplete (2)
- grounded: unsupported_claim (3)
- grounded: verbose (1)
- grounded: wrong_answer (1)
- multi_step: arithmetic_error (5)
- multi_step: awkward_phrasing (2)
- multi_step: contrived (11)
- multi_step: ill_posed (6)
- multi_step: malformed (4)
- multi_step: no_document_needed (12)
- multi_step: passage_reference (2)
- multi_step: value_leaked (2)
- multi_step: wrong_document_value (4)

### Round 1: every defect

- `fema-p-1100-2c:p12:c3:f2:closed_book:2` (closed_book, mistral-large-2512): wrong_answer. The question's "aspect ratio does not exceed 2 to 1" includes exactly 2:1, but the passage gives 1/10 only for "aspect ratio less than 2 to 1" and says "aspect ratio of 2 to 1 or more shall have ... 1/8", so the pair teaches the wrong boundary.
- `fema-p-1100:p110:c0:q0:closed_book:3` (closed_book, mistral-large-2512): unsupported, ambiguous_question. "SDS = 1.2" comes only from the caption "Figure 5.4-16 Earthquake Retrofit Schedule at SDS = 1.2 at front of garage...", one schedule among several, and nothing in the passage says it is a "minimum SDS value [that] must be used".
- `fhwa-hif19063:p237:c0:f0:closed_book:1` (closed_book, mistral-large-2512): worked_example_value. "Fy = specified minimum yield strength = 50 ksi" is the assumed input of the end-post design example (next to "Ag = ... = 215.5 inch2", "for the purpose of this example"), not a value the proposed specifications set for all noncomposite box members.
- `fhwa-hif19067-nov2021:p125:c1:f3:closed_book:1` (closed_book, mistral-medium-2604): wrong_answer. "AASHTO T 1619" is a PDF-extraction artefact (standard T 161 + footnote 19; likewise "M 1951" = M 195 + fn 51), as the same passage shows with "AASHTO T 27725 (ASTM C1202)" whose footnote reads "25 AASHTO T 277"; the ASTM C666 equivalent is AASHTO T 161.
- `usace-em-1110-1-1002:p13:c0:f0:closed_book:2` (closed_book, mistral-large-2512): unsupported. The passage says only that the description "Normally ... puts the individual within 100 meters (328 feet) of the mark"; it sets no "initial search radius" that the manual "require[s] my crew to use".
- `usace-em-1110-2-1413:p27:c0:f10:closed_book:1` (closed_book, mistral-large-2512): ambiguous_question, unsupported. The passage only lists "4) interceptor systems" in the feasibility sequence alongside gravity outlets, ponding and pumping stations, never defining them, so the "capture and redirect excess water" description is not in the passage and fits several interior-flood measures (or 'diversion').
- `usace-em-1110-2-1604:p153:c0:q1:closed_book:3` (closed_book, mistral-large-2512): unsupported, ambiguous_question. The passage gives no "approved limits": it reports one model study of the Arkansas River low-lift locks where "Conditions produced by the 8.9- and 10.4-sq-ft ports were rated as satisfactory" (12.7 sq ft result not given), while the desirable area vs lock width comes from Figure D-1.
- `usace-em-1110-2-1613:p39:c0:f2:closed_book:1` (closed_book, mistral-large-2512): partial_answer. The passage says blockage ratios vary "from 2 to 3 for very restricted narrow canals ... up to ... open channels at ratios of 20 or more", so "2 to 20" turns an open-ended upper end into a cap (fix: '2 to 20 or more').
- `fema-p-2208:p233:c0:f11:definition:1` (definition, mistral-medium-2604): unfaithful. The passage gives a condition, 'The foundation of the short wall or short braced frame is not the governing mechanism (in other words, stronger ...)', but the completion turns that condition into the definition, so 'short braced frame' (a braced frame of short height) is wrongly defined as a system whose foundation is stronger than the frame.
- `fhwa-sbdh-v04:p88:c0:f8:definition:1` (definition, mistral-medium-2604): unfaithful, awkward_phrasing. The passage says the tests were 'close to the knife-edge end conditions ... (but with less than rigid Y-axis restraint and considering some minor X-axis restraint)', so the ideal knife-edge has rigid Y-axis restraint and free X-axis rotation, yet the completion gives it 'minimal rotational restraint about the Y-axis' and describes the tests' deviations as the definition, opening with the pronoun 'It provides'.
- `fhwa-sbdh-v05:p23:c1:f6:definition:1` (definition, mistral-large-2512): unfaithful, unsupported_addition. The passage says 'towers can be constructed in less than optimal foundation conditions because having good rock near the surface is not a requirement', which the completion garbles into 'without requiring deep rock anchorage', and it never says the stays carry the deck directly, so nothing separates it from a suspension bridge.
- `fema-p-1051:p219:c0:q4:grounded:1` (grounded, mistral-large-2512): missing_citation. Format only: the first sentence (MCE case amplified by 2.0 plus the DE case) has no citation, though p219:c0 supports it; the second sentence is cited correctly.
- `fema-p-58-1:p268:c1:q4:grounded:1` (grounded, mistral-large-2512): unsupported_claim, missing_citation. 'This process accounts for insufficient data by relying on fewer analyses (m=5)' inverts p268:c1, which says the 4-number draw shows how insufficient data degrades the realizations (similar statistics but less precision than the full-rank m = 11 case); the first sentence also has no citation.
- `fema-p-749:p179:c0:q4:grounded:1` (grounded, mistral-large-2512): question_not_answerable, unsupported_claim. p179:c0 only lists 'Mike Mahoney (Project Officer)' under FEMA in a participants roster; 'oversees and manages the development of FEMA P-749 (2022)' is stated nowhere (nor is 2022), so the target should have declined or said only who held the title.
- `fhwa-sbdh-ex3:p19:c0:q4:grounded:1` (grounded, mistral-large-2512): wrong_answer. 'the top flange width meets the guideline of 85 times the shipping piece length' inverts Eq. (C6.10.3.4-1) in p19:c0, which is bfc >= L/85 (the worked check is 123(12)/85 = 17.4 in. <= 20 in.); it also drops 'used in conjunction with Eq. (6.10.2.2-2)'.
- `nasa-std-5020b:p46:c0:q4:grounded:1` (grounded, mistral-medium-2604): incomplete. The whole answer is 'Ppi-min = (1 -Γ)Tmin KnomD' copied from an extraction that lost the fraction bar, so it reads as a product instead of (1-Γ)Tmin/(Knom·D) (p46:c0 notes 1/Knom is the inverse nut factor), and none of the symbols p46:c0 defines (Γ preload variation, Tmin minimum effective torque, Knom, D) is explained.
- `usace-em-1110-1-4006:p70:c0:q4:grounded:1` (grounded, mistral-large-2512): missing_citation. Format only: the first sentence (air mixed with water and blown from the hole, reactions with contaminants, unrepresentative samples) has no citation, though p70:c0 supports it; the question's 'UST site investigations' framing appears in no passage.
- `usace-em-1110-2-1911:p19:c1:q4:grounded:1` (grounded, mistral-medium-2604): incomplete. It lists 'sprayed asphalt, bituminous materials, or resin emulsions' as protection against spalling and slaking but drops p19:c1's caveat that these 'do not always provide adequate protection' and that the asphalt emulsion at Waco Dam let surfaces spall and slake; it also omits that burlap mats are unsatisfactory.
- `usace-em-1110-2-2000:p91:c2:q4:grounded:1` (grounded, mistral-large-2512): missing_citation. Format only: the first sentence (pull the insert pipes above the grout surface before it stiffens and rod them clear) has no citation, though p91:c2 supports it; the second sentence is cited correctly.
- `usace-em-1110-2-6055:p15:c0:q4:grounded:1` (grounded, mistral-large-2512): missing_citation, unsupported_claim. The first sentence (adopting IENC 2.2 and 'participating in the IENC Harmonization Group') has no citation, and no passage says USACE participates in the IEHG (p15:c0 says only that USACE IENCs follow the IEHG-derived S-57 standard with extensions).
- `fema-p-1051:p944:c0:q0:multi_step:1` (multi_step, mistral-large-2512): value_leaked. The prompt states every document value (D = 706 lb, 10.08 ft x 3.5 ft, 100 psf), so 706 + 3,528 = 4,234 lb needs no recall; the arithmetic is right.
- `fema-p-2208:p335:c0:q1:multi_step:1` (multi_step, mistral-large-2512): ill_posed. The premise that EcIeff scales with the cube of the height ratio is physically wrong (EI of a geometrically scaled wall scales with L^4, i.e. 16x), is not in FEMA P-2208, and no document value is used, so the target teaches '2^3 = 8' as the stiffness scaling.
- `fema-p-2343-v1:p398:c0:q1:multi_step:1` (multi_step, mistral-large-2512): arithmetic_error, wrong_document_value, passage_reference. Treats overturning as a force balance (1600 lb / 25 psf = 64 sf) and ignores the 9-ft height lever arm; the moment balance 1600 lb x 9 ft / (25 psf x 4 ft) = 144 sf matches the document's ~140 sf (12'x12'), and the completion also says '(from passage)'.
- `fema-p-424:p181:c0:q1:multi_step:1` (multi_step, mistral-large-2512): ill_posed, contrived. With 3 engineers on distinct, indivisible 12-hour steps, 8 steps need 3 rounds = 36 h = 4.5 days, not the pooled 96/24 = 4 days (and the EO 11988 steps are sequential), so the stated answer does not follow from the setup.
- `fhwa-hif15016:p360:c0:q1:multi_step:1` (multi_step, mistral-large-2512): malformed. Completion has no worked steps or comparison, only 'Answer: 0' (the value is right: Nc = 2 < 3 and S = 14 ft > 13 ft).
- `fhwa-hif17019:p296:c0:q1:multi_step:1` (multi_step, mistral-large-2512): passage_reference, wrong_document_value. The completion says '90,000 psi (from passage)', and sizing a bar to 'safely carry' a load on its ultimate tensile strength misapplies the coupler-qualification value (yield, 60 ksi, would govern: 2.0 in²).
- `fhwa-hif23003:p171:c0:q0:multi_step:1` (multi_step, mistral-large-2512): arithmetic_error. 1542.21 x 9.81 x 1.524 = 23,056.7 J, not 23,036.57 J, so the answer is about 23.05-23.06 kJ, not 23.04 kJ; small, but the stated product is wrong.
- `fhwa-nhi-15-044:p111:c0:q0:multi_step:1` (multi_step, mistral-large-2512): arithmetic_error, no_document_needed. Fencepost error: 48 m / 6 m = 8 spaces gives 9 cross-frame lines with the ends (63 x 12 = 756 bolts) or 7 intermediate lines (49 x 12 = 588), never 8; the passage supplies no value either.
- `fhwa-nhi-15-058:p598:c0:q1:multi_step:1` (multi_step, mistral-large-2512): malformed. Completion has no worked steps, only 'Answer: 10.8 in²' (the value is right: 1.2 x n = 9 gives 10.8).
- `nist-gcr-17-917-47:p26:c0:q0:multi_step:1` (multi_step, mistral-large-2512): ill_posed. Topping thickness is a missing input: the 3-in. topping belongs to the brief's Example 4 building, not a TB13 requirement, and the prompt never identifies that building; the 25 psf superimposed load is an unused distractor (0.25 ft x 150 pcf = 37.5 psf is otherwise right).
- `usace-em-1110-1-4006:p237:c0:q1:multi_step:1` (multi_step, mistral-large-2512): arithmetic_error. The heat balance omits warming the meltwater from 0 to 15 C: m(334 + 4.18 x 15) = 41,800 kJ gives about 105.4 kg, not 125 kg.
- `usace-em-1110-2-1003:p213:c0:q0:multi_step:1` (multi_step, mistral-large-2512): arithmetic_error, wrong_document_value. 0.0624 x 692 = 43.18 km³, not 43.1968 (the rounded 43.2 survives), and the document says reservoirs around 'many' of the 692 Corps dams, so 692 reservoirs misreads it.
- `usace-em-1110-2-1603:p70:c1:q1:multi_step:1` (multi_step, mistral-large-2512): malformed. Completion has no worked steps, only 'Answer: 24 ft' (the value is right: 3d2 = 3 x 8 = 24 ft for the SPF, and 45 ft/s < 60 ft/s).
- `usace-em-1110-2-1612:p67:c0:q0:multi_step:1` (multi_step, mistral-medium-2604): ill_posed. 'Drop the albedo by 40% or more' is a lower bound and reads either as relative (60% to 36%, 64% absorbed) or absolute (60% to 20%, 80% absorbed), and the completion mislabels the snow-covered 60% as the 'albedo of black ice'; the 0.5-m thickness is unused.
- `usace-em-1110-2-1911:p19:c1:q0:multi_step:1` (multi_step, mistral-medium-2604): ill_posed, wrong_document_value. The document gives only 'a few minutes to several hours'; the completion invents 'few minutes = 2 minutes' and multiplies it by 2.5 sections, which is meaningless (exposure limits do not add per section), so 5 minutes is fabricated.
- `usace-em-1110-2-3006-2024apr22:p186:c1:q0:multi_step:1` (multi_step, mistral-large-2512): malformed, ill_posed. The prompt contains generator debris ('Name the document the voltage classification comes from...'), the power factor is not given (the completion's 'assumed 1 for MVA' is a false justification), and no document value is used; 50 x 0.985 = 49.25 MW otherwise.
- `usace-em-1110-2-3800:p320:c0:q1:multi_step:1` (multi_step, mistral-large-2512): value_leaked, contrived. The prompt states the document's value (two additional seismographs at the Government's discretion), so 4 + 2 = 6 is pure counting with no recall.

### Round 1: every minor

- `fema-p-1050-1:p356:c2:q0:abstain:1` (abstain, mistral-large-2512): off_topic_passages. Ceiling-penetration clearance question paired with paint procurement, isolator lambda factors, levee construction and importance-factor commentary; nothing on nonstructural components or ceilings.
- `fema-p-1050-1:p411:c0:q0:abstain:1` (abstain, mistral-large-2512): off_topic_passages. Passages are on TBM records, redundancy and seismic load effects, gravity-system overstrength and transfer diaphragms; nothing on ground-supported tanks or granular material, so declining is trivial.
- `fema-p-1050-1:p551:c0:q2:abstain:1` (abstain, mistral-medium-2604): trivia_question. Front-matter question about committee/task-group organization; isolation, cofferdam, snow-survey and SDC passages correctly do not answer it.
- `fema-p-1051:p44:c0:q1:abstain:1` (abstain, mistral-large-2512): ambiguous_question. 'Moderately low damping' and 'moderately large earthquake' make the question vague; the passages (isolator uplift, reliability table, masonry Cs = 0.106, weld detailing) give no 1.0 g peak response.
- `fema-p-2006:p205:c0:q1:abstain:1` (abstain, mistral-medium-2604): off_topic_passages. Foundation-flexibility question (gold 'Ac/A') paired with precast deck panels, shear-wall stress-strain curves, a column k factor and Euler buckling; no passage concerns foundations or soil stiffness.
- `fema-p-2006:p711:c0:q0:abstain:1` (abstain, mistral-large-2512): ambiguous_question, off_topic_passages. Gold 41.1 ft is a design-example value but the question asks generically about 'a multistory building', and the passages (FRP column, DBE drift table, flange lateral bending, column D-3 shear) never touch foundation damping.
- `fema-p-2343-v1:p125:c1:q1:abstain:1` (abstain, mistral-large-2512): off_topic_passages. Coupling-beam stiffness question paired with powerhouse shop equipment, concrete spall definitions, wood shear-wall overstrength and a COM archetype results table; nothing on coupling beams or effective stiffness.
- `fema-p-2355:p209:c0:q4:abstain:1` (abstain, mistral-large-2512): trivia_question. Front-matter question about why participants and trial users are listed; inspection and damage-class passages correctly do not answer it.
- `fema-p-50-1:p34:c0:q0:abstain:1` (abstain, mistral-large-2512): off_topic_passages. Soft-soil amplification question paired with integral abutments, tile-roof retrofit, fillet-weld terminations and anchor-bolt/crawl-space inspection; no passage concerns ground shaking or site effects.
- `fhwa-hif-18-046:p296:c0:q1:abstain:1` (abstain, mistral-medium-2604): ambiguous_question. 'The refined bridge analysis model described in HIF-18-046' does not identify which of the manual's example models is meant; the steel-girder and curved-girder modeling passages give no deck width.
- `fhwa-nhi-04-041:p452:c0:q1:abstain:1` (abstain, mistral-large-2512): ambiguous_question. Asks generically about 'a steel girder superstructure bridge' for an example-specific value (7.24 K/ft) and mixes 'factored' with Service I; passages (bearing stiffener, collapse table, column shear, C-PSW strength) do not give it.
- `fhwa-nhi-15-047:p420:c1:q1:abstain:1` (abstain, mistral-large-2512): off_topic_passages. Type VI girder end-block question paired with connection-plate fatigue, gusset chord splices, RC building drift checks and intake-tower lumped masses; nothing on prestressed girders or anchorage zones.
- `nist-gcr-13-917-24:p24:c0:q0:abstain:1` (abstain, mistral-large-2512): ambiguous_question. Gold is a fragment ('projected over the connection depth'), and passage nist-gcr-13-917-24:p7:c0 only says 'A story drift of approximately 2.5 % is commonly assumed as a target inelastic deformation ... prior to brace fracture', a system drift target rather than a stated joint rotation requirement, so the abstain holds but is a near-miss.
- `usace-em-1110-2-1406:p112:c0:q1:abstain:1` (abstain, mistral-medium-2604): trivia_question. A unit-conversion constant (4.186 J per gram-calorie) asked 'according to EM 1110-2-1406'; snow-hydrology and turbidity passages do not contain it, so the abstain is correct but low-value.
- `usace-em-1110-2-5025:p537:c0:q0:abstain:1` (abstain, mistral-large-2512): off_topic_passages. Batiquitos Lagoon beach-material question paired with museum-artifact restraint, a fine-grained soil test table, marsh-plant propagation and a DCR table; none concerns beach sediment, so declining is trivial.
- `usace-em-1110-2-5025:p720:c0:q0:abstain:1` (abstain, mistral-medium-2604): off_topic_passages. Question is about mature height of wild buckwheat, but passages cover polymer dosing, a consolidation table, MPRS spectra and sealing-bolt spacing; no vegetation content at all.
- `usace-em-1110-2-5025:p867:c0:q0:abstain:1` (abstain, mistral-medium-2604): off_topic_passages. Polymer storage question paired with box-section flexure, a tree-species table, oyster-larvae entrainment and P-155 fragility parameters; no passage concerns chemical clarification or storage containers.
- `fema-p-1026:p145:c0:f11:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing, trivia. "thermal expansion joints" appears only in passing in the passage; the question is a generic vocabulary definition (plain 'expansion joints' equally correct) with an irrelevant "reviewing plans for a municipality" persona.
- `fema-p-2082-1:p577:c0:f1:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Correct per "1.2 at 0.2 s and 1.25 at 1.0 s", but the "As part of my PE exam prep" persona is padding.
- `fema-p-695:p110:c0:f4:closed_book:2` (closed_book, mistral-large-2512): trivia, notation. Correct but taken from the running footer "FEMA P695 5: Nonlinear Model Development 5-17": a document-locator question with a header-style answer rather than 'Chapter 5, Nonlinear Model Development'.
- `fema-p-795:p108:c0:f0:closed_book:1` (closed_book, mistral-large-2512): notation. 1.2 is correct, but the garbled source symbols were rendered as "Q_PC/Q_RC" when the limited quantity is the ratio of overstrength ratios R_Q,PC / R_Q,RC.
- `fhwa-sbdh-v05:p24:c0:f2:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Supported by "shipping pieces up to 160 feet have been used", but "documented ... as having been successfully implemented in real-world projects" is a clunky rewording of 'have been used'.
- `usace-em-1110-1-1002:p13:c0:f3:closed_book:2` (closed_book, mistral-large-2512): trivia, awkward_phrasing. The URL is quoted correctly, but memorizing a (double-slash) web address is document trivia and the "forensic investigation of a structural failure" persona is irrelevant to survey-mark stability.
- `usace-em-1110-2-1009:p175:c0:f1:closed_book:2` (closed_book, mistral-medium-2604): awkward_phrasing. Supported by "RMS is consistently below two (2) for at least five (5) satellites", but the "municipal plan reviewer" persona is odd, "the specified threshold" leaves the value 2 unstated, and "at least" is dropped.
- `usace-em-1110-2-1613:p39:c0:f1:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Correct per "from Fh = 0.2 at BBR = 2 to Fh = 0.7 at BBR = 20", but the "For the PE exam" framing is irrelevant persona padding.
- `usace-em-1110-2-1901:p330:c0:q2:closed_book:1` (closed_book, mistral-large-2512): trivia. Correct (passage header is EM 1110-2-1901, Seepage Analysis and Control for Dams), but it is a document-identity question answerable only from the title, and "governs" overstates a guidance manual.
- `usace-em-1110-2-2002:p52:c0:f11:closed_book:2` (closed_book, mistral-medium-2604): awkward_phrasing. "Polymer impregnation" is listed in Table 4-2 and the monomer/polymerization description is correct, but the description is not in the passage and "When inspecting bridges" misframes a manual on civil-works concrete structures.
- `usace-em-1110-2-3006-2024apr22:p142:c0:f0:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Correct per "consider annual snow accumulation for determining the minimum fan heights", but the "FEMA mitigation planner" persona is incongruous.
- `usace-em-1110-2-3402:p74:c0:f5:closed_book:2` (closed_book, mistral-medium-2604): awkward_phrasing, trivia. Correct per "1,000 kip/in. constitutes a practical threshold for use with rigid walls (Consolazio et al., 2014)", but the "NASA structures analyst evaluating barge impact" persona is incongruous and a citation is document trivia.
- `usace-em-1110-2-5025:p697:c1:f10:closed_book:2` (closed_book, mistral-large-2512): trivia, awkward_phrasing. "storm surges" appears in the passage (CAD pits "may be exposed to ... storm surges") but the "abrupt, short-lived rises" definition is not from the passage, CAD pits are misnamed "containment areas", and it is general vocabulary rather than document knowledge.
- `fema-nehrp-2000-commentary:p20:c0:f6:definition:1` (definition, mistral-large-2512): too_vague. Broadly correct (the identified requirements and assumptions behind a design, recorded for future modifications), but 'ensure consistency and safety' is generic, and the passage ties it to identifying the seismic force-resisting system and its design level.
- `fema-p-1019:p53:c0:f11:definition:1` (definition, mistral-large-2512): generic_term. 'supply chains' is a general business word with no structural-engineering sense, although the definition is faithful to the passage's mention of energy resources.
- `fema-p-154:p199:c0:f10:definition:1` (definition, mistral-large-2512): generic_term. 'Supervising Engineer' is a project role title in the example RVS program ('remained available throughout the field screening to advise and consult'), not a technical term, though the definition is harmless and faithful.
- `fema-p-695:p313:c0:f7:definition:1` (definition, mistral-medium-2604): awkward_phrasing. Correct in substance (dispersion in collapse response across ground motion records), but it opens with the pronoun 'It quantifies' and doesn't say that it is a dispersion (lognormal standard deviation) value.
- `fhwa-nhi-15-044:p637:c0:f10:definition:1` (definition, mistral-large-2512): unsupported_addition. The passage only distinguishes 'Deck forms not in place' and 'in place'; calling them 'temporary structures' is too narrow, since stay-in-place deck forms are permanent (the reference definition says 'Temporary or permanent').
- `fhwa-nhi-23-024:p1217:c0:f8:definition:1` (definition, mistral-large-2512): awkward_phrasing. Correct content (surface hardness gives an estimate of in-place compressive strength), but 'through an underwater impact mechanism' is awkward, because the device is a waterproofed rebound hammer whose spring-mass impact is not underwater-specific.
- `fhwa-sbdh-v05:p24:c0:f8:definition:1` (definition, mistral-large-2512): unsupported_addition. Limiting lateral bracing to 'during construction' is too narrow: the passage lists 'lateral bracing, sway bracing and floor system members' among a truss's permanent members, and the sentence also switches from 'primary members' to 'the girder's'.
- `nasa-std-5001b:p24:c0:f7:definition:1` (definition, mistral-medium-2604): awkward_phrasing. Content is correct (the proof load as a multiple of the expected or limit load), but it opens with the pronoun 'It is the factor' instead of a noun-phrase definition.
- `usace-em-1110-2-1601:p9:c0:f6:definition:1` (definition, mistral-medium-2604): awkward_phrasing. Faithful to the passage's map, scale and profile steps, but it opens with 'It is the creation of', and 'preliminary layout' is closer to a section heading than a term.
- `fema-e-74:p56:c0:q4:grounded:1` (grounded, mistral-large-2512): verbose, loose_paraphrase. The first sentence is supported by p56:c0; the second ('incorporated into standards like ASCE/SEI 41 ... allowing flexibility in defining nonstructural performance levels') is off-question and overstates p56:c0, which says the concepts are 'gradually finding their way' into standards and that ASCE 41 gives a flexible approach to evaluating nonstructural components.
- `fema-nehrp-examples-v1:p373:c0:q4:grounded:1` (grounded, mistral-large-2512): slightly_incomplete. The connection types are supported by p373:c0, but the answer leaves out the required displacement capacity (DpI, not less than 0.5 in.; 1.5DpI, not less than 1.0 in., for sliding bearings without keepers) and the no-loss-of-vertical-support condition.
- `fema-p-2012:p135:c0:q4:grounded:1` (grounded, mistral-medium-2604): slightly_incomplete. The modeling description is supported by p135:c0, but it omits the actual assessment metric given in p136:c0 (ACMR from incremental dynamic analysis per FEMA P695, compared against a regular baseline archetype).
- `fhwa-hif19063:p171:c0:q4:grounded:1` (grounded, mistral-large-2512): figure_reference. Correctly restates p171:c0 (Eurocode elastic-to-ULS mapping applied to Eqs. 6.12.2.2.2g-2 and -3) but appends 'as illustrated by the bold solid grey curve in the plot', a figure the reader cannot see.
- `fhwa-nhi-15-044:p344:c0:q4:grounded:1` (grounded, mistral-large-2512): loose_paraphrase, bundled_citations. 'often utilizing falsework' is not stated: p345:c0 gives falsework or cable stays from temporary towers as alternatives; both ids are bundled at the end in the reverse order of the claims.
- `fhwa-nhi-16-016:p229:c0:q3:grounded:1` (grounded, mistral-medium-2604): bundled_citations. Each claim is supported (distortion-induced in p229:c0, transverse-element rotation in p228:c0, abrupt stiffness change in unstiffened web gaps in p227:c1), but all three ids are bundled at the end, so the claim-to-passage mapping is lost.
- `usace-em-1110-1-1802:p16:c1:q4:grounded:1` (grounded, mistral-large-2512): loose_paraphrase. Mostly supported, but 'layer depths' in sentence 1 is cited to p16:c1/p17:c0, which give only velocities from slopes, and sentence 2's claim that crossover distance and intercept time yield 'depths and velocities' stretches p18:c1 (which says the crossover equation is most useful for survey design).
- `usace-em-1110-2-4300:p94:c0:q4:grounded:1` (grounded, mistral-medium-2604): loose_paraphrase, bundled_citations. Vee-notch weirs in p94:c0 measure total cumulative gutter flow, so offering them as a way to measure individual drains 'separately' is loose; the directly supported method is p95:c0's weight or volume over a known time, and both ids are bundled at the end in reverse order.
- `usace-em-1110-2-6056:p198:c0:q3:grounded:1` (grounded, mistral-large-2512): awkward_question. The answer is supported, but the question turns a reference-list entry in p198:c0 into 'the required reference for identifying and mapping special hazard areas', when 44 CFR 65 is just one of the manual's required references (bibliographic trivia).
- `fema-p-2208:p172:c0:q1:multi_step:1` (multi_step, mistral-large-2512): no_document_needed, contrived. 0.0045 - 0.002 = 0.0025 = 2,500 microstrain is correct, but the yield strain is given, the section dimensions are unused, and the document only contributes that fiber models use plastic strain.
- `fema-p-58-6:p79:c0:q0:multi_step:1` (multi_step, mistral-medium-2604): contrived. 0.015 x 144 in. = 2.16 in. is correct, but 0.015 is just the drift-ratio selection shown in the PET screenshot, relabelled as a 'maximum allowable' drift.
- `fhwa-hif-18-046:p353:c0:q0:multi_step:1` (multi_step, mistral-large-2512): no_document_needed. 1200 x 0.65 = 780 kip-ft is correct, but the distribution factor is given in the prompt, so the manual contributes nothing.
- `fhwa-if-12-027:p149:c0:q0:multi_step:1` (multi_step, mistral-large-2512): no_document_needed. 8 x 0.80 = 6.4 mm is correct, but the 80% penetration requirement is stated in the prompt and the passage has no numeric value.
- `fhwa-nhi-15-047:p249:c0:q1:multi_step:1` (multi_step, mistral-medium-2604): no_document_needed. 1200 x 1.33 = 1596 kip-ft is correct, but IM = 33% is given in the prompt and absent from the passage, so nothing is recalled (IM is also applied to the whole moment, though AASHTO excludes the lane load).
- `nist-gcr-10-917-9:p7:c0:q0:multi_step:1` (multi_step, mistral-medium-2604): contrived, no_document_needed. $250,000 x 1.15 = $287,500 is correct, but the funding figures are invented and the contract number plays no role in the calculation.
- `nist-gcr-16-917-38:p15:c0:q0:multi_step:1` (multi_step, mistral-large-2512): contrived. Uses the 24-in. (610 mm) maximum stud spacing correctly (4 spaces x 0.61 m = 2.44 m), but 'maximum wall length with 5 studs' is an artificial counting exercise.
- `usace-em-1110-1-1005:p335:c1:q0:multi_step:1` (multi_step, mistral-large-2512): no_document_needed, contrived. 5 x $25,000 x 1.10 = $137,500 is correct, but cost and fee are invented and the document's point (one cooperative agreement may cover any number of projects) does not change the number.
- `usace-em-1110-1-4006:p237:c0:q0:multi_step:1` (multi_step, mistral-large-2512): awkward_phrasing. Correctly uses the 3.5 L/day lower bound (8 x 3.5 = 28 L), but the prompt asks 'per shift' while the document value is per day, leaving shift = day implicit.
- `usace-em-1110-1-400:p61:c0:q1:multi_step:1` (multi_step, mistral-large-2512): contrived. Correct (the table marks 'All Park Attendant campsites ... universally accessible' as Required, so 8 of 8), but there is no arithmetic step.
- `usace-em-1110-2-1612:p293:c1:q1:multi_step:1` (multi_step, mistral-large-2512): no_document_needed, contrived. 1,000,000 m² x 0.2 kg/m² = 200 t is correct, but the spread rate is given and the document's timing (2-3 weeks before breakup) and 50% coverage target do not enter the calculation.
- `usace-em-1110-2-1612:p337:c0:q1:multi_step:1` (multi_step, mistral-large-2512): no_document_needed, awkward_phrasing. Fr = 1.2/sqrt(9.81 x 3.5) = 0.205 is correct, but the document gives no numeric juxtaposition Froude number, so the asked underturning check is never made and nothing is recalled.
- `usace-em-1110-2-1902:p98:c0:q1:multi_step:1` (multi_step, mistral-large-2512): no_document_needed. 200 - 50 = 150 kPa is correct but generic effective-stress arithmetic; the 80 kPa shear strength is unused and no document value is needed.
- `usace-em-1110-2-1911:p19:c1:q1:multi_step:1` (multi_step, mistral-medium-2604): no_document_needed, contrived. lambda = 300/200 = 1.5 m uses only stated values; the document says only 'high frequency', so nothing is recalled, and the 5-m depth is unused.
- `usace-em-1110-2-5025:p739:c1:q0:multi_step:1` (multi_step, mistral-large-2512): no_document_needed, contrived. Answer (2,500/100 x 0.5 = 12.5 kg) uses only stated inputs; the document's content (collect May-July, store dry) plays no role and 'during the typical collection period' is a red herring.

## Round 2

After the fixes (multi_step dropped, every grounded sentence cited, every closed-book record filtered by a full-passage reader): a fresh 40 closed-book + 40 grounded, disjoint from round 1 and review.md.

| format | defect | minor | ok |
|---|---|---|---|
| closed_book | 10/40 (25%; 14%-40%) | 15/40 (38%; 24%-53%) | 15/40 (38%; 24%-53%) |
| grounded | 3/40 (8%; 3%-20%) | 8/40 (20%; 10%-35%) | 29/40 (72%; 57%-84%) |
| all | 13/80 (16%; 10%-26%) | 23/80 (29%; 20%-39%) | 44/80 (55%; 44%-65%) |

| teacher | defect | minor |
|---|---|---|
| mistral-large-2512 | 6/60 (10%; 5%-20%) | 18/60 (30%; 20%-43%) |
| mistral-medium-2604 | 7/20 (35%; 18%-57%) | 5/20 (25%; 11%-47%) |

| origin | defect | minor |
|---|---|---|
| eval_seen | 10/31 (32%; 19%-50%) | 11/31 (35%; 21%-53%) |
| ordinary | 3/49 (6%; 2%-17%) | 12/49 (24%; 15%-38%) |

Categories (defect and minor):

- closed_book: ambiguous_question (7)
- closed_book: awkward_phrasing (10)
- closed_book: boundary (1)
- closed_book: overclaim (2)
- closed_book: trivia (5)
- closed_book: unsupported (4)
- closed_book: worked_example_value (1)
- grounded: awkward_question (3)
- grounded: bundled_citations (2)
- grounded: incomplete (2)
- grounded: slightly_incomplete (5)
- grounded: unsupported_claim (1)
- grounded: verbose (1)

### Round 2: every defect

- `fema-p-2208:p653:c0:f0:closed_book:1` (closed_book, mistral-large-2512): ambiguous_question. 'a C′' is a garbled extraction symbol and the passage gives two stiff-diaphragm values ('a C = 0.5 for stiff diaphragms' and 'Cg = Modification factor for ground level walls ... = 1.0 for stiff diaphragms'), and only Cg is labelled a 'Modification factor', so the question does not pin down 0.5.
- `fema-p-424:p355:c0:f6:closed_book:2` (closed_book, mistral-large-2512): overclaim, ambiguous_question. The figure caption only says 'Bolted splice connectors are recommended to prevent free ends of connectors from being whipped around by wind', not that FEMA P-424 'prescribe[s]' them, and the question never says which connectors (what system) are meant.
- `fhwa-hif19067-nov2021:p60:c0:f6:closed_book:1` (closed_book, mistral-medium-2604): ambiguous_question, unsupported. Same fact as the :2 variant: several materials are 'engineered to reduce the weight of bridge superstructures' (e.g. FRP or aluminum decks), and 'maintaining structural performance' is not stated; the passage says only that benefits 'are related to the reduction in the weight of the structure' plus 'durability and extended service life'.
- `fhwa-hif19067-nov2021:p60:c0:f6:closed_book:2` (closed_book, mistral-medium-2604): ambiguous_question, unsupported. A reverse lookup with several correct answers (other lightweight deck materials such as FRP or aluminum also cut dead load), and 'without compromising their strength' is not in the passage, which says only that benefits 'are related to the reduction in the weight of the structure' plus 'durability and extended service life'.
- `fhwa-nhi-04-041:p237:c0:f0:closed_book:1` (closed_book, mistral-medium-2604): worked_example_value. 50 ksi is an assumed input of one design example ('where, from Design Step 4.1: Fy 50ksi = Minimum yield strength of the connected material'), not a rule, requirement or finding.
- `nist-gcr-11-917-11:p36:c0:f0:closed_book:2` (closed_book, mistral-large-2512): boundary. The question ties the 3 in to 'the event of a connection failure', but the passage gives it as a placement tolerance that prevents loss of integrity: 'oversize the plates to allow for misplacement up to 3 inches without compromising the integrity of the connection'.
- `usace-em-1110-2-2607:p54:c0:f3:closed_book:2` (closed_book, mistral-large-2512): ambiguous_question. More than one EM fits: the passage pairs 'EM 1110-2-1601 and EM 1110-2-1901 contain further design information relative to riprap and filter designs' but also says 'Gradation layers, size of stones, and extent of protection are usually designed ... using guidance contained in EM 1110-2-1605', and the unanchored question ('detailed design specifications') overstates 'further design information'.
- `usace-em-1110-2-2906:p58:c0:f3:closed_book:1` (closed_book, mistral-medium-2604): ambiguous_question, unsupported. The pile manual only cites the EM in passing ('in seismic Zones 0 and 1 (EM 1110-21902)') and never says it 'defines seismic Zones 0 through 4'; since the question doesn't name the citing document (EM 1110-2-2906), other USACE documents that map seismic zones could equally be the answer.
- `usace-em-1110-2-3401:p49:c0:f1:closed_book:2` (closed_book, mistral-medium-2604): overclaim. The question says each layer 'must' measure the range under a 'specification', but the manual only says the sealer 'should be thinned 15 percent by volume and applied to a dry film thickness of 37.5 to 50 µm' and reserves 'must' for other rules ('vinyl-type sealers must only be applied by conventional spray').
- `usace-em-1110-2-5025:p496:c0:f11:closed_book:2` (closed_book, mistral-large-2512): unsupported, ambiguous_question. The passage defines no term for erosion 'when water flow patterns are disrupted' near corals 'or similar structures'; it only says 'Erosion and scour at the base of the corals in the dredged area also may damage corals', and the target answers 'the term for the erosion' with 'erosion and scour'.
- `fema-p-2343-v1:p55:c1:q4:grounded:1` (grounded, mistral-medium-2604): incomplete. The question asks how the report explains the difference, but the answer only restates that collapse probability is higher (p55:c1) and leaves out the explanation p56:c0 gives: FEMA P-2139 identified archetype overstrength, which was consistently lower for the very high-seismic archetypes, as the key factor.
- `fhwa-hif19063:p285:c0:q4:grounded:1` (grounded, mistral-large-2512): unsupported_claim. "The longitudinal stiffeners ... are deemed adequate if the flange satisfies the unstiffened slenderness limit" is not stated in p285:c0, which only says the deflection-stiffness provision of E6.1.4 is not applicable in that case (adequacy also rests on other checks, and stiffeners serving a web in flexure must still meet 6.10.11.3), so a one-provision waiver becomes a general adequacy rule.
- `usace-em-1110-2-6051:p10:c0:q4:grounded:1` (grounded, mistral-medium-2604): incomplete. "Time-history analysis is required for the OBE when the ground motions are severe" drops the hedge in p10:c0, which says response spectrum is usually adequate and time-history analysis "may be required" for severe OBE motions, so a permissive provision is taught as mandatory.

### Round 2: every minor

- `fema-nehrp-2000-commentary:p380:c0:f2:closed_book:1` (closed_book, mistral-large-2512): trivia. Matches 'The estimate given in the Provision Sec 14.7.3.7.1.2', but it is a bare section-number lookup, and the commentary's own next heading '14.7.3.6.1.5 Sliding Resistance' suggests the cited number may be a source typo for 14.7.3.6.1.2, so it is worth verifying.
- `fema-p-1024:p32:c0:f5:closed_book:1` (closed_book, mistral-large-2512): trivia. Correct per the citation '(ATC, 2000)', but the publication year of a cited report is bibliographic trivia.
- `fema-p-2208:p619:c0:f1:closed_book:1` (closed_book, mistral-large-2512): awkward_phrasing. Correct per 'OCSW detailing must include 90-degree hooks at wall ends', though 'minimum hook angle' is an inference from a stated 90-degree requirement that 'could be adapted to 180-degree hooks'.
- `fema-p-695:p313:c0:f0:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. 0.40 matches 'Most structures are relatively ductile and have T > 3. For these structures βRTR should be taken as 0.40'; the 'building official' persona is unnecessary framing.
- `fhwa-sbdh-v19:p34:c0:f4:closed_book:2` (closed_book, mistral-medium-2604): awkward_phrasing. NBI Item 59 is right per the figure axis '(NBI Coding Item #59) Superstructure Condition Rating', but the 'overseeing quality control for bridge construction' persona is mismatched to an inspection condition rating.
- `nasa-std-5001b:p21:c0:f1:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. 1.2 is the Table 3 proof test factor for nonpressurized glass/ceramics under the Test approach ('Nonpressurized 3.0 1.2'), but 'When inspecting structures' mislabels proof testing as inspection.
- `nasa-std-5020b:p109:c0:f0:closed_book:1` (closed_book, mistral-large-2512): awkward_phrasing. 25 percent matches the non-separation-critical torque-control cell ('b. Greater of (1) 25 percent (if lubricated)'), but calling it the 'minimum ... allowed' ignores the 'a. Statistical basis' alternative in the same cell, and the source table text is garbled.
- `nist-gcr-11-917-15:p88:c1:f0:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. 15 km matches 'say 15 km for faults capable of generating earthquakes of magnitude 7.25', but 'for quality-control purposes' is an invented context (the passage is about correcting the design spectrum) and 'say' marks the distance as approximate.
- `usace-em-1110-1-2908:p26:c0:q0:closed_book:3` (closed_book, mistral-medium-2604): trivia. Correct per 'These criteria divide a typical curved envelope into two linear segments', but 'bilinear' already implies two, so the item teaches nothing beyond the word itself.
- `usace-em-1110-2-1611:p109:c0:q0:closed_book:1` (closed_book, mistral-large-2512): trivia. Matches 'Our nations wetlands have been diminishing rapidly during the past half century', but it is a non-engineering, time-relative remark from a 1980 manual.
- `usace-em-1110-2-1902:p74:c0:f4:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Spencer's Method is right per 'where major structures are designed using the Simplified Bishop Method, the final design should be checked using Spencer's Method', but the 'forensic investigation of a slope failure' framing is off-context for a design-check recommendation.
- `usace-em-1110-2-2201:p76:c0:q1:closed_book:1` (closed_book, mistral-large-2512): trivia. Correct per 'The standard test method for measurement of the concrete properties is given in Chapter 9', but the answer is an internal chapter locator, not a test method.
- `usace-em-1110-2-2901:p102:c0:q1:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. 50 percent is right, but the passage reports practice rather than recommending it ('For preliminary use, a transient pressure 50 percent higher than the operating design pressure is often used'), and 'what percentage of the operating design pressure' does not match the '50 percent higher' answer form.
- `usace-em-1110-2-2902a:p78:c0:f2:closed_book:2` (closed_book, mistral-medium-2604): awkward_phrasing. Correct per 'proper design of the inlet and outlet structures per EM 1110-2-1602 reduces the probability of erosion initiating', but the 'reviewing municipal plans' scenario is an odd framing the passage doesn't support.
- `usace-em-1110-2-3006-2024apr22:p224:c0:f2:closed_book:1` (closed_book, mistral-large-2512): awkward_phrasing. The question asks for a 'specific guidance document' but the passage names none ('These valves must comply with the applicable HSS guidance of USACE'), so the target is a generic phrase rather than a document designation.
- `fema-nehrp-2000-commentary:p116:c1:q4:grounded:1` (grounded, mistral-large-2512): slightly_incomplete. Supported by p116:c1, but it drops "independently" (the 1% forces are applied independently in each of two orthogonal directions), which leaves it ambiguous whether they act simultaneously.
- `fema-p-1051:p840:c0:q4:grounded:1` (grounded, mistral-large-2512): awkward_question. The answer (dampers on interior gravity lines B, G, 2, 5 to avoid perimeter bracing) matches p840:c0, but the question wrongly presumes the dampers are part of the SFRS, which is the perimeter SMRF in p840:c0 and is designed independently of the damping system per p839:c1.
- `fema-p-2078:p154:c0:q4:grounded:1` (grounded, mistral-large-2512): bundled_citations. Correct (C = A x B is in p154:c0), but the two ids are stacked at the end of one sentence instead of each sitting beside its claim (p153:c1 only supports the period dependence).
- `fema-p-58-2:p33:c0:q4:grounded:1` (grounded, mistral-large-2512): verbose, slightly_incomplete. The first two sentences are supported by p33:c0, but the third ("More detailed methods allow custom development") is a loose paraphrase that ties custom fragilities to detailed analysis, while the passage says any combination of options may be used, and the answer omits that point.
- `fhwa-nhi-15-047:p183:c0:q4:grounded:1` (grounded, mistral-large-2512): bundled_citations. Correct (e = 0.77 + de/9.1, reconstructed from the garbled Eq. 4.4.2.2.2.2-2 and confirmed by the worked value 0.990; -1.0 <= de <= 5.5 ft per p183:c0 and p181:c0), but both ids are stacked at the end of one sentence.
- `nist-gcr-22-917-50:p444:c0:q4:grounded:1` (grounded, mistral-medium-2604): awkward_question, slightly_incomplete. The answer says what the trigger was (first-story masonry walls, matching the surveys) as p444:c0 states, but not how it was identified: the drift-ratio histories in p442:c1 show the walls losing capacity near 62.9 s and overloading the shear-critical columns.
- `usace-em-1110-2-1602:p41:c0:q4:grounded:1` (grounded, mistral-medium-2604): slightly_incomplete, awkward_question. The question asks where the sluices go, but the answer only places the air vent intakes in the crest pier and never says the sluices themselves sit on the spillway monolith center line (p41:c0), which is the link the passage makes.
- `usace-em-1110-2-3200:p235:c0:q2:grounded:1` (grounded, mistral-large-2512): slightly_incomplete. Tucking the strand ends back into the rope is supported by p235:c0, but the second splicing method the same definition gives (loops or eyes formed by mechanical attachments pressed onto the rope) is left out.

## Round 3

After the stricter second filter pass over the closed-book records pass 1 had kept as minor: a fresh 40 closed-book, disjoint from rounds 1-2 and review.md.

| format | defect | minor | ok |
|---|---|---|---|
| closed_book | 2/40 (5%; 1%-17%) | 18/40 (45%; 31%-60%) | 20/40 (50%; 35%-65%) |
| all | 2/40 (5%; 1%-17%) | 18/40 (45%; 31%-60%) | 20/40 (50%; 35%-65%) |

| teacher | defect | minor |
|---|---|---|
| mistral-large-2512 | 1/30 (3%; 1%-17%) | 15/30 (50%; 33%-67%) |
| mistral-medium-2604 | 1/10 (10%; 2%-40%) | 3/10 (30%; 11%-60%) |

| origin | defect | minor |
|---|---|---|
| eval_seen | 1/32 (3%; 1%-16%) | 15/32 (47%; 31%-64%) |
| ordinary | 1/8 (12%; 2%-47%) | 3/8 (38%; 14%-69%) |

Categories (defect and minor):

- closed_book: ambiguous_question (1)
- closed_book: awkward_phrasing (8)
- closed_book: boundary (1)
- closed_book: notation (3)
- closed_book: trivia (7)
- closed_book: unsupported (1)
- closed_book: verbose (2)

### Round 3: every defect

- `usace-em-1110-1-2907:p28:c0:q1:closed_book:2` (closed_book, mistral-medium-2604): boundary, unsupported. The passage says 'There are no restrictions on block shapes and no limits to the magnitude of displacement and rotations'; it gives no limit (or absence of one) on the number of blocks or geometries, so the count question asserts something the passage never states.
- `usace-em-1110-2-3001:p21:c2:f1:closed_book:2` (closed_book, mistral-large-2512): ambiguous_question. The value matches 'from 12 percent to 18 percent for smaller cranes', but the question never names EM 1110-2-3001, and other crane-runway provisions (e.g. ASCE 7 vertical impact of 25%/10% by crane operation type) give different impact allowances for crane runways in general.

### Round 3: every minor

- `fema-p-154:p117:c0:f3:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing, verbose. 17 is correct ('each of the 17 FEMA Building Types'), but the PE-exam persona framing is odd and the completion repeats 'FEMA Building Types' instead of the bare number.
- `fema-p-2208:p418:c1:f4:closed_book:1` (closed_book, mistral-large-2512): trivia. Correct per 'effective flexural stiffness values at General Yield, ECEIeff, proposed in Chapter 1', but it only teaches a chapter locator.
- `fema-p-2355:p7:c0:f1:closed_book:1` (closed_book, mistral-large-2512): trivia. The contract number matches 'under contract HSFE60-17-D-0002' but is administrative trivia; the preface itself calls the report 'FEMA P-2335', not P-2355.
- `fema-p-424:p355:c0:f2:closed_book:1` (closed_book, mistral-large-2512): trivia. Correct per 'provisions should be made to accommodate disruption of municipal utilities, as discussed in 6.3.5.1, 6.3.5.2, and 6.3.5.3', but it only teaches section-number locators.
- `fhwa-hif22031:p59:c0:q1:closed_book:2` (closed_book, mistral-large-2512): notation. Matches the PCI Bridge Design Manual (2014) row 'Max. aggregate should be smaller than tightest space concrete is to fill', but the completion clips it to 'smaller than tightest space'.
- `fhwa-nhi-15-047:p1169:c0:q0:closed_book:3` (closed_book, mistral-large-2512): awkward_phrasing. Zero is correct ('for straight spans or segments, Fp may be taken equal to zero'), but 'specify as the default' turns a permission into a default, and Fp is not tied to its shear-connector context.
- `fhwa-nhi-15-047:p154:c1:f3:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Correct per 'one-half the distance to the adjacent interior girder (or web) plus the full deck overhang width', but the PE-exam persona is filler, and 'must' hardens 'may be taken'.
- `fhwa-nhi-16-016:p212:c0:f0:closed_book:2` (closed_book, mistral-medium-2604): awkward_phrasing. Correct per 'Detailing to minimize the potential for distortion-induced fatigue ... is specified in AASHTO LRFD Article 6.6.1.3', but 'in the context of a steel bridge girder failure analysis' is irrelevant framing.
- `fhwa-sbdh-ex2b:p19:c0:f0:closed_book:2` (closed_book, mistral-large-2512): trivia. Correct per 'The AASHTO LRFD (7th Edition, 2014)', but which edition a design example cites is bibliographic trivia, not engineering knowledge.
- `nist-gcr-16-917-39:p261:c0:q0:closed_book:3` (closed_book, mistral-large-2512): trivia. Correct per Table A-2 ('2 96-110 mph'), but the Saffir-Simpson category range is general knowledge, and the power-grid framing adds nothing.
- `usace-em-1110-2-1413:p27:c0:f1:closed_book:2` (closed_book, mistral-medium-2604): trivia, awkward_phrasing. Chapter 7 is correct ('risk analysis framework as discussed in Chapter 7'), but the question is a chapter locator, and 'from a dam safety perspective' misframes an interior-area hydrology manual.
- `usace-em-1110-2-1604:p161:c0:f3:closed_book:2` (closed_book, mistral-large-2512): verbose. Correct per 'A 1,270- by 110-ft lock requires a 4-min valve time for all lifts', but the completion '4-min valve time' adds words to the bare value '4 min'.
- `usace-em-1110-2-1605:p67:c0:f6:closed_book:1` (closed_book, mistral-large-2512): trivia. Correct per 'High vertical-lift gates are sometimes split into two or more sections in order to reduce hoist capacity', but 'moving vertically' in the question gives the answer away.
- `usace-em-1110-2-1902:p74:c0:f5:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Correct per 'An approximate check of calculations can also be performed using the Ordinary Method of Slices, although the OMS will usually give a lower value', but the 'USACE dam safety review' persona is gratuitous.
- `usace-em-1110-2-2200:p12:c1:f0:closed_book:2` (closed_book, mistral-medium-2604): awkward_phrasing. ASTM C 512 is correct ('creep should be based on the standard test for creep of concrete in compression (ASTM C 512)'), but the 'state DOT bridge designer' persona asking about gravity dams is incongruous.
- `usace-em-1110-2-3800:p194:c0:f6:closed_book:2` (closed_book, mistral-large-2512): awkward_phrasing. Answer is right ('A jack-up barge eliminates this concern'), but 'is deployed' implies standard practice while the passage notes jack-up barges 'are far more expensive to operate and are not as readily available'.
- `usace-em-1110-2-5025:p843:c0:f0:closed_book:2` (closed_book, mistral-large-2512): notation. 2.0 g/L is right ('Fill a 1- or 2-L beaker with a 2.0-g/L suspension'), but the completion copies the noun ('2.0-g/L suspensions') instead of giving the bare value.
- `usace-em-1110-2-6053:p54:c0:f10:closed_book:2` (closed_book, mistral-large-2512): notation. Correct per 'damage control performance requirements for MDE loadings', but the completion is an unexpanded acronym ('MDE loadings'), and the question asks for a name.
