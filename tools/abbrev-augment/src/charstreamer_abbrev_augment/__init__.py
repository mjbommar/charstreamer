"""Abbreviation-aware synthetic training data generator for charstreamer.

Produces JSONL records compatible with the trainer
(``crates/charstreamer-segmentation/examples/train_sentence_burn.rs``) and the
existing ``data/synthetic/kl3m_streaming_spans_*.jsonl`` schema.

Each record has the form::

    {"text": "...", "spans": [{"label": "sentence", "start": int, "end": int,
                               "char_start": int, "char_end": int,
                               "right_open": false}, ...]}

Records contain 1-4 sentences combined from a library of templates. Templates
weave abbreviations (titles, citations, decimals, addresses, time, etc.) into
contexts that exercise both *no-break* (the period is mid-sentence) and *break*
(the period really is a sentence end) shapes.

Lexicons (names, places, citations, etc.) are intentionally disjoint from any
eval suite shipped under ``data/eval/``, so that gains on the eval reflect
generalization, not memorization. To extend the generator with domain-specific
content, add to the lexicon constants near the top of this module and adjust
the ``TEMPLATES`` list.

Pure standard library; deterministic given ``--seed``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Lexicons — intentionally disjoint from eval_suite_v2.jsonl
TITLE_DR = ["Dr. Andersen", "Dr. Bhatt", "Dr. Garcia", "Dr. Hsu", "Dr. Klein", "Dr. Marsh", "Dr. Okafor", "Dr. Romero", "Dr. Tanaka", "Dr. Whitfield"]
TITLE_MR = ["Mr. Carter", "Mr. Donovan", "Mr. Edwards", "Mr. Fleming", "Mr. Hayashi", "Mr. Iyer", "Mr. Mortensen", "Mr. Petrov", "Mr. Salazar", "Mr. Yamamoto"]
TITLE_MRS = ["Mrs. Acosta", "Mrs. Beaumont", "Mrs. Castellanos", "Mrs. Doyle", "Mrs. Engstrom", "Mrs. Fontaine", "Mrs. Goldberg", "Mrs. Holloway", "Mrs. Ito", "Mrs. Jankowski"]
TITLE_MS = ["Ms. Almeida", "Ms. Brennan", "Ms. Choudhury", "Ms. Delgado", "Ms. Eriksson", "Ms. Fernandes", "Ms. Greenberg", "Ms. Hassan", "Ms. Ishikawa", "Ms. Jakobsen"]
TITLE_PROF = ["Prof. Aldrich", "Prof. Bekele", "Prof. Cervantes", "Prof. Devereux", "Prof. Erikson", "Prof. Faulkner", "Prof. Goyal", "Prof. Hartmann", "Prof. Ivanova", "Prof. Jorgensen"]
TITLE_REV = ["Rev. Abernathy", "Rev. Caldwell", "Rev. Davenport", "Rev. Eastman", "Rev. Fitzgerald"]
SUFFIX_JR_SR = ["John Adams, Jr.", "Robert Brooks, Sr.", "Michael Chen, Jr.", "Daniel Evans, Sr.", "Patrick Fox, Jr.", "Thomas Garrett, Sr."]
CORP = ["Acme Holdings Inc.", "Beacon Systems Inc.", "Cygnus Labs Ltd.", "Delta Pharma Ltd.", "Echo Industries Co.", "Frontera Corp.", "Galileo Corp.", "Helios Co.", "Innova Inc.", "Junix Ltd.", "Keystone Inc.", "Lumen Corp."]
PLACE_ST_NAMED = ["St. Augustine", "St. Petersburg", "St. Cloud", "St. Albans", "St. Charles", "St. George"]
ADDRESS_ST = ["742 Evergreen St.", "1600 Cedar Ave.", "350 Maple Blvd.", "88 Oak St.", "401 Pine Ave."]
ACRONYMS = ["U.S.", "U.K.", "E.U.", "U.A.E.", "U.N."]
CITATIONS = ["410 U.S. 113 (1973)", "347 U.S. 483 (1954)", "163 U.S. 537 (1896)", "521 U.S. 793 (1997)", "576 U.S. 644 (2015)"]
CASE_NAMES = ["Anderson v. Baxter", "Carlson v. Drummond", "Eaton v. Fenwick", "Garrison v. Holcomb", "Ito v. Jamison", "Kettering v. Lambert"]
DEGREES = ["a Ph.D.", "an M.D.", "an M.S.", "a B.S.", "a B.A.", "a J.D.", "an LL.M.", "a D.D.S."]
MONTHS_ABBREV = ["Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
DAYS_ABBREV = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]
TIME_AM = ["7 a.m.", "8 a.m.", "10 a.m.", "11 a.m."]
TIME_PM = ["1 p.m.", "3 p.m.", "4 p.m.", "6 p.m.", "7 p.m.", "8 p.m.", "10 p.m."]
DECIMALS = ["3.14", "2.718", "1.618", "9.81", "6.022"]
SECTION_REFS = ["Section 2.4.1", "Section 5.6.7", "Article 3.2.1", "Rule 12.4", "Clause 8.5.2"]
PRICES = ["$12.95", "$49.99", "$1,234.56", "$0.99"]
URL_HOSTS = ["docs.example.com", "blog.acme.org", "wiki.demo.net", "api.test.io"]
ETC_PHRASES = ["apples, oranges, pears, etc.", "books, papers, pens, etc.", "trains, planes, buses, etc."]
EG_IE = ["e.g.", "i.e."]

# Sentence pools — these are filler sentences placed before/after abbreviations.
# All correctly terminated.
SENT_OPENERS = [
    "The board approved the plan.",
    "She closed the door behind her.",
    "It was raining heavily.",
    "The conference began on time.",
    "Markets opened with strong gains.",
    "He poured a glass of water.",
    "The committee adjourned for lunch.",
    "Light streamed through the windows.",
    "Critics praised the performance.",
    "The investigation continued for weeks.",
    "She filed the brief on Friday.",
    "Visitors arrived in small groups.",
    "He took notes throughout the meeting.",
    "The proposal received unanimous support.",
    "Snow fell quietly through the night.",
]
SENT_FOLLOWUPS = [
    "Everyone agreed it was time.",
    "The room fell silent.",
    "Reporters waited outside.",
    "The news spread quickly.",
    "Witnesses corroborated the account.",
    "Many questions remained unanswered.",
    "The decision drew mixed reactions.",
    "Several copies were distributed.",
    "Further details would emerge later.",
    "The ruling took immediate effect.",
    "Observers noted the timing.",
    "The speaker yielded the floor.",
    "Members began to file out.",
    "The chair called for a vote.",
    "Photographers captured the moment.",
]


def pick(rng: random.Random, lst: list[str]) -> str:
    return rng.choice(lst)


# Templates: each returns (text, [(sentence_text, ...)]) where each sentence_text
# is the exact substring of `text` that should be a sentence span.

def t_title_subject(rng):
    title = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_MS + TITLE_PROF + TITLE_REV)
    verb_phrase = pick(rng, [
        "addressed the audience at length.",
        "filed a motion that morning.",
        "reviewed the documents carefully.",
        "approved the request without delay.",
        "denied any wrongdoing.",
        "left the building shortly after.",
        "spoke for over an hour.",
        "signed the affidavit yesterday.",
    ])
    s1 = f"{title} {verb_phrase}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_two_titles(rng):
    a = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_PROF)
    b = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_PROF)
    while b == a:
        b = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_PROF)
    s1 = f"{a} and {b} disagreed about the approach."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_title_at_time(rng):
    title = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_MS)
    time = pick(rng, TIME_AM + TIME_PM)
    s1 = f"{title} arrived at {time} for the meeting."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_title_no_break(rng):
    title = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_MS + TITLE_PROF)
    rest = pick(rng, [
        "had not been informed of the change.",
        "was the only one present at the time.",
        "later confirmed the report in writing.",
        "remained in chambers throughout the day.",
        "later stated the matter was closed.",
    ])
    s1 = f"{title} {rest}"
    return s1, [s1]


def t_corp_inline(rng):
    corp = pick(rng, CORP)
    s1 = f"{corp} announced its quarterly results."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_corp_no_break(rng):
    corp = pick(rng, CORP)
    s1 = f"He worked at {corp} for many years."
    return s1, [s1]


def t_corp_end(rng):
    corp = pick(rng, CORP)
    s1 = f"The deal was concluded with {corp}"
    return f"{s1}.", [f"{s1}."]


def t_st_named_place(rng):
    place = pick(rng, PLACE_ST_NAMED)
    s1 = f"The conference was held in {place} last year."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_st_named_no_break(rng):
    a = pick(rng, PLACE_ST_NAMED)
    b = pick(rng, PLACE_ST_NAMED)
    while b == a:
        b = pick(rng, PLACE_ST_NAMED)
    s1 = f"They drove from {a} to {b} in one afternoon."
    return s1, [s1]


def t_address(rng):
    addr = pick(rng, ADDRESS_ST)
    s1 = f"He moved to {addr} earlier this year."
    return s1, [s1]


def t_us_acronym(rng):
    s1 = "The U.S. delegation arrived first."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_acronym_inline(rng):
    a = pick(rng, ACRONYMS)
    s1 = f"Officials from the {a} attended the summit."
    return s1, [s1]


def t_acronym_pair(rng):
    a, b = rng.sample(ACRONYMS, 2)
    s1 = f"The {a} and {b} cooperated on the program."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_citation(rng):
    cite = pick(rng, CITATIONS)
    s1 = f"The court relied on {cite} in its analysis."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_case_v(rng):
    case = pick(rng, CASE_NAMES)
    s1 = f"In {case} the panel applied a different standard."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_cf_citation(rng):
    case = pick(rng, CASE_NAMES)
    s1 = f"Cf. {case} for analogous reasoning."
    return s1, [s1]


def t_id_citation(rng):
    s1 = "See id. at 45."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_eg_inline(rng):
    e = pick(rng, EG_IE)
    s1 = f"Many factors, {e} cost and timing, influenced the decision."
    return s1, [s1]


def t_etc_inline(rng):
    p = pick(rng, ETC_PHRASES)
    s1 = f"He purchased {p} for the trip."
    return s1, [s1]


def t_etc_break(rng):
    p = pick(rng, ETC_PHRASES)
    s1 = f"He purchased {p}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_decimal_inline(rng):
    d = pick(rng, DECIMALS)
    s1 = f"The constant {d} appears throughout the analysis."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_section_ref(rng):
    sec = pick(rng, SECTION_REFS)
    s1 = f"{sec} addresses this directly."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_price(rng):
    p = pick(rng, PRICES)
    s1 = f"The item costs {p} after tax."
    return s1, [s1]


def t_time_break(rng):
    t = pick(rng, TIME_AM + TIME_PM)
    s1 = f"By {t} the venue had emptied."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_url(rng):
    h = pick(rng, URL_HOSTS)
    s1 = f"Visit {h} for the latest version."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_url_no_break(rng):
    h = pick(rng, URL_HOSTS)
    s1 = f"The materials live at {h} for now."
    return s1, [s1]


def t_month_abbrev(rng):
    m = pick(rng, MONTHS_ABBREV)
    day = rng.randint(1, 28)
    year = rng.randint(2010, 2024)
    s1 = f"On {m} {day}, {year}, the order took effect."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_day_abbrev(rng):
    d = pick(rng, DAYS_ABBREV)
    s1 = f"The session resumes {d} morning at the courthouse."
    return s1, [s1]


def t_jr_sr(rng):
    n = pick(rng, SUFFIX_JR_SR)
    s1 = f"{n} testified before the panel last week."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_jr_sr_no_break(rng):
    n = pick(rng, SUFFIX_JR_SR)
    s1 = f"The estate was administered by {n} for two years."
    return s1, [s1]


def t_degree_end(rng):
    d = pick(rng, DEGREES)
    s1 = f"She holds {d} from a top university."
    return s1, [s1]


def t_clean(rng):
    n = rng.randint(2, 4)
    sents = rng.sample(SENT_OPENERS + SENT_FOLLOWUPS, n)
    text = " ".join(sents)
    return text, sents


def t_long_abbrev_mix(rng):
    titles = rng.sample(TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_MS, 3)
    s1 = f"{titles[0]}, {titles[1]}, and {titles[2]} attended the briefing together."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_question(rng):
    questions = [
        "Did anyone object to the motion?",
        "Where did they stop for lunch?",
        "Could the appeal still proceed?",
        "Would the parties accept the offer?",
        "Was the document properly notarized?",
    ]
    s1 = pick(rng, questions)
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_exclamation(rng):
    excls = [
        "Stop right there!",
        "What a remarkable result!",
        "How quickly things changed!",
        "That settles it!",
        "Watch out below!",
    ]
    s1 = pick(rng, excls)
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_quoted_dialogue(rng):
    quotes = [
        '"Stop talking," he said.',
        '"That is enough," she replied.',
        '"Hello there," they answered.',
        '"Move forward," she ordered.',
        '"Stay calm," he advised.',
    ]
    s1 = pick(rng, quotes)
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_quoted_end(rng):
    inners = ["Hello.", "Goodbye.", "Stop.", "I refuse.", "Come closer."]
    inner = pick(rng, inners)
    s1 = f'He said, "{inner}"'
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_paren_end(rng):
    inners = [
        "He left (quietly).",
        "She arrived (early).",
        "The door slammed (loudly).",
        "They departed (without warning).",
    ]
    s1 = pick(rng, inners)
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_short_consec(rng):
    n = rng.randint(3, 5)
    nouns = ["One.", "Two.", "Three.", "Four.", "Five.", "Six.", "Yes.", "No.", "Maybe.", "Sure.", "Indeed."]
    sents = rng.sample(nouns, n)
    return " ".join(sents), sents


def t_lowercase_continuation(rng):
    # Real sentences sometimes continue with lowercase (informal style).
    s1 = "The book ended."
    s2 = "then he wrote a sequel quietly."
    return f"{s1} {s2}", [s1, s2]


def t_address_lowercase(rng):
    """Address ending in St. followed by a lowercase noun/verb (no break)."""
    addr = pick(rng, ADDRESS_ST)
    cont = pick(rng, [
        "with the family settled in.",
        "before the storm arrived.",
        "while the renovations continued.",
        "during the entire decade.",
        "after years of saving.",
    ])
    s1 = f"She lived at {addr} {cont}"
    return s1, [s1]


def t_paired_titles_no_break(rng):
    """Paired titles: 'Mr. and Mrs. Surname verbed.' — no break inside."""
    a = pick(rng, ["Mr.", "Dr."])
    b = pick(rng, ["Mrs.", "Ms.", "Dr."])
    surname = pick(rng, ["Brennan", "Choudhury", "Engstrom", "Hartmann", "Klein", "Petrov"])
    verb = pick(rng, ["arrived early.", "left after dinner.", "celebrated quietly.",
                       "attended the gala.", "thanked the host."])
    s1 = f"{a} and {b} {surname} {verb}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_two_titles_separate_subjects(rng):
    """Two titles as subjects: 'Prof. Surname and Dr. Other verbed.'"""
    a = pick(rng, TITLE_PROF + TITLE_DR)
    b = pick(rng, TITLE_DR + TITLE_MR)
    while a == b:
        b = pick(rng, TITLE_DR + TITLE_MR)
    verb = pick(rng, ["published a paper.", "agreed on the next steps.",
                       "co-authored the brief.", "appeared at the hearing."])
    s1 = f"{a} and {b} {verb}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_decimal_then_sent_end(rng):
    """Decimal-then-sentence-end pattern: 'Section 5.6.7 applies. Followup.'"""
    sec = pick(rng, SECTION_REFS)
    verb = pick(rng, ["applies.", "is satisfied.", "was amended.",
                      "remained in force.", "was struck down."])
    s1 = f"{sec} {verb}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_decimal_compare(rng):
    """'Version X.Y.Z. ran out. Version X.Y.W is in beta.'"""
    a = pick(rng, ["Version 4.5.6", "Version 7.8.9", "Version 1.2.3"])
    b = pick(rng, ["Version 4.5.7", "Version 7.8.10", "Version 1.2.4"])
    s1 = f"{a} is current."
    s2 = f"{b} is in beta."
    return f"{s1} {s2}", [s1, s2]


def t_long_title_list(rng):
    """A list of titled people as subject — no internal breaks."""
    titles = rng.sample(TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_MS, 3)
    verb = pick(rng, ["attended the meeting.", "signed the document.",
                      "arrived together.", "filed the appeal."])
    s1 = f"{titles[0]}, {titles[1]}, and {titles[2]} {verb}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_excl_break(rng):
    """Exclamation followed by sentence (the common pattern)."""
    excls = ["Stop right there!", "What a day!", "How extraordinary!", "Get out!"]
    s1 = pick(rng, excls)
    s2 = pick(rng, SENT_FOLLOWUPS + [
        "I'm exhausted.", "She turned away.", "He stayed silent.",
    ])
    return f"{s1} {s2}", [s1, s2]


def t_qmark_break(rng):
    """Question followed by sentence."""
    questions = ["Did he go?", "Where is the file?", "What time is it?"]
    s1 = pick(rng, questions)
    s2 = pick(rng, [
        "I am here.", "Yes, indeed.", "He nodded silently.",
        "The answer was clear.", "She refused to say.",
    ])
    return f"{s1} {s2}", [s1, s2]


def t_long_abbrev_then_break(rng):
    """Long sentence with multiple abbreviations followed by real sentence end."""
    a = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS)
    b = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS)
    while a == b:
        b = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS)
    c = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS)
    while c in (a, b):
        c = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS)
    time = pick(rng, TIME_AM + TIME_PM)
    day = pick(rng, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    s1 = f"{a} and {b} met with {c} at {time} on {day}."
    s2 = pick(rng, SENT_FOLLOWUPS + [
        "They discussed the case.",
        "The discussion lasted hours.",
        "Coffee was served afterwards.",
    ])
    return f"{s1} {s2}", [s1, s2]


def t_abbrev_at_time_break(rng):
    """Title arrived at TIME. Real break."""
    title = pick(rng, TITLE_DR + TITLE_MR + TITLE_MRS + TITLE_PROF)
    time = pick(rng, TIME_AM + TIME_PM)
    s1 = f"{title} arrived at {time}."
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_corp_then_break(rng):
    """'X Inc. announced earnings. Followup.'"""
    corp = pick(rng, CORP)
    verb = pick(rng, [
        "announced earnings.", "filed for an IPO.", "released the report.",
        "completed the merger.", "declared bankruptcy.",
    ])
    s1 = f"{corp} {verb}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]


def t_st_named_then_break(rng):
    """'St. Place verbed. Followup.'"""
    place = pick(rng, PLACE_ST_NAMED)
    verb = pick(rng, [
        "hosted the conference.", "celebrated its anniversary.",
        "drew large crowds.", "received the dignitaries.",
    ])
    s1 = f"{place} {verb}"
    s2 = pick(rng, SENT_FOLLOWUPS)
    return f"{s1} {s2}", [s1, s2]



TEMPLATES = [
    t_title_subject, t_title_subject, t_title_subject,
    t_two_titles, t_two_titles,
    t_title_at_time, t_title_at_time,
    t_title_no_break, t_title_no_break, t_title_no_break,
    t_corp_inline, t_corp_inline,
    t_corp_no_break, t_corp_no_break,
    t_corp_end,
    t_st_named_place, t_st_named_place,
    t_st_named_no_break,
    t_address, t_address,
    t_us_acronym, t_us_acronym,
    t_acronym_inline,
    t_acronym_pair,
    t_citation, t_citation,
    t_case_v, t_case_v,
    t_cf_citation,
    t_id_citation,
    t_eg_inline,
    t_etc_inline, t_etc_inline,
    t_etc_break,
    t_decimal_inline, t_decimal_inline,
    t_section_ref, t_section_ref,
    t_price,
    t_time_break,
    t_url, t_url_no_break,
    t_month_abbrev,
    t_day_abbrev,
    t_jr_sr, t_jr_sr_no_break,
    t_degree_end,
    t_long_abbrev_mix,
    # Recall-recovery cases (questions, exclamations, quotes, parens, etc.)
    t_question, t_question, t_question,
    t_exclamation, t_exclamation,
    t_qmark_break, t_qmark_break, t_qmark_break,
    t_excl_break, t_excl_break, t_excl_break,
    t_quoted_dialogue, t_quoted_dialogue,
    t_quoted_end, t_quoted_end,
    t_paren_end, t_paren_end,
    t_short_consec, t_short_consec,
    t_lowercase_continuation,
    # v3 specific failure-targeted templates
    t_address_lowercase, t_address_lowercase,
    t_paired_titles_no_break, t_paired_titles_no_break,
    t_two_titles_separate_subjects, t_two_titles_separate_subjects,
    t_decimal_then_sent_end, t_decimal_then_sent_end, t_decimal_then_sent_end,
    t_decimal_compare, t_decimal_compare,
    t_long_title_list, t_long_title_list,
    # v4: real-end-after-abbrev-laden contexts
    t_long_abbrev_then_break, t_long_abbrev_then_break, t_long_abbrev_then_break,
    t_abbrev_at_time_break, t_abbrev_at_time_break,
    t_corp_then_break, t_corp_then_break,
    t_st_named_then_break,
    # heavy clean text presence so the model doesn't lose recall on plain sentences
    t_clean, t_clean, t_clean, t_clean, t_clean, t_clean, t_clean, t_clean,
]


def build_record(rng: random.Random) -> dict:
    # Each record: 1-2 templates, joined by space, generates a 1-4 sentence chunk.
    n_templates = rng.choices([1, 2], weights=[0.7, 0.3])[0]
    parts = [pick(rng, TEMPLATES)(rng) for _ in range(n_templates)]
    text_pieces = [p[0] for p in parts]
    text = " ".join(text_pieces)

    spans = []
    cursor = 0
    for piece_idx, (piece_text, piece_sents) in enumerate(parts):
        if piece_idx > 0:
            cursor += 1  # the space between pieces
        local_cursor = 0
        for sent in piece_sents:
            idx_local = piece_text.index(sent, local_cursor)
            start = cursor + idx_local
            end = start + len(sent.encode("utf-8"))
            spans.append({
                "label": "sentence",
                "start": start,
                "end": end,
                "char_start": start,
                "char_end": end,
                "right_open": False,
            })
            local_cursor = idx_local + len(sent)
        cursor += len(piece_text)

    return {"text": text, "spans": spans}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    n_written = 0
    with out.open("w") as f:
        for _ in range(args.n):
            rec = build_record(rng)
            # Validate spans: each sentence span text must match the substring
            ok = True
            for sp in rec["spans"]:
                if rec["text"][sp["start"]:sp["end"]] not in rec["text"]:
                    ok = False
                    break
            if not ok:
                continue
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"wrote {n_written} records to {out}")


if __name__ == "__main__":
    main()
