"""Silver-label real-world text via nupunkt + a sanity filter.

The filter suppresses nupunkt's known false-positive patterns
(Roman-numeral section markers, ``Sec.`` followed by section numbers,
``U.S.`` mid-name, decimal contexts, single-digit enumerated items,
common abbreviations). The output is a JSONL training corpus matching
``train_sentence_burn``'s input schema.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import nupunkt


KNOWN_ABBREVS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Hon.", "Jr.", "Sr.",
    "Inc.", "Ltd.", "Co.", "Corp.", "Capt.", "Cmdr.", "Rev.",
    "Sec.", "No.", "Vol.", "Pub.",
    "U.S.", "U.K.", "E.U.", "U.S.C.", "C.F.R.",
    "Ph.D.", "M.D.", "M.S.", "B.A.", "B.S.", "LL.M.",
    "et al.", "e.g.", "i.e.", "vs.", "cf.", "id.", "et seq.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.",
    "Sept.", "Oct.", "Nov.", "Dec.",
    "a.m.", "p.m.",
    "St.",
}
ROMAN = {f"{r}." for r in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                            "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]}


def is_obvious_fp(text: str, end_char: int) -> bool:
    """Reject nupunkt boundary if it's at a known false-positive shape."""
    # Look at the token preceding the boundary.
    cursor = end_char
    # Skip trailing whitespace.
    while cursor > 0 and text[cursor - 1] in " \t\n\r":
        cursor -= 1
    if cursor == 0:
        return True
    # Walk back over alphanumeric+dot.
    start = cursor
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    tok = text[start:cursor]

    if tok in ROMAN:
        return True
    if re.fullmatch(r"\d{1,4}\.", tok):
        # But allow if it looks like end-of-year-date: tok ends with 4-digit year and previous chars suggest a date.
        # Easier: allow if token is exactly 4 digits (year) and preceded by ", " or " " in a date context.
        if re.fullmatch(r"\d{4}\.", tok):
            # Check prev 12 chars for a month name or comma — if so, it's a date.
            ctx = text[max(0, start - 15):start]
            if re.search(r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                         r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
                         r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d+,\s+$", ctx):
                return False
        return True
    if tok in KNOWN_ABBREVS:
        return True
    if re.fullmatch(r"(?:[A-Z]\.){2,}", tok):
        # X.Y. or X.Y.Z. style multi-period acronym
        return True

    # Decimal: digit before period, digit after.
    if cursor > 1 and text[cursor - 1] == "." and text[cursor - 2].isdigit():
        nxt = text[cursor:cursor + 1]
        if nxt and nxt.isdigit():
            return True

    # Next char checks.
    after = text[cursor:cursor + 5].lstrip()
    if not after:
        return True
    if after[0] in "[_*":
        return True
    if not (after[0].isupper() or after[0].isdigit() or after[0] in '"\'“‘'):
        return True

    return False


def label_text(text: str) -> dict | None:
    """Run nupunkt + filter, return JSONL record with sentence spans (char offsets)."""
    char_spans = list(nupunkt.sent_spans(text))
    if len(char_spans) < 2:
        return None
    # Filter spans whose end is at an obvious FP shape.
    kept = []
    prev_end = 0
    for s, e in char_spans:
        # Trim trailing whitespace from the span end.
        e_trim = e
        while e_trim > s and text[e_trim - 1] in " \t\n\r":
            e_trim -= 1
        s_trim = s
        while s_trim < e_trim and text[s_trim] in " \t\n\r":
            s_trim += 1
        if e_trim <= s_trim:
            continue
        # If this is not the last span and the boundary at e_trim is a FP, merge with next.
        kept.append((s_trim, e_trim))

    # Walk kept and merge across FP boundaries.
    if not kept:
        return None
    merged = [list(kept[0])]
    for s, e in kept[1:]:
        # Test boundary at the previous merged span's end.
        prev_s, prev_e = merged[-1]
        if is_obvious_fp(text, prev_e):
            # Merge: the boundary at prev_e is wrong, so this span continues the previous.
            merged[-1][1] = e
        else:
            merged.append([s, e])

    if len(merged) < 1:
        return None

    spans = []
    for s, e in merged:
        # Convert char offsets to byte offsets.
        s_byte = len(text[:s].encode("utf-8"))
        e_byte = len(text[:e].encode("utf-8"))
        spans.append({
            "label": "sentence",
            "start": s_byte,
            "end": e_byte,
            "char_start": s,
            "char_end": e,
            "right_open": False,
        })
    return {"text": text, "spans": spans}


# Source 1: Federal Register API — pull a wider range of recent docs.
import urllib.parse

def fetch_fedreg(n=30):
    url = "https://www.federalregister.gov/api/v1/documents.json?" + urllib.parse.urlencode({
        "per_page": str(n),
        "order": "newest",
        "fields[]": "raw_text_url",
    }, doseq=True)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"  fedreg API failed: {e}")
        return []
    texts = []
    for r in data.get("results", []):
        u = r.get("raw_text_url")
        if not u:
            continue
        try:
            with urllib.request.urlopen(u, timeout=15) as resp:
                raw = resp.read().decode("utf-8", "ignore")
        except Exception:
            continue
        m = re.search(r"<pre>(.*?)</pre>", raw, re.DOTALL)
        body = m.group(1) if m else raw
        body = re.sub(r"<[^>]+>", "", body).strip()
        if len(body) > 1500:
            texts.append(body)
    return texts


# Source 2: open-access PMC articles via search.
def fetch_pmc(n=20):
    qurl = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({
        "query": "OPEN_ACCESS:Y AND HAS_FT:Y AND SRC:MED",
        "format": "json",
        "sort": "FIRST_PDATE_D desc",
        "pageSize": str(n + 5),
    })
    try:
        with urllib.request.urlopen(qurl, timeout=15) as resp:
            d = json.load(resp)
    except Exception:
        return []
    pmcids = []
    for r in d.get("resultList", {}).get("result", []):
        pmcid = r.get("pmcid")
        if pmcid and pmcid.startswith("PMC"):
            pmcids.append(pmcid[3:])
    texts = []
    for pid in pmcids[:n]:
        url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_JSON/PMC{pid}/unicode"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = resp.read().decode("utf-8", "ignore")
            data = json.loads(raw)
            chunks = []
            for doc in data[0].get("documents", []):
                for p in doc.get("passages", []):
                    t = p.get("text", "")
                    if t:
                        chunks.append(t)
            text = "\n\n".join(chunks)
            if len(text) > 1500:
                texts.append(text[:50000])
        except Exception:
            continue
    return texts


# Source 3: a couple Gutenberg books (we already have Pride locally).
def fetch_gutenberg(gutenberg_ids):
    out = []
    for gid in gutenberg_ids:
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = resp.read().decode("utf-8", "ignore")
            s = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
            e = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", raw)
            if s and e:
                body = raw[s.end():e.start()].strip()
                # Skip front matter — start from middle, take 50k.
                if len(body) > 100000:
                    out.append(body[40000:90000])
                elif len(body) > 50000:
                    out.append(body[20000:70000])
        except Exception:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="charstreamer-silver-corpus")
    parser.add_argument("--n-fedreg", type=int, default=40,
                        help="Federal Register documents to pull (latest first).")
    parser.add_argument("--n-pmc", type=int, default=30,
                        help="PubMed Central open-access articles to pull (latest first).")
    parser.add_argument("--gutenberg-id", type=int, action="append",
                        help="Gutenberg book IDs to include (repeatable). "
                             "Default: 1342, 98, 84, 1661.")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Max chars per output JSONL record (default 10000).")
    parser.add_argument("--min-chunk-bytes", type=int, default=500,
                        help="Skip output chunks smaller than this.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSONL path. Parent directory will be created.")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gutenberg_ids = args.gutenberg_id or [1342, 98, 84, 1661]

    print("Pulling Federal Register...")
    fedreg_texts = fetch_fedreg(n=args.n_fedreg)
    print(f"  got {len(fedreg_texts)} fedreg documents")

    print("Pulling PMC...")
    pmc_texts = fetch_pmc(n=args.n_pmc)
    print(f"  got {len(pmc_texts)} pmc articles")

    print("Pulling Gutenberg...")
    gutenberg_texts = fetch_gutenberg(gutenberg_ids)
    print(f"  got {len(gutenberg_texts)} gutenberg slices")

    all_texts = fedreg_texts + pmc_texts + gutenberg_texts
    print(f"\nTotal source texts: {len(all_texts)}")

    print("Running nupunkt + filter...")
    n_records = 0
    with args.out.open("w") as f:
        for text in all_texts:
            for i in range(0, len(text), args.chunk_size):
                chunk = text[i:i + args.chunk_size]
                if len(chunk) < args.min_chunk_bytes:
                    continue
                rec = label_text(chunk)
                if rec is None or len(rec["spans"]) < 2:
                    continue
                f.write(json.dumps(rec) + "\n")
                n_records += 1

    print(f"wrote {n_records} silver-labeled records to {args.out}")
    print(f"  size: {args.out.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
