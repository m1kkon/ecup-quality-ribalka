"""Text builders for the БАД classifier.

Two things matter here:
  1. descriptions are raw marketplace HTML up to 6000 chars — tags are pure token waste;
  2. the decisive phrase ("...является БАД", "не является БАД", "спортивное питание")
     sits at a median offset of ~400 chars but past 1800 chars for 9% of rows, i.e.
     outside any 512-token window. Head-truncation silently drops that evidence.

So we clean the HTML and then build the model input out of marker-centred windows
instead of a prefix.
"""
from __future__ import annotations

import html
import re

# ---------------------------------------------------------------- cleaning
_TAG = re.compile(r"<[^>]{0,200}>")
_LI = re.compile(r"</li\s*>|</p\s*>|<br\s*/?>", re.I)
_WS = re.compile(r"[ \t\xa0]+")
_NL = re.compile(r"\n{2,}")


def clean(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = _LI.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = html.unescape(s)
    s = s.replace("\r", "\n")
    s = _WS.sub(" ", s)
    s = _NL.sub("\n", s)
    return s.strip()


# ---------------------------------------------------------------- markers
NEG_PAT = re.compile(
    r"(?:не\s+явля\w*\s+(?:бад\b|биологически\s+активн|биодобавк|пищев\w*\s+добавк)"
    r"|не\s+(?:бад|является\s+бад)"
    r"|бад\w*\s+не\s+явля\w*"
    r"|не\s+относится\s+к\s+бад"
    r"|не\s+бад\b)",
    re.I,
)
# "бад" + Russian case endings only — so бадминтон / бадьян / бадяга do not match.
POS_PAT = re.compile(
    r"(?:\bбад(?:ами|ах|ов|ам|ом|ы|а|у|е|)\b"
    r"|биологически\s+активн\w*\s+(?:пищев\w*\s+)?добавк\w*"
    r"|dietary\s+supplement"
    r"|биодобавк\w*"
    r"|\bб\.\s?а\.\s?д\.?)",
    re.I,
)
CTX_PAT = re.compile(
    r"(?:спортивн\w*\s+питани\w*|\bbcaa\b|л[-\s]?карнитин|l[-\s]?carnitine|протеин\w*"
    r"|изолят|гейнер|креатин|аминокислот\w*|сывороточн\w*|предтрен"
    r"|свидетельств\w*\s+о\s+государствен\w*\s+регистрац|\bсгр\b|\bту\s?\d"
    r"|не\s+явля\w*\s+лекарствен\w*|пищев\w*\s+добавк\w*|food\s+supplement"
    r"|\bбиологически\s+активн\w*"
    r"|противопоказан|индивидуальн\w*\s+непереносим|суточн\w*\s+(?:доз|потребл)"
    r"|перед\s+примен\w*\s+прокон|беременн\w*\s+и\s+кормя)",
    re.I,
)


def _windows(text: str, pat: re.Pattern, left: int, right: int, limit: int):
    spans = []
    for m in pat.finditer(text):
        a, b = max(0, m.start() - left), min(len(text), m.end() + right)
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
        if len(spans) >= limit:
            break
    return spans


def evidence(
    name: str,
    desc: str,
    head: int = 400,
    radius: int = 140,
    budget: int = 1600,
    limit: int = 12,
    do_clean: bool = True,
) -> str:
    """name + head-of-description + every marker-centred window, in document order."""
    desc = clean(desc) if do_clean else (desc or "")
    name = clean(name) if do_clean else (name or "")
    spans = [(0, min(head, len(desc)))] if head else []
    # a bit more left context than right: negations sit before the marker
    for pat in (NEG_PAT, POS_PAT, CTX_PAT):
        spans += _windows(desc, pat, int(radius * 1.3), radius, limit)
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 8:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    used, chunks = 0, []
    for a, b in merged:
        piece = desc[a:b]
        if used + len(piece) > budget:
            piece = piece[: max(0, budget - used)]
        if piece:
            chunks.append(piece)
            used += len(piece)
        if used >= budget:
            break
    return f"{name} | " + " … ".join(chunks)


def head_only(name: str, desc: str, budget: int = 2000, do_clean: bool = True) -> str:
    d = clean(desc) if do_clean else (desc or "")
    n = clean(name) if do_clean else (name or "")
    return f"{n} | {d[:budget]}"


def full(name: str, desc: str, do_clean: bool = True) -> str:
    d = clean(desc) if do_clean else (desc or "")
    n = clean(name) if do_clean else (name or "")
    return f"{n} | {d}"


def raw_head(name: str, desc: str, budget: int = 2000) -> str:
    """No HTML cleaning — the control condition."""
    return f"{name or ''} | {(desc or '')[:budget]}"


# ---------------------------------------------------------------- sentence view
_SENT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def sentences(desc: str, do_clean: bool = True):
    d = clean(desc) if do_clean else (desc or "")
    return [x.strip() for x in _SENT.split(d) if x.strip()]


def evidence_sent(
    name: str,
    desc: str,
    n_head: int = 1,
    neighbours: int = 0,
    budget: int = 1600,
    limit: int = 14,
    do_clean: bool = True,
) -> str:
    """Same idea as evidence() but the unit is a sentence, not a char window —
    a transformer gets whole clauses instead of clipped fragments."""
    sents = sentences(desc, do_clean)
    n = clean(name) if do_clean else (name or "")
    if not sents:
        return f"{n} | "
    keep = set(range(min(n_head, len(sents))))
    hits = 0
    for i, s in enumerate(sents):
        if NEG_PAT.search(s) or POS_PAT.search(s) or CTX_PAT.search(s):
            for j in range(max(0, i - neighbours), min(len(sents), i + neighbours + 1)):
                keep.add(j)
            hits += 1
            if hits >= limit:
                break
    used, chunks = 0, []
    for i in sorted(keep):
        piece = sents[i]
        if used + len(piece) > budget:
            piece = piece[: max(0, budget - used)]
        if piece:
            chunks.append(piece); used += len(piece)
        if used >= budget:
            break
    return f"{n} | " + " … ".join(chunks)
