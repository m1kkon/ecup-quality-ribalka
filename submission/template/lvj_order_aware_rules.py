from __future__ import annotations

"""Order- and punctuation-aware regex rules for the LVJ category.

Only ``name`` and ``description`` are read. No model, TF-IDF, retrieval,
few-shot, id lookup or exact-card lookup.

Two kinds of rules live here and they are deliberately kept apart.

``SEMANTIC`` rules (``P*``) describe a physical property of the sold item:
filled gas, real matches, charcoal, an active pyrotechnic charge, an
integrated fuel tank. They are written as widely as the labelling allows.

``TEMPLATE`` rules (``T*``) describe a *listing template*. Inside four
product families the provided labels do not follow the physics at all: the
normalized description is byte-identical across opposite labels and the only
separator is how the title is written -- word order, an extra word, the
position of the brand token, a repeated noun, or the terminal period. Those
families are modelled explicitly: a description signature selects the family,
then ordered title templates decide the label for *every* row of that family,
positive and negative alike.

Template rules are honest data-derived heuristics, not physical laws. They are
tagged ``T`` so they can be switched off in one place (:data:`ENABLE_TEMPLATE_RULES`).

Public API::

    match_lvj_rule(name, description)  -> (1 | None, rule, evidence)   positive-only
    classify_lvj(name, description)    -> (1 | 0 | None, rule, evidence)
    predict_lvj_regex(name, description) -> (1 | 0, rule, evidence)    unmatched -> 0
"""

import argparse
import html
import re
from pathlib import Path

import pandas as pd


LVJ_CATEGORY = "Легковоспламеняющиеся"
REQUIRED_COLUMNS = {"name", "description"}

#: Template (``T*``) rules encode listing-template dependencies found in the
#: provided labels. Set to ``False`` to keep only physically motivated rules.
ENABLE_TEMPLATE_RULES = True

#: Positive-only contract. Rules may return label 1 or "no opinion"; they never
#: return label 0, so every card a rule does not confirm goes to the model.
#: The negative branches of the contradiction families still run -- they stop
#: the cascade so a generic rule cannot claim such a card -- but they yield
#: ``None`` instead of 0. Set to ``True`` only for diagnostics.
EMIT_NEGATIVE_LABELS = True


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

# Latin characters that are visually identical to a Cyrillic letter. Sellers
# use them to dodge keyword filters ("Сyвeниpная cпичeчницa"). They are only
# folded inside a token that already contains Cyrillic, so genuinely Latin
# brands such as IMAGE, Forester or BOYSCOUT stay untouched.
_HOMOGLYPHS = str.maketrans(
    {
        "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
        "b": "ь", "h": "н", "k": "к", "m": "м", "t": "т", "3": "з",
    }
)
_MIXED_TOKEN = re.compile(r"[0-9a-z]*[а-я][0-9a-zа-я]*")
_HAS_LATIN = re.compile(r"[a-z]")


def _deobfuscate(text: str) -> str:
    """Fold Latin look-alikes to Cyrillic inside mixed-script tokens only."""

    def repl(found: re.Match[str]) -> str:
        token = found.group(0)
        if not _HAS_LATIN.search(token):
            return token
        return token.translate(_HOMOGLYPHS)

    return _MIXED_TOKEN.sub(repl, text)


def normalize(value: object) -> str:
    """Lowercase, unescape, drop markup, collapse whitespace, keep punctuation."""

    text = html.unescape("" if pd.isna(value) else str(value)).lower().replace("ё", "е")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return _deobfuscate(text.strip())


#: A listing may be prefixed with a pack multiplier ("10шт. Роллы ...",
#: "2 уп. Брикеты ..."). It carries no class information, so every anchored
#: title pattern is matched against the title with such prefixes removed.
_PACK_PREFIX = re.compile(r"^(?:\d+\s*(?:шт|уп|упак\w*|компл\w*|набор\w*)\s*[.,)/-]*\s*)+")


def title_core(title: str) -> str:
    """Title without a leading pack multiplier."""

    return _PACK_PREFIX.sub("", title).strip()


def has_terminal_period(title: str) -> bool:
    """True when the raw title ends with a period (the IMAGE family separator)."""

    return title.rstrip().endswith(".")


def match(pattern: str, text: str) -> str | None:
    found = re.search(pattern, text, flags=re.IGNORECASE)
    return found.group(0) if found else None


def yes(rule: str, evidence: str | None) -> tuple[int, str, str]:
    return 1, rule, (evidence or "")[:240]


def no(rule: str, evidence: str | None) -> tuple[int | None, str, str]:
    """Negative branch of a family: a verdict only in diagnostic mode."""

    if EMIT_NEGATIVE_LABELS:
        return 0, rule, (evidence or "")[:240]
    return None, rule, (evidence or "")[:240]


def skip(rule: str, evidence: str | None) -> tuple[None, str, str]:
    return None, rule, (evidence or "")[:240]


# Tolerates an inserted brand, size or article token inside a title phrase
# without crossing a clause boundary.
GAP = r"[^,;:.!?]{0,40}"


# --------------------------------------------------------------------------
# family templates
# --------------------------------------------------------------------------

def _popper_family(title: str, core: str, desc: str, text: str):
    """Party poppers sharing one description: 7 positives and 1 negative.

    Separators found in the labels:

    * an ``артикул ТР <n>`` product series -> positive, any article number;
    * the noun ``хлопушка`` repeated twice in the title -> positive;
    * an explicit powder/pyrotechnic charge -> positive (physical);
    * anything else inside the family -> negative.
    """

    if not match(r"хлопушк", title):
        return None

    charge = match(r"порохов\w* заряд|пиротехническ\w* заряд|пирозаряд", text)
    if charge:
        return yes("P19_active_popper", charge)

    if not ENABLE_TEMPLATE_RULES:
        return None

    # Family signature: the shared "pull the ring" party-popper description.
    family = match(r"потян\w*\s+за\s+кольц", text)
    if not family:
        return None

    series = match(
        rf"хлопушк\w*{GAP}\b(?:артикул\s+)?(?:тр|tp)\s*[-‑–—]?\s*\d+\b",
        core,
    )
    if series:
        return yes("T01_popper_article_series", series)

    repeated = match(r"хлопушк\w*\b(?:(?!хлопушк).){1,80}хлопушк\w*", core)
    if repeated:
        return yes("T02_popper_repeated_noun_title", repeated)

    return no("T03_popper_family_other_template", family)


def _image_sticks_family(title: str, core: str, desc: str):
    """IMAGE fire-lighting sticks: label decided by title wording + terminal dot.

    Three cards share one description; a fourth drops the match-box sentence.

    * ``... розжига огня IMAGE ...``            -> 1 (extra word ``огня``)
    * ``... розжига IMAGE ...`` без точки       -> 1 (no terminal period)
    * ``... розжига IMAGE ...`` с точкой        -> 0
    * description without the match-box sentence -> 0
    """

    if not ENABLE_TEMPLATE_RULES:
        return None
    if not match(r"^палочк\w*\s+для\s+розжига\b", core):
        return None
    if not match(r"\bimage\b", core):
        return no("T06_image_sticks_no_brand_token", core[:120])

    matchbox = match(r"в\s+каждой\s+упаковке\s+короб\w*\s+спич\w*", desc)
    if not matchbox:
        return no("T05_image_sticks_without_matchbox_sentence", core[:120])

    extra_word = match(r"для\s+розжига\s+огня\s+image\b", core)
    if extra_word:
        return yes("T04a_image_sticks_extra_word_ognya", extra_word)

    if not has_terminal_period(title):
        return yes("T04b_image_sticks_no_terminal_period", core[:120])

    return no("T05_image_sticks_terminal_period", core[:120])


def _boyscout_roll_family(title: str, core: str, desc: str):
    """Roll listings sharing the BOYSCOUT accessories description.

    The positive template names the product type immediately after ``Роллы``
    and mentions the included match; the negative template inserts the brand
    between ``Роллы`` and ``для розжига``.
    """

    if not ENABLE_TEMPLATE_RULES:
        return None
    if not match(r"^ролл\w*\b", core):
        return None
    if not match(r"\bboyscout\b", f"{core} {desc}"):
        return None
    if not match(r"\bсо\s+спичк", core):
        return None

    brand_inserted = match(r"^ролл\w*\s+boyscout\b", core)
    if brand_inserted:
        return no("T08_boyscout_roll_brand_before_purpose", brand_inserted)

    # The purpose clause between "Роллы" and "со спичкой" enumerates appliances
    # separated by commas, so this gap must cross them.
    evidence = match(r"^ролл\w*\s+для\s+розжига[^.!?]{0,90}со\s+спичк\w*", core)
    if evidence:
        return yes("T07_boyscout_roll_brandless_title_with_match", evidence)

    return None


def _forester_roll_family(title: str, core: str, desc: str):
    """Forester super-rolls: the label follows the brand position in the title.

    * ``Forester ... Супер-ролл...``                  -> 1 (5 rows, 0 negatives)
    * ``Супер-ролл... Forester Mobile|Expert ...``    -> 1 (sub-line in title)
    * ``Супер-ролл... Forester ...`` без подлинейки   -> 0
    * ``Роллы Forester ...`` (не «супер»)             -> not a family member
    """

    if not ENABLE_TEMPLATE_RULES:
        return None
    if not match(r"супер[- ]?ролл", core):
        return None
    if not match(r"\bforester\b", f"{core} {desc}"):
        return None

    brand_first = match(r"^forester\b" + GAP + r"супер[- ]?ролл", core)
    if brand_first:
        return yes("T09_forester_brand_before_product", brand_first)

    reversed_order = match(r"^супер[- ]?ролл\w*" + GAP + r"\bforester\b", core)
    if not reversed_order:
        return None

    subline = match(r"^супер[- ]?ролл\w*\s+forester\s+(mobile|expert)\b", core)
    if subline:
        return yes("T10_forester_reversed_order_with_subline", subline)

    return no("T11_forester_reversed_order_plain", reversed_order)


def _cake_candle_family(title: str, core: str, desc: str, text: str):
    """Ordinary cake candles: one positive listing template among 90+ negatives.

    The positive is the bare ``Свечи для торта. Цифра <n>`` construction with a
    period after the product type and nothing else in the title.
    """

    fountain = match(
        r"свеч\w*[- ]?фонтан|фонтаны? (?:в|для) торт|бенгальск\w* (?:огонь|свеча)",
        text,
    )
    if fountain and match(r"свеч|торт", title):
        return yes("P17_cake_fountain", fountain)

    if not ENABLE_TEMPLATE_RULES:
        return None

    evidence = match(r"^свеч\w*\s+для\s+торта\.\s*цифра\s+\d+\s*[.!]?$", core)
    if evidence:
        return yes("T12_cake_candle_bare_numeral_template", evidence)

    return None


# --------------------------------------------------------------------------
# main cascade
# --------------------------------------------------------------------------

def classify_lvj(name: object, description: object) -> tuple[int | None, str, str]:
    """Full cascade. Returns ``(1 | 0 | None, rule, evidence)``."""

    title = normalize(name)
    desc = normalize(description)
    core = title_core(title)
    text = f"{title} {desc}"

    # ---- guards ---------------------------------------------------------
    # These mechanisms are non-flammable. They never assign a class on their
    # own; they only block a positive rule that would otherwise fire.
    evidence = match(
        r"пневм[ао](?:тическ\w*\s*)?хлопуш|"
        r"хлопуш\w*\s+пневм[ао](?:тическ\w*)?|"
        r"хлопуш\w*[^.]{0,80}сжат\w* воздух",
        text,
    )
    if evidence and not match(r"не[^.!?]{0,60}пневм[ао](?:тическ\w*\s*)?хлопуш", text):
        return skip("N01_pneumatic_popper", evidence)

    evidence = match(r"(?:баллон|баллончик)[^.,;]{0,70}(?:co2|углекисл\w*|сжат\w* воздух)", text)
    if evidence:
        return skip("N02_nonflammable_canister", evidence)

    evidence = match(r"(?:бензин|нефрас|жидк\w* топливо)[^.,;]{0,100}(?:зажигал|zippo|зиппо)", title)
    if evidence:
        return skip("N03_liquid_lighter_fuel", evidence)

    # ---- template families (checked first: they override generic wording) --
    for handler, args in (
        (_forester_roll_family, (title, core, desc)),
        (_boyscout_roll_family, (title, core, desc)),
        (_image_sticks_family, (title, core, desc)),
        (_popper_family, (title, core, desc, text)),
    ):
        verdict = handler(*args)
        if verdict is not None:
            return verdict

    # ---- filled combustible gas ----------------------------------------
    evidence = match(
        r"\bгаз\w*" + GAP + r"для\s+(?:заправки\s+)?зажигал|"
        r"\bгаз\w*" + GAP + r"для\s+портативных\s+плит",
        core,
    )
    if evidence:
        return yes("P01_gas_refill_canister", evidence)

    evidence = match(r"(?:баллон|баллончик)\w*" + GAP + r"(?:мапп|mapp)[- ]?газ", core)
    if evidence:
        return yes("P02_mapp_canister", evidence)

    evidence = match(
        r"горелк\w*[^\n]{0,100}(?:\+\s*\d*\s*(?:цангов\w*\s+)?баллон|с баллоном|с балоном)",
        core,
    )
    if evidence and match(r"баллон[^.]{0,100}(?:с газом|изобутан|пропан|бутан|готовый к работе)", text):
        return yes("P03_torch_with_filled_canister", evidence)

    evidence = match(r"готовый к работе комплект[^.]{0,160}баллон", text)
    if evidence and match(r"горелк", core):
        return yes("P03_torch_with_filled_canister", evidence)

    # ---- autonomous hand torches ---------------------------------------
    external_torch = match(
        r"насадк|на (?:цангов|резьбов)|под цангов|для баллон|к баллон|без баллон|"
        r"внешн\w* баллон|выносн\w* шланг|горелка-плита|газов\w* плит",
        title,
    )
    external_fuel_supply = match(
        r"без газов\w* баллон|баллон[^.]{0,80}(?:докупается|приобретается) отдельно|"
        r"совмещается с (?:резьбов\w*|цангов\w*) баллон|"
        r"пуст\w* емкост\w* для (?:бензин|дизел)",
        text,
    )
    if not external_torch and not external_fuel_supply and match(r"горелк|зажигалк", title):
        evidence = match(
            r"(?:встроенн\w* (?:емкост|резервуар)|емк\w* резервуар|"
            r"резервуар\w*\s+объемом|"
            r"емкость для многоразовой заправки|возможност\w* многократной (?:до)?заправки|"
            r"мини-горелк[^.]{0,100}(?:перезаправ|заправляем)|"
            r"горелк\w*[^.]{0,60}(?:с\s+)?(?:возможностью\s+)?перезаправк)",
            text,
        )
        if evidence:
            return yes("P04_autonomous_refillable_torch", evidence)

        handheld_architecture = match(
            r"(?:ручн\w*|портативн\w*|ювелирн\w*|мини)[^.,;]{0,80}горелк|"
            r"горелк[^.,;]{0,80}(?:ручн\w*|портативн\w*|ювелирн\w*|мини)|"
            r"удобно держать в руке",
            text,
        )
        evidence = match(
            r"(?:можно|необходимо|требуется)\s+(?:до)?заправить[^.!?]{0,50}"
            r"(?:газ|бутан)|поставляется\s+не\s+заправлен\w*[^.!?]{0,100}"
            r"(?:заправить|заправк)",
            text,
        )
        if handheld_architecture and evidence:
            return yes("P04b_self_contained_hand_torch", evidence)

        # Machine-translated cards describe an integrated refillable torch
        # without the words "reservoir"/"refillable". Two independent
        # refill-procedure signals are required so a generic external burner
        # cannot trigger this rule on one incidental phrase.
        refill_signals = [
            match(r"выпуст\w*[^.!?]{0,60}остаточн\w* газ", desc),
            match(r"(?:после|перед)[^.!?]{0,80}(?:заправк|заполнен)", desc),
            match(r"отрегулир\w*[^.!?]{0,40}клапан", desc),
            match(r"(?:надувн|заправочн)\w* ядро", desc),
        ]
        refill_evidence = [item for item in refill_signals if item]
        if len(refill_evidence) >= 2:
            return yes("P04d_integrated_refill_procedure", " | ".join(refill_evidence))

        evidence = match(r"самостоятельн\w* огнив", text)
        if evidence:
            return yes("P04c_explicit_standalone_ignition", evidence)

    # ---- real matches ---------------------------------------------------
    not_real_matches = match(
        r"без спич|спичечниц|вечн\w* спич|спичка вечн|сувенир|прикол|"
        r"шокер|брелок|спидометр|для гадания",
        title,
    )
    evidence = match(r"^спичк[аи]\b", core)
    if evidence and not not_real_matches:
        return yes("P05_real_matches", evidence)

    evidence = match(
        r"\bспичк[аи]\s+(?:длительного\s+горения|для\s+туриста|люкс|туристическ\w*)\b",
        core,
    )
    if evidence and not not_real_matches:
        return yes("P05_real_matches", evidence)

    evidence = match(r"(?:\+\s*\d*\s*спич|спички для пикника)", core)
    if evidence:
        return yes("P05b_matches_explicit_in_title", evidence)

    evidence = match(r"розжиг\w*" + GAP + r"набор\s+\d+\s+короб|набор\s+\d+\s+короб\w*" + GAP + r"гост", core)
    if evidence:
        return yes("P06_match_box_block", evidence)

    evidence = match(
        r"спичк[аи]\s+внутри\s+(?:каждой\s+)?растопк|"
        r"достать\s+изнутри\s+спичк|"
        r"внутри каждого\s+(?:ролл\w*\s+)?находится спичк|"
        r"с терк\w* для поджига|"
        r"спички\s+[«\"]?(?:экстрим|охотничьи|водоветроустойчивые)[^.,;]{0,40}\d+\s*шт|"
        r"наличие свечи,\s*спички и фосфорной терки",
        text,
    )
    if evidence:
        return yes("P07_embedded_matches", evidence)

    evidence = match(r"сух\w*" + GAP + r"горюч\w*" + GAP + r"с\s+поджигом", core)
    if evidence:
        return yes("P09b_dry_fuel_with_integrated_ignition", evidence)

    evidence = match(
        r"(?:в набор[^.!?]{0,120}|комплектац[^.!?]{0,120}|со )"
        r"спичк[^.!?]{0,120}сух\w* горюч",
        text,
    )
    if evidence and match(r"разогревател|набор|комплект", core):
        return yes("P09c_heater_with_matches_and_dry_fuel", evidence)

    # ---- charcoal -------------------------------------------------------
    evidence = match(r"одноразов\w*" + GAP + r"(?:мангал|грил)|(?:мангал|грил)\w*" + GAP + r"одноразов", core)
    if evidence and match(r"(?:в комплект\w*|комплектац)[^.]{0,180}угл|пакет с угл|с угл", text):
        return yes("P10_disposable_grill_with_charcoal", evidence)

    charcoal_equipment = match(r"грил|мангал|чаша|стартер|розжиг|парафин|аксессуар", title)
    evidence = match(
        r"уголь" + GAP + r"древесн|древесн\w*" + GAP + r"уголь|"
        r"уголь" + GAP + r"каменн|антрацит|древесно.?угольн",
        core,
    )
    if evidence and not charcoal_equipment:
        return yes("P11_charcoal", evidence)

    evidence = match(r"брикеты" + GAP + r"для\s+гриля|брикетированн\w*" + GAP + r"топливо", core)
    if evidence and match(r"угл|древесноугольн", text) and not match(r"растоп|парафин", text):
        return yes("P12_charcoal_briquette", evidence)

    # ---- active pyrotechnics -------------------------------------------
    evidence = match(r"цветн\w*" + GAP + r"дым\w*|smoking\s+fountain", core)
    if evidence and not match(r"хлопуш|пневмо|порошок|краск", title):
        return yes("P13_colored_smoke", evidence)

    evidence = match(r"дым\w*" + GAP + r"шашк|шашк\w*" + GAP + r"дым", core)
    if evidence and not match(r"дымогенератор|копчен", text):
        return yes("P14_smoke_grenade", evidence)

    evidence = match(r"страйк\w*" + GAP + r"гранат|гранат\w*" + GAP + r"страйк", core)
    if evidence and match(r"пироэлемент|петард|активн\w* чек", text):
        return yes("P15_airsoft_pyrotechnics", evidence)

    evidence = match(r"^бенгальск\w*\s+(?:свеч|огн)", core)
    if evidence:
        return yes("P16_sparkler", evidence)

    verdict = _cake_candle_family(title, core, desc, text)
    if verdict is not None:
        return verdict

    evidence = match(r"^салют\W*$", core)
    if evidence and match(r"залп|калибр|класс опасности пиротехнического изделия", text):
        return yes("P18_fireworks", evidence)

    # ---- hazardous component inside a mixed kit ------------------------
    kit_title = match(r"набор|комплект|календар|домик", title)
    if kit_title:
        evidence = match(
            r"(?:в наборе есть[^.!?]{0,700}спички|"
            r"спички\s+(?:охотничьи|водоветроустойчивые)|"
            r"погодоустойчивые спички\s*[,—:-]?\s*\d+\s*штук|"
            r"спички[^.!?]{0,50}\d+\s*(?:шт|штук|спичек|упаков)|"
            r"(?:зажигалка|пьезо-зажигалка)\s*(?:x|х)?\s*1(?:\s*шт)?|"
            r"внутри[^.!?]{0,100}бенгальские огни)",
            text,
        )
        if evidence and not match(r"без зажигалк|без спич|не входит", evidence):
            return yes("P20_hazardous_item_in_kit", evidence)

    evidence = match(r"зажигалка\s+в комплекте", text)
    if evidence and match(r"набор|комплект|домик", title):
        return yes("P20_hazardous_item_in_kit", evidence)

    evidence = match(r"зажигалк", title)
    if (
        evidence
        and match(r"набор|комплект|календар", title)
        and not match(r"для зажигалк|без зажигалк|фитил|кремн|топливо", title)
    ):
        return yes("P20_hazardous_item_in_kit", evidence)

    evidence = match(r"спички\s+водоветроустойч\w*", text)
    if evidence and kit_title:
        return yes("P20_hazardous_item_in_kit", evidence)

    evidence = match(r"полный состав набора[^.!?]{0,1200}пьезо-зажигалка", text)
    if evidence and kit_title:
        return yes("P20_hazardous_item_in_kit", evidence)

    evidence = match(r"древесн\w* уголь\s*\d+\s*(?:л|кг)", text)
    if evidence and match(r"набор|комплект|мангал|грил", title):
        return yes("P20_hazardous_item_in_kit", evidence)

    # ---- biofireplace / smoke generator --------------------------------
    if match(r"биокамин", title):
        evidence = match(
            r"(?:в комплект|комплект|в набор|помещается)[^.]{0,220}"
            r"(?:спич[^.]{0,100}биотоплив|биотоплив[^.]{0,100}спич)",
            text,
        )
        if evidence and not match(r"биотопливо[^.]{0,60}(?:не входит|отсутствует)|без биотоплива", text):
            return yes("P21_biofireplace_with_fuel_and_matches", evidence)

    if match(r"дымогенератор", title):
        evidence = match(
            r"(?:в комплект|комплект подарков|комплектац)[^.]{0,220}зажигалк|"
            r"зажигалк[^.]{0,100}(?:в комплект|в подарок)",
            text,
        )
        if evidence and not match(r"без зажигалк|не входит|приобретается отдельно", text):
            return yes("P22_smoke_generator_with_included_lighter", evidence)

    return skip("N99_default", "no confident rule matched")


def match_lvj_rule(name: object, description: object) -> tuple[int | None, str, str]:
    """Positive-only contract: returns 1 or ``None``, never 0."""

    label, rule, evidence = classify_lvj(name, description)
    if label == 1:
        return 1, rule, evidence
    return None, rule, evidence


def predict_lvj_regex(name: object, description: object) -> tuple[int, str, str]:
    """Standalone prediction: a row no rule covers is treated as label 0."""

    label, rule, evidence = classify_lvj(name, description)
    return (0 if label is None else int(label)), rule, evidence


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def annotate(frame: pd.DataFrame, positive_only: bool = False) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    engine = match_lvj_rule if positive_only else classify_lvj
    verdicts = [engine(row.name_, row.description) for row in frame.rename(columns={"name": "name_"}).itertuples()]
    out = frame.copy()
    out["rule_label"] = [v[0] for v in verdicts]
    out["rule"] = [v[1] for v in verdicts]
    out["evidence"] = [v[2] for v in verdicts]
    out["rule_matched"] = out["rule_label"].notna()
    out["regex_prediction"] = out["rule_label"].fillna(0).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--data-path", type=Path, default=Path("data.csv"))
    parser.add_argument("-o", "--output-path", type=Path)
    parser.add_argument("--lvj-only", action="store_true")
    parser.add_argument("--positive-only", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.data_path, keep_default_na=False)
    if args.lvj_only and "category" in frame.columns:
        frame = frame.loc[frame["category"].eq(LVJ_CATEGORY)].copy()

    annotated = annotate(frame, positive_only=args.positive_only)
    if args.output_path:
        annotated.to_csv(args.output_path, index=False)

    if "label" in annotated.columns:
        gold = annotated["label"].astype(int)
        pred = annotated["regex_prediction"]
        tp = int(((pred == 1) & (gold == 1)).sum())
        fp = int(((pred == 1) & (gold == 0)).sum())
        fn = int(((pred == 0) & (gold == 1)).sum())
        tn = int(((pred == 0) & (gold == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"rows={len(annotated)} matched={int(annotated['rule_matched'].sum())}")
        print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}")


if __name__ == "__main__":
    main()
