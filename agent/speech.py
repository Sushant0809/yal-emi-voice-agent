"""Turn written amounts and dates into the way a person says them.

Runs as a TTS text transform, so it applies to every sentence right before it is
spoken. The prompt also asks the LLM to write naturally; this is the safety net
for when it writes "₹12,500" or "2026-12-05" anyway.

    "₹12,500 due on 2026-12-05" -> "twelve thousand five hundred rupees due on the fifth of December"
"""

import re
from datetime import date

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
    8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
    14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 21: "twenty-first", 22: "twenty-second",
    23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth", 26: "twenty-sixth",
    27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth",
    31: "thirty-first",
}
_MONTHS = (
    "January February March April May June July August September October November December"
).split()


def _below_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _below_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(f"{_ONES[hundreds]} hundred")
    if rest:
        parts.append(_below_hundred(rest))
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """Indian numbering: 1,25,000 -> 'one lakh twenty-five thousand'."""
    if n == 0:
        return "zero"
    parts = []
    crore, n = divmod(n, 10_000_000)
    lakh, n = divmod(n, 100_000)
    thousand, n = divmod(n, 1000)
    if crore:
        parts.append(f"{number_to_words(crore)} crore")
    if lakh:
        parts.append(f"{_below_hundred(lakh)} lakh")
    if thousand:
        parts.append(f"{_below_hundred(thousand)} thousand")
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def rupees_to_words(amount: float) -> str:
    rupees = int(amount)
    paise = round((amount - rupees) * 100)
    spoken = f"{number_to_words(rupees)} rupee{'s' if rupees != 1 else ''}"
    if paise:
        spoken += f" and {number_to_words(paise)} paise"
    return spoken


def date_to_words(d: date, *, with_year: bool = False) -> str:
    spoken = f"the {_ORDINALS[d.day]} of {_MONTHS[d.month - 1]}"
    return f"{spoken} {d.year}" if with_year else spoken


_AMOUNT = re.compile(r"(?:₹|Rs\.?|INR)\s?(\d[\d,]*(?:\.\d{1,2})?)(?:\s?(?:rupees|/-))?", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


# --- Hindi -------------------------------------------------------------------------
# Every number from 1 to 99 has its own word in Hindi, so they're listed rather than composed.
_HI_1_99 = (
    "एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
    "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस "
    "छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस "
    "अड़तालीस उनचास पचास इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ "
    "तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर "
    "छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी "
    "नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
_HI_MONTHS = "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split()


def hindi_number_to_words(n: int) -> str:
    """Indian numbering in Hindi: 38200 -> 'अड़तीस हज़ार दो सौ'."""
    if n == 0:
        return "शून्य"
    parts = []
    for size, word in ((10_000_000, "करोड़"), (100_000, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        count, n = divmod(n, size)
        if count:
            parts.append(f"{hindi_number_to_words(count)} {word}")
    if n:
        parts.append(_HI_1_99[n - 1])
    return " ".join(parts)


# --- Per-language verbalisers ---------------------------------------------------------
class English:
    def amount(self, rupees: float) -> str:
        return rupees_to_words(rupees)

    def date(self, d: date) -> str:
        return date_to_words(d)

    def days(self, n: int) -> str:
        return "one day" if n == 1 else f"{number_to_words(n)} days"


class Hindi:
    def amount(self, rupees: float) -> str:
        return f"{hindi_number_to_words(int(rupees))} रुपये"

    def date(self, d: date) -> str:
        return f"{hindi_number_to_words(d.day)} {_HI_MONTHS[d.month - 1]}"

    def days(self, n: int) -> str:
        return f"{hindi_number_to_words(n)} दिन"


VERBALISERS = {"en": English(), "hi": Hindi()}


def detect_language(text: str) -> str | None:
    """'hi' if the text is mostly Devanagari, 'en' if mostly Latin, None if unclear."""
    deva = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if deva + latin < 3:
        return None
    return "hi" if deva > latin else "en"


def verbaliser(language: str):
    return VERBALISERS.get(language, VERBALISERS["en"])


def to_spoken(text: str) -> str:
    """Rewrite any raw amounts and ISO dates in a TTS chunk in the chunk's own language."""
    v = verbaliser("hi" if re.search(r"[\u0900-\u097f]", text) else "en")
    text = _AMOUNT.sub(lambda m: v.amount(float(m.group(1).replace(",", ""))), text)
    return _ISO_DATE.sub(lambda m: _iso_to_words(m, v), text)


def _iso_to_words(match: re.Match, v) -> str:
    try:
        return v.date(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
    except ValueError:
        return match.group(0)





async def spoken_transform(text: str, text_type: str) -> str:
    """Pipecat ``TextTransform`` signature."""
    return to_spoken(text)
