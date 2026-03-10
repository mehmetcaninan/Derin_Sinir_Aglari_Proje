from __future__ import annotations

from dataclasses import dataclass


CAPITAL_SIGN = "⠠"
NUMBER_SIGN = "⠼"


@dataclass(frozen=True)
class BrailleOptions:
    use_capital_sign: bool = True
    use_number_sign: bool = True


# Grade-1 (English) braille patterns (Unicode braille cells).
# Letters a-z
_LETTERS = {
    "a": "⠁",
    "b": "⠃",
    "c": "⠉",
    "d": "⠙",
    "e": "⠑",
    "f": "⠋",
    "g": "⠛",
    "h": "⠓",
    "i": "⠊",
    "j": "⠚",
    "k": "⠅",
    "l": "⠇",
    "m": "⠍",
    "n": "⠝",
    "o": "⠕",
    "p": "⠏",
    "q": "⠟",
    "r": "⠗",
    "s": "⠎",
    "t": "⠞",
    "u": "⠥",
    "v": "⠧",
    "w": "⠺",
    "x": "⠭",
    "y": "⠽",
    "z": "⠵",
}

# Digits 1-0 map to a-j with number sign prefix.
_DIGITS = {
    "1": _LETTERS["a"],
    "2": _LETTERS["b"],
    "3": _LETTERS["c"],
    "4": _LETTERS["d"],
    "5": _LETTERS["e"],
    "6": _LETTERS["f"],
    "7": _LETTERS["g"],
    "8": _LETTERS["h"],
    "9": _LETTERS["i"],
    "0": _LETTERS["j"],
}

_PUNCT = {
    " ": " ",
    "\n": "\n",
    ".": "⠲",
    ",": "⠂",
    ";": "⠆",
    ":": "⠒",
    "?": "⠦",
    "!": "⠖",
    "-": "⠤",
    "'": "⠄",
    '"': "⠶",
    "(": "⠶",
    ")": "⠶",
    "/": "⠌",
    "\\": "⠡",
    "@": "⠈⠁",
    "#": NUMBER_SIGN,
}


def to_braille(text: str, options: BrailleOptions | None = None) -> str:
    """
    Convert Latin text to Unicode Braille (grade-1 style).

    - Uppercase is represented with a capital sign prefix (⠠) by default.
    - Digits are represented with number sign prefix (⠼) by default, using a-j mapping.
    """
    if options is None:
        options = BrailleOptions()

    out: list[str] = []
    in_number_mode = False

    for ch in text:
        if ch.isdigit():
            if options.use_number_sign and not in_number_mode:
                out.append(NUMBER_SIGN)
                in_number_mode = True
            out.append(_DIGITS.get(ch, "⍰"))
            continue

        in_number_mode = False

        if ch.isalpha():
            if ch.isupper():
                if options.use_capital_sign:
                    out.append(CAPITAL_SIGN)
                out.append(_LETTERS.get(ch.lower(), "⍰"))
            else:
                out.append(_LETTERS.get(ch, "⍰"))
            continue

        out.append(_PUNCT.get(ch, "⍰"))

    return "".join(out)


def supported_charset() -> set[str]:
    chars = set(_LETTERS.keys()) | set(k.upper() for k in _LETTERS.keys())
    chars |= set(_DIGITS.keys())
    chars |= set(_PUNCT.keys())
    return chars

