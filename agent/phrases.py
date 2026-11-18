"""Phrase packs: everything the agent says verbatim, per lender and language, as data.

    config/lenders/<lender>/lender.yaml   name, agent name, supported languages
    config/lenders/<lender>/<lang>.yaml   greeting, reminder, fillers, idle/recovery lines...

Adding a lender or a language is a new folder or file, not a code change. Packs are
validated when loaded (required keys, allowed placeholders), so a typo fails at start-up
instead of mid-call. A phrase missing for a language falls back to the default language.
"""

import string
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from agent.config import CONFIG, ROOT

LENDERS_DIR = ROOT / "config" / "lenders"

# key -> placeholders it may use. Dotted keys address nested sections.
SCHEMA: dict[str, set[str]] = {
    "greeting": {"name", "first_name", "agent_name", "lender"},
    "reminder_due": {"first_name", "loan_type", "amount", "due_date"},
    "reminder_overdue": {"first_name", "loan_type", "amount", "due_date", "days_overdue"},
    "fillers.verify_identity": set(),
    "fillers.send_payment_link": set(),
    "fillers.check_payment_status": set(),
    "idle_prompts": set(),
    "idle_goodbye": set(),
    "recovery.apology": set(),
    "recovery.ask_again": set(),
    "fallback_goodbye": set(),
}


class PhrasePackError(ValueError):
    pass


@dataclass(frozen=True)
class Lender:
    key: str
    name: str
    agent_name: str
    agent_names: dict  # language -> agent name in that language's script (optional)
    languages: tuple[str, ...]
    default_language: str
    packs: dict  # language -> raw pack dict

    def phrase(self, language: str, key: str, **values) -> str:
        """A single phrase, filled in. Falls back to the default language."""
        raw = self._raw(language, key)
        if not isinstance(raw, str):
            raise PhrasePackError(f"{key} is a list; use phrases()")
        agent_name = self.agent_names.get(language, self.agent_name)
        return raw.format(agent_name=agent_name, lender=self.name, **values)

    def phrases(self, language: str, key: str) -> list[str]:
        raw = self._raw(language, key)
        return list(raw) if isinstance(raw, list) else [raw]

    def loan_type(self, language: str, loan_type: str) -> str:
        pack = self.packs.get(language) or self.packs[self.default_language]
        return (pack.get("loan_types") or {}).get(loan_type, loan_type)

    def all_lines(self, key: str) -> set[str]:
        """Every language's version of a fixed line (e.g. to recognise fillers)."""
        return {line for lang in self.languages for line in self.phrases(lang, key)}

    def _raw(self, language: str, key: str):
        for lang in (language, self.default_language):
            value = _lookup(self.packs.get(lang, {}), key)
            if value is not None:
                return value
        raise PhrasePackError(f"phrase {key!r} missing for {language!r} and default language")


def _lookup(pack: dict, dotted: str):
    node = pack
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _placeholders(text: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


def validate_pack(pack: dict, where: str) -> None:
    for key, allowed in SCHEMA.items():
        value = _lookup(pack, key)
        if value is None:
            raise PhrasePackError(f"{where}: missing required phrase {key!r}")
        for text in value if isinstance(value, list) else [value]:
            if not isinstance(text, str) or not text.strip():
                raise PhrasePackError(f"{where}: {key!r} must be non-empty text")
            unknown = _placeholders(text) - allowed
            if unknown:
                raise PhrasePackError(f"{where}: {key!r} uses unknown placeholder(s) {sorted(unknown)}")


def load_lender(key: str, root: Path = LENDERS_DIR) -> Lender:
    folder = root / key
    meta = yaml.safe_load((folder / "lender.yaml").read_text())
    packs = {}
    for lang in meta["languages"]:
        path = folder / f"{lang}.yaml"
        if not path.exists():
            raise PhrasePackError(f"{folder}: language {lang!r} listed but {path.name} missing")
        pack = yaml.safe_load(path.read_text()) or {}
        validate_pack(pack, str(path))
        packs[lang] = pack
    if meta["default_language"] not in packs:
        raise PhrasePackError(f"{folder}: default language {meta['default_language']!r} has no pack")
    return Lender(
        key=key,
        name=meta["name"],
        agent_name=meta["agent_name"],
        agent_names=meta.get("agent_names") or {},
        languages=tuple(meta["languages"]),
        default_language=meta["default_language"],
        packs=packs,
    )


@cache
def lender() -> Lender:
    """The lender this deployment speaks for (LENDER in .env), validated once."""
    return load_lender(CONFIG.lender)
