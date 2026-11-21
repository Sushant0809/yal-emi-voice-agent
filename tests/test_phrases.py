import shutil

import pytest
import yaml

from agent.phrases import LENDERS_DIR, PhrasePackError, load_lender


def _copy_lumen(tmp_path):
    shutil.copytree(LENDERS_DIR / "lumen", tmp_path / "acme")
    return tmp_path / "acme"


def test_shipped_packs_are_valid():
    lumen = load_lender("lumen")
    assert lumen.languages == ("en", "hi")
    assert lumen.phrase("hi", "greeting", name="Amit Verma", first_name="Amit", recording_notice="").startswith("नमस्ते, मैं Lumen Finance से आशा")


def test_new_lender_needs_no_code(tmp_path):
    folder = _copy_lumen(tmp_path)
    meta = yaml.safe_load((folder / "lender.yaml").read_text())
    meta.update(name="Acme Credit", agent_name="Priya", agent_names={})
    (folder / "lender.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True))
    acme = load_lender("acme", root=tmp_path)
    assert acme.phrase("en", "greeting", name="Rahul Sharma", first_name="Rahul", recording_notice="") == (
        "Hello, this is Priya calling from Acme Credit. Am I speaking with Rahul Sharma?"
    )


def test_missing_phrase_fails_at_load(tmp_path):
    folder = _copy_lumen(tmp_path)
    pack = yaml.safe_load((folder / "hi.yaml").read_text())
    del pack["recovery"]["ask_again"]
    (folder / "hi.yaml").write_text(yaml.safe_dump(pack, allow_unicode=True))
    with pytest.raises(PhrasePackError, match="recovery.ask_again"):
        load_lender("acme", root=tmp_path)


def test_unknown_placeholder_fails_at_load(tmp_path):
    folder = _copy_lumen(tmp_path)
    pack = yaml.safe_load((folder / "en.yaml").read_text())
    pack["greeting"] = "Hi {nmae}"  # typo
    (folder / "en.yaml").write_text(yaml.safe_dump(pack, allow_unicode=True))
    with pytest.raises(PhrasePackError, match="nmae"):
        load_lender("acme", root=tmp_path)
