"""mark_brands: the brand-like tokens a translation must leave alone.
Cases from the 250 hand-labelled names and titles of 2026-10-01."""
import pytest

from src.domain.brand_marks import is_common_word, only_names, protect, restore


def mark_brands(text: str) -> str:
    """The protected spans shown in <...>, to read the cases at a glance."""
    out, names = protect(text)
    for i, name in enumerate(names, 1):
        out = out.replace("{" + str(i) + "}", f"<{name}>")
    return out


@pytest.mark.parametrize("text, marked", [
    # coined names: digits, camel humps
    ("save2safe", "<save2safe>"),
    ("Green Energy Partnership - OptiNERG", "Green Energy Partnership - <OptiNERG>"),
    ("BASgas", "<BASgas>"),
    # acronyms that are not words, in mixed case
    ("CPAM des Yvelines", "<CPAM> des Yvelines"),
    ("Elders IV vol. I – MIPOS IV- vol. I", "Elders IV vol. I – <MIPOS> IV- vol. I"),
    # dotted abbreviations and legal forms
    ("A.O.U. Policlinico", "<A.O.U.> Policlinico"),
    ("Kerckhoff-Klinik GmbH", "Kerckhoff-Klinik <GmbH>"),
    ("Savona s.r.l. (SEA-S s.r.l.)", "Savona <s.r.l.> (SEA-S <s.r.l.>)"),
    # a one-word all-caps text that is not a word, or is a coined word
    ("QUEST", "<QUEST>"),
    # left alone: ordinary words, places, Roman numerals, ordinals
    ("Ville de Dugny", "Ville de Dugny"),
    ("MAIRIE DE SAINT-GILLES", "MAIRIE DE SAINT-GILLES"),
    ("Water infrastructure - PHASE 2", "Water infrastructure - PHASE 2"),
    ("Active in the community - Phase IV", "Active in the community - Phase IV"),
    ("Thermomodernization of the 13th Gymnasium", "Thermomodernization of the 13th Gymnasium"),
    ("", ""),
])
def test_marks_only_what_no_language_uses_as_a_word(text, marked):
    assert mark_brands(text) == marked


def test_adjacent_marks_share_one_pair():
    assert mark_brands("SRIP GoDigital") == "<SRIP GoDigital>"


def test_a_two_word_all_caps_text_of_common_words_is_translated():
    """'UOC PROVVEDITORATO' must still be translated; two capitalised
    common words carry no sign of being a brand."""
    assert mark_brands("ALTER ENERGIES") == "ALTER ENERGIES"


def test_common_words_come_from_any_eu_language():
    assert is_common_word("Phase") and is_common_word("Mairie") and is_common_word("STRASSE")
    assert not is_common_word("OKANTIS") and not is_common_word("CPAM")


@pytest.mark.parametrize("word, camel", [
    ("OptiNERG", True), ("BASgas", True), ("GoDigital", True), ("iPhone", True),
    ("Paris", False), ("CPAM", False), ("Saint-Martin", False), ("ABc", False),
])
def test_camel_humps(word, camel):
    from src.domain.brand_marks import _camel  # pylint: disable=import-outside-toplevel
    assert _camel(word) is camel



def test_protect_and_restore_round_trip():
    text, names = protect("Projekt save2safe der CPAM, Phase IV")
    assert text == "Projekt {1} der {2}, Phase IV" and names == ["save2safe", "CPAM"]
    assert restore("{1} project of {2}, phase IV", names) == "save2safe project of CPAM, phase IV"
    assert restore("project of {2}", names) is None              # {1} lost
    assert restore("{1} {1} of {2}", names) is None              # {1} doubled


@pytest.mark.parametrize("text, names_only", [
    ("QUEST", True), ("save2safe", True), ("SRIP GoDigital", True), ("CPAM 87", True),
    ("CPAM des Yvelines", False), ("Ville de Dugny", False), ("", False),
])
def test_a_text_of_nothing_but_names(text, names_only):
    assert only_names(text) is names_only
