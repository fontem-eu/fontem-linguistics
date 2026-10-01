"""mark_brands: the brand-like tokens a translation must leave alone.
Cases from the 250 hand-labelled names and titles of 2026-10-01."""
import pytest

from src.domain.brand_marks import is_common_word, mark_brands


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
