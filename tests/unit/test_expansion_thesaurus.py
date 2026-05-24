"""Unit tests for the medical thesaurus used by query expansion."""

from __future__ import annotations

from plugins.expansion.thesaurus import MEDICAL_THESAURUS, expand_term


class TestExpandTerm:
    def test_known_term_returns_synonyms(self):
        out = expand_term("hipertensión")
        assert "hta" in out
        assert "presión" in out

    def test_unknown_term_returns_empty_frozenset(self):
        out = expand_term("zzz_no_such_term")
        assert out == frozenset()

    def test_case_insensitive(self):
        assert expand_term("HIPERTENSIÓN") == expand_term("hipertensión")

    def test_does_not_include_itself(self):
        # 'hipertensión' must NOT appear in its own synonym set
        assert "hipertensión" not in expand_term("hipertensión")


class TestBidirectionality:
    def test_hipertension_hta_bidirectional(self):
        assert "hta" in expand_term("hipertensión")
        assert "hipertensión" in expand_term("hta")

    def test_infarto_iam_bidirectional(self):
        assert "iam" in expand_term("infarto")
        assert "infarto" in expand_term("iam")

    def test_ataque_to_infarto_bidirectional(self):
        # Lay term 'ataque' must point to technical 'infarto'
        assert "infarto" in expand_term("ataque")
        # And the reverse: someone writing 'infarto' will get 'ataque' offered
        assert "ataque" in expand_term("infarto")

    def test_azucar_to_glucosa_bidirectional(self):
        assert "glucosa" in expand_term("azúcar")
        assert "azúcar" in expand_term("glucosa")


class TestThesaurusInvariants:
    def test_all_values_are_frozensets(self):
        for term, synonyms in MEDICAL_THESAURUS.items():
            assert isinstance(synonyms, frozenset), term

    def test_no_self_loops(self):
        for term, synonyms in MEDICAL_THESAURUS.items():
            assert term not in synonyms, f"self-loop: {term}"

    def test_all_keys_lowercase(self):
        for term in MEDICAL_THESAURUS:
            assert term == term.lower(), f"key not lower-cased: {term!r}"

    def test_no_empty_synonym_sets(self):
        for term, synonyms in MEDICAL_THESAURUS.items():
            assert synonyms, f"{term!r} has empty synonym set"
