from __future__ import annotations

from conftest import AMINO_ESTER, ESTER, NITRO_ESTER
from retroenv.reactions import free_key


def test_a_corpus_reaction_is_supported_regardless_of_order_and_stereo(library):
    assert library.support(ESTER, ["CC(=O)O", "CCO"]).kind == "corpus"
    assert library.support(AMINO_ESTER, [NITRO_ESTER]).kind == "corpus"


def test_a_template_supports_an_unseen_reaction(library):
    support = library.support(ESTER, ["CCO", "CC(=O)Cl"])
    assert support.kind == "template" and support.template_count == 124
    assert support.stereo_match is True


def test_common_reagents_may_pad_a_step(library):
    assert library.support(ESTER, ["CCO", "CC(=O)O", "CCN(CC)CC"]).supported
    assert library.support(ESTER, ["CCO", "CC(=O)Cl", "O"]).supported


def test_unrelated_or_invalid_steps_are_rejected(library):
    assert library.support(ESTER, ["CCO", "CN"]).kind == "none"
    assert library.support(ESTER, ["not a smiles"]).kind == "invalid"
    assert library.support(ESTER, [ESTER, "CCO"]).kind == "invalid"


def test_excluding_a_corpus_key_falls_back_to_templates_only(library):
    known = free_key(ESTER, ["CCO", "CC(=O)O"])
    support = library.support(ESTER, ["CCO", "CC(=O)O"], exclude=frozenset({known}))
    assert support.kind == "template"
