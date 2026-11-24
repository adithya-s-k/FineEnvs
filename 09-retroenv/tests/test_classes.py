from __future__ import annotations

import pytest
from retroenv.classes import CONSTRAINABLE_CLASSES, REACTION_CLASSES, RULES, TAXONOMY, classify_step

CASES = [
    ("amide coupling", "Cc1ccc(N)cc1.O=C(O)c1ccccc1", "Cc1ccc(NC(=O)c2ccccc2)cc1"),
    ("esterification", "CO.O=C(O)CCC(=O)c1ccccc1O", "COC(=O)CCC(=O)c1ccccc1O"),
    ("sulfonamide formation", "Cc1ccc(N)c(F)c1.O=S(=O)(Cl)c1ccc(Br)cc1", "Cc1ccc(NS(=O)(=O)c2ccc(Br)cc2)c(F)c1"),
    ("urea/carbamate formation", "CN.O=C=Nc1ccc(Br)cc1", "CNC(=O)Nc1ccc(Br)cc1"),
    ("SNAr", "CC1CCCCCN1.Fc1c(Cl)ncnc1Cl", "CC1CCCCCN1c1ncnc(Cl)c1F"),
    ("Suzuki coupling", "CNc1cc(I)nc(N)n1.COc1cc(C)c(B(O)O)cc1C", "CNc1cc(-c2cc(C)c(OC)cc2C)nc(N)n1"),
    ("Sonogashira coupling", "C#C[Si](C)(C)C.COC1CCc2cc(Br)ccc2C1", "COC1CCc2cc(C#C[Si](C)(C)C)ccc2C1"),
    ("N-alkylation", "CC(=O)NCc1ccc(CCl)cc1.c1cnc(N2CCNCC2)nc1", "CC(=O)NCc1ccc(CN2CCN(c3ncccn3)CC2)cc1"),
    ("reductive amination", "O=Cc1ccccc1.NC1CCCC1", "c1ccc(CNC2CCCC2)cc1"),
    ("Boc deprotection", "CC(C)(C)OC(=O)N1CCc2ncnc(Cl)c2CC1", "Clc1ncnc2c1CCNCC2"),
    ("ester hydrolysis", "COC(=O)c1noc(-c2ccccc2)c1C1CC1", "O=C(O)c1noc(-c2ccccc2)c1C1CC1"),
    ("nitro reduction", "O=C(O)CCC(=O)c1cc([N+](=O)[O-])ccc1O", "Nc1ccc(O)c(C(=O)CCC(=O)O)c1"),
    ("heteroaromatic ring formation", "Nc1cc(C(F)(F)F)ccc1S.O=Cc1ccncc1F", "Fc1cnccc1-c1nc2cc(C(F)(F)F)ccc2s1"),
    ("halogenation/nitration", "BrC(Br)(Br)Br.CC(C)(C)OC(=O)N1CCC(CCCO)CC1", "CC(C)(C)OC(=O)N1CCC(CCCBr)CC1"),
]


@pytest.mark.parametrize("expected,reactants,product", CASES)
def test_named_reactions_get_their_class(expected: str, reactants: str, product: str) -> None:
    assert classify_step(product, reactants.split(".")).name == expected


def test_reactant_order_does_not_matter() -> None:
    _, reactants, product = CASES[0]
    assert classify_step(product, reactants.split(".")) == classify_step(product, reactants.split(".")[::-1])


def test_unparseable_input_is_other_not_an_error() -> None:
    assert classify_step("not a smiles", ["CCO"]).name == "other"


def test_every_rule_maps_to_a_class_and_constrainable_classes_exist() -> None:
    assert all(rule in TAXONOMY for rule, _, _ in RULES)
    assert set(CONSTRAINABLE_CLASSES) <= set(REACTION_CLASSES)
