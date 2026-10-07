from __future__ import annotations

from retroenv.chemistry import canonicalize_smiles
from retroenv.disconnections import describe, role, step_family, strategic_disconnections

TARGET = "C[C@@H]1CCCN1CCc1ccc(-c2ccc(S(C)(=O)=O)cc2)cc1"


def test_the_biaryl_cut_puts_the_boronic_acid_on_the_smaller_partner():
    cuts = strategic_disconnections(TARGET)
    suzuki = next(c for c in cuts if c.family == "Suzuki-type coupling")
    assert set(suzuki.reactants) == {
        canonicalize_smiles("CS(=O)(=O)c1ccc(B(O)O)cc1"),
        canonicalize_smiles("C[C@@H]1CCCN1CCc1ccc(Br)cc1"),
    }
    assert cuts[0] == suzuki  # reliable and convergent, so ranked first


def test_amide_and_sulfonamide_cuts_give_their_textbook_partners():
    amide = strategic_disconnections("CC(=O)Nc1ccc(O)cc1")[0]
    assert set(amide.reactants) == {"CC(=O)O", "Nc1ccc(O)cc1"}
    sulfonamide = strategic_disconnections("Cc1ccc(S(=O)(=O)NCc2ccccc2)cc1")[0]
    assert set(sulfonamide.reactants) == {"Cc1ccc(S(=O)(=O)Cl)cc1", "NCc1ccccc1"}


def test_partners_are_named_for_the_group_that_reacts():
    alcohol = "OCCc1ccc(Br)cc1"
    assert role(alcohol) == "aryl halide"
    assert role(alcohol, "sulfonylation") == "alcohol"
    reactants = ["CS(=O)(=O)Cl", alcohol]
    assert (
        describe(step_family(reactants, "CS(=O)(=O)OCCc1ccc(Br)cc1"), reactants, "CS(=O)(=O)OCCc1ccc(Br)cc1")
        == "mesylation of the alcohol with methanesulfonyl chloride"
    )


def test_one_reactant_steps_and_carbamates_get_specific_names():
    azide = ["CC(C)(C)OC(=O)N(Cc1cc(CCN=[N+]=[N-])ccc1Cl)C1CC1"]
    product = "CC(C)(C)OC(=O)N(Cc1cc(CCN)ccc1Cl)C1CC1"
    assert describe(step_family(azide, product), azide, product) == "reduction of the azide"
    carbamate = ["COC(=O)Cl", product]
    target = "COC(=O)NCCc1ccc(Cl)c(CN(C(=O)OC(C)(C)C)C2CC2)c1"
    assert describe(step_family(carbamate, target), carbamate, target).startswith("carbamate formation of the amine")


def test_a_reaction_too_coarse_to_name_is_not_named():
    reactants = ["CC(=O)O", "c1ccccc1"]  # no rule recognises this pairing
    text = describe("other", reactants, "CC(=O)c1ccccc1")
    assert text.startswith("a reported reaction of")
