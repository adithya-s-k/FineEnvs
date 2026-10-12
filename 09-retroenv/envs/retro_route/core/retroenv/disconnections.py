"""The chemistry a planner reasons with: reaction families, reagent roles and strategic cuts.

Everything here is rule-based on structures alone, so any statement built from it
can be checked against the molecules. ``step_family`` names the reaction a step
performs, ``role`` names what a reagent is (a boronic acid, an aryl halide, an
amine), and ``strategic_disconnections`` proposes the cuts a chemist would
consider first: amide, sulfonamide, ester and ether bonds, biaryl bonds, and C-N
bonds made by alkylation, reductive amination or aryl substitution. A proposal is
a plausible disconnection, not a claim that it was the route actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from rdkit import Chem

from .chemistry import canonicalize_smiles

PATTERNS = {
    "boronic": Chem.MolFromSmarts("[#6]B([OX2])[OX2]"),
    "acyl_halide": Chem.MolFromSmarts("C(=O)[Cl,Br]"),
    "acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    "amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
    "alcohol": Chem.MolFromSmarts("[OX2H][CX4,c]"),
    "carbonyl": Chem.MolFromSmarts("[CX3H1,CX3H0;!$(C(=O)[O,N])](=O)[#6,#1]"),
    "halide": Chem.MolFromSmarts("[#6][Cl,Br,I]"),
    "sulfonyl_halide": Chem.MolFromSmarts("S(=O)(=O)[Cl,F]"),
    "isocyanate": Chem.MolFromSmarts("N=C=O"),
}
BOC = Chem.MolFromSmarts("CC(C)(C)OC(=O)[N,O,n]")
BOC2O = Chem.MolFromSmiles("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")


@lru_cache(maxsize=65536)
def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def _has(smiles: str, key: str) -> bool:
    mol = _mol(smiles)
    return mol is not None and mol.HasSubstructMatch(PATTERNS[key])


def step_family(reactants: Iterable[str], product: str) -> str:
    """Coarse rule-based reaction family; 'other' when no rule fires."""
    reactants = list(reactants)
    mols = [_mol(r) for r in reactants]
    if any(m is not None and m.HasSubstructMatch(BOC2O) for m in mols):
        return "Boc protection"
    if len(reactants) == 1:
        p = _mol(product)
        if mols[0].HasSubstructMatch(BOC) and len(p.GetSubstructMatches(BOC)) < len(mols[0].GetSubstructMatches(BOC)):
            return "Boc deprotection"
        return "one-reactant FGI"
    if any(_has(r, "boronic") for r in reactants):
        return "Suzuki-type coupling"
    if any(_has(r, "sulfonyl_halide") for r in reactants):
        return "sulfonylation"
    if any(_has(r, "isocyanate") for r in reactants):
        return "urea / carbamate"
    if any(_has(r, "acyl_halide") or _has(r, "acid") for r in reactants) and any(
        _has(r, "amine") or _has(r, "alcohol") for r in reactants
    ):
        return "amide/ester coupling"
    if any(_has(r, "carbonyl") for r in reactants) and any(_has(r, "amine") for r in reactants):
        return "reductive amination"
    if any(_has(r, "halide") for r in reactants):
        return "alkylation / SNAr"
    return "other"


# --- What a reagent is ---------------------------------------------------------

SMALL = {
    "N": "ammonia",
    "O": "water",
    "CO": "methanol",
    "CCO": "ethanol",
    "CN": "methylamine",
    "CNC": "dimethylamine",
    "C=O": "formaldehyde",
    "CC(=O)O": "acetic acid",
    "CI": "methyl iodide",
    "CS(=O)(=O)Cl": "methanesulfonyl chloride",
    "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C": "Boc anhydride",
    "CC(=O)Cl": "acetyl chloride",
    "CC(=O)OC(C)=O": "acetic anhydride",
    "NN": "hydrazine",
    "NO": "hydroxylamine",
}
# First match wins, so the more specific groups come first.
ROLES = [
    (name, Chem.MolFromSmarts(smarts))
    for name, smarts in (
        ("boronic acid", "[#6]B([OX2H])[OX2H]"),
        ("boronate ester", "[#6]B([OX2][#6])[OX2][#6]"),
        ("sulfonyl chloride", "[#6]S(=O)(=O)[Cl,F]"),
        ("acyl chloride", "[#6]C(=O)[Cl,Br]"),
        ("chloroformate", "[#6]OC(=O)Cl"),
        ("isocyanate", "N=C=O"),
        ("carboxylic acid", "[CX3](=O)[OX2H1]"),
        ("sulfonate ester", "[#6]S(=O)(=O)O[CX4]"),
        ("aldehyde", "[CX3H1](=O)[#6]"),
        ("ketone", "[#6][CX3](=O)[#6]"),
        ("heteroaryl halide", "[$(c([Cl,Br,I,F])n),$(c([Cl,Br,I,F])an),$(c([Cl,Br,I,F])aan)]"),
        ("aryl halide", "c[Cl,Br,I]"),
        ("alkyl halide", "[CX4][Cl,Br,I]"),
        ("amine", "[NX3;H2,H1;!$(N[C,S]=[O,S,N]);!$(N-a)]"),
        ("aniline", "[NX3;H2,H1;!$(N[C,S]=O)]-c"),
        ("phenol", "c[OX2H]"),
        ("alcohol", "[CX4][OX2H]"),
        ("ester", "[#6][CX3](=O)O[#6]"),
        ("nitro compound", "[N+](=O)[O-]"),
        ("azide", "[N-]=[N+]=[N-]"),
        ("organic azide", "[#6]N=[N+]=[N-]"),
        ("cyanide", "[C-]#N"),
        ("nitrile", "C#N"),
    )
]


# The groups that take part in each family, so a reagent is named for the group that
# reacts: in a mesylation, 4-bromophenethyl alcohol is the alcohol, not the aryl halide.
FAMILY_ROLES = {
    "Suzuki-type coupling": ("boronic acid", "boronate ester", "heteroaryl halide", "aryl halide"),
    "amide/ester coupling": ("acyl chloride", "carboxylic acid", "amine", "aniline", "alcohol", "phenol"),
    "sulfonylation": ("sulfonyl chloride", "amine", "aniline", "alcohol", "phenol"),
    "reductive amination": ("aldehyde", "ketone", "amine", "aniline"),
    "alkylation / SNAr": (
        "alkyl halide",
        "sulfonate ester",
        "heteroaryl halide",
        "aryl halide",
        "amine",
        "aniline",
        "phenol",
        "alcohol",
    ),
    "urea / carbamate": ("isocyanate", "chloroformate", "amine", "aniline", "alcohol"),
    "Boc protection": ("amine", "aniline"),
}
_ROLE_PATTERNS = dict(ROLES)


@lru_cache(maxsize=65536)
def role(smiles: str, family: str | None = None) -> str:
    """The reagent's characteristic group, in words a chemist would use.

    With a ``family``, the group that reaction uses wins over others the molecule
    happens to carry.
    """
    canonical = canonicalize_smiles(smiles)
    if canonical in SMALL:
        return SMALL[canonical]
    mol = _mol(canonical)
    for name in FAMILY_ROLES.get(family or "", ()):
        if mol.HasSubstructMatch(_ROLE_PATTERNS[name]):
            return name
    for name, pattern in ROLES:
        if mol.HasSubstructMatch(pattern):
            return name
    return "building block"


AMINES = {"amine", "aniline", "ammonia", "methylamine", "dimethylamine", "hydrazine", "hydroxylamine"}
NUCLEOPHILES = AMINES | {"phenol", "alcohol", "methanol", "ethanol", "water", "azide", "cyanide"}
TERMINAL_ALKYNE = Chem.MolFromSmarts("[CH]#C")
# Names too coarse to state as fact: text omits the reaction type rather than guess it.
UNCERTAIN = {"halide substitution", "functional-group interconversion", "a reported transformation"}


def reaction_phrase(family: str, reactants: Iterable[str], product: str) -> str:
    """The forward reaction a step performs, e.g. 'Suzuki coupling' or 'nitro reduction'."""
    reactants = list(reactants)
    roles = {role(r, family) for r in reactants}
    if family == "Suzuki-type coupling":
        return "Suzuki coupling"
    if family == "amide/ester coupling":
        if "chloroformate" in roles:
            return "carbamate formation"
        return "amide coupling" if roles & AMINES else "esterification"
    if family == "sulfonylation":
        if roles & AMINES:
            return "sulfonamide formation"
        return "mesylation" if "methanesulfonyl chloride" in roles else "sulfonylation"
    if family == "reductive amination":
        return "reductive amination"
    if family == "alkylation / SNAr":
        if roles & {"acyl chloride", "acetyl chloride"}:
            return "acylation"
        if roles & {"alkyl halide", "sulfonate ester", "methyl iodide"}:
            if "azide" in roles:
                return "azide displacement"
            if "cyanide" in roles:
                return "cyanide displacement"
            return "alkylation"
        aryl = roles & {"aryl halide", "heteroaryl halide"}
        if aryl and any(_mol(r).HasSubstructMatch(TERMINAL_ALKYNE) for r in reactants):
            return "Sonogashira coupling"
        if "heteroaryl halide" in roles and roles & NUCLEOPHILES:
            return "SNAr substitution"
        if "aryl halide" in roles and roles & AMINES:
            return "Buchwald-Hartwig amination"
        if "aryl halide" in roles and roles & {"phenol", "alcohol"}:
            return "aryl ether coupling"
        return "halide substitution"
    if family == "urea / carbamate":
        return "urea or carbamate formation"
    if family in ("Boc protection", "Boc deprotection"):
        return family
    if family == "one-reactant FGI" and reactants:
        for name, lost, gained in FGI:
            if _count(reactants[0], lost) > _count(product, lost) and (
                gained is None or _count(product, gained) > _count(reactants[0], gained)
            ):
                return name
        return "functional-group interconversion"
    return "a reported transformation"


# The partner a chemist names first ("Suzuki coupling of the aryl halide with the
# boronic acid", "alkylation of the amine with the mesylate").
SUBSTRATE_FIRST = {
    "Suzuki coupling": ("heteroaryl halide", "aryl halide"),
    "amide coupling": ("carboxylic acid", "acyl chloride", "acetic acid", "acetyl chloride"),
    "esterification": ("carboxylic acid", "acyl chloride", "acetic acid", "acetyl chloride"),
    "sulfonamide formation": tuple(AMINES),
    "mesylation": ("alcohol", "phenol", "amine", "aniline"),
    "sulfonylation": ("alcohol", "phenol", "amine", "aniline"),
    "reductive amination": ("aldehyde", "ketone", "formaldehyde"),
    "alkylation": tuple(NUCLEOPHILES),
    "SNAr substitution": ("heteroaryl halide",),
    "Buchwald-Hartwig amination": ("aryl halide",),
    "aryl ether coupling": ("aryl halide",),
    "Sonogashira coupling": ("heteroaryl halide", "aryl halide"),
    "urea or carbamate formation": tuple(NUCLEOPHILES),
    "Boc protection": ("amine", "aniline"),
    "carbamate formation": tuple(NUCLEOPHILES),
    "acylation": (),
    "azide displacement": ("sulfonate ester", "alkyl halide"),
    "cyanide displacement": ("sulfonate ester", "alkyl halide"),
}
# A one-reactant step, said the way a chemist would.
SINGLE = {
    "Boc deprotection": "removal of the Boc group",
    "nitro reduction": "reduction of the nitro group",
    "azide reduction": "reduction of the azide",
    "nitrile reduction": "reduction of the nitrile",
    "ester hydrolysis": "hydrolysis of the ester",
    "ester reduction": "reduction of the ester to the alcohol",
    "O-demethylation": "O-demethylation of the methyl ether",
    "debenzylation": "removal of the benzyl group",
    "carbonyl reduction": "reduction of the carbonyl",
    "alcohol oxidation": "oxidation of the alcohol",
}
# Phrases whose verb reads better on its own when the partner is named after it.
VERB = {"azide displacement": "displacement", "cyanide displacement": "displacement"}
# The reagent a chemist names last.
REAGENT_LAST = {"acylation": ("acyl chloride", "acetyl chloride")}
NAMED = set(SMALL.values())


def _named(role_name: str) -> str:
    return role_name if role_name in NAMED else f"the {role_name}"


def describe(family: str, reactants: Iterable[str], product: str) -> str:
    """The forward reaction with its partners, e.g. 'mesylation of the alcohol with methanesulfonyl chloride'."""
    reactants = list(reactants)
    phrase = reaction_phrase(family, reactants, product)
    if len(reactants) == 1:
        if phrase in UNCERTAIN:
            return "a reported one-step transformation of a single precursor"
        return SINGLE.get(phrase) or f"{phrase} of {_named(role(reactants[0], family))}"
    first, last = SUBSTRATE_FIRST.get(phrase, ()), REAGENT_LAST.get(phrase, ())
    ordered = sorted(reactants, key=lambda r: (role(r, family) in last, role(r, family) not in first))
    names = [_named(role(r, family)) for r in ordered]
    if phrase in UNCERTAIN:
        # Too coarse to name: say what reacts, not a reaction we cannot vouch for.
        return f"a reported reaction of {names[0]} with {' and '.join(names[1:])}"
    return f"{VERB.get(phrase, phrase)} of {names[0]} with {' and '.join(names[1:])}"


# --- Strategic disconnections ----------------------------------------------------


# One-reactant changes, most specific first: (name, group lost, group gained or None).
FGI = [
    (name, Chem.MolFromSmarts(lost), Chem.MolFromSmarts(gained) if gained else None)
    for name, lost, gained in (
        ("nitro reduction", "[N+](=O)[O-]", "[NX3;H2]"),
        ("azide reduction", "[#6]N=[N+]=[N-]", "[NX3;H2]"),
        ("nitrile reduction", "C#N", "[CH2][NX3;H2]"),
        ("ester hydrolysis", "[#6][CX3](=O)O[CX4]", "[CX3](=O)[OX2H1]"),
        ("ester reduction", "[#6][CX3](=O)O[CX4]", "[CH2][OX2H]"),
        ("O-demethylation", "c[OX2][CH3]", "c[OX2H]"),
        ("debenzylation", "[O,N][CH2]c1ccccc1", None),
        ("carbonyl reduction", "[#6][CX3](=O)[#6,#1]", "[CX4][OX2H]"),
        ("alcohol oxidation", "[CX4][OX2H]", "[CX3]=O"),
    )
]


def _count(smiles: str, pattern) -> int:
    return len(_mol(smiles).GetSubstructMatches(pattern))


@dataclass(frozen=True)
class Disconnection:
    reactants: tuple[str, ...]
    family: str
    bond: str
    score: float


# (family, bond in words, SMARTS whose atoms :1 and :2 are the bond, cap on 1, cap on 2, reliability)
RULES = [
    ("amide/ester coupling", "amide C-N bond", "[#6][CX3:1](=O)-!@[NX3;!$(N[S](=O)=O):2]", "OH", "H", 1.0),
    ("sulfonylation", "sulfonamide S-N bond", "[#6][SX4:1](=O)(=O)-!@[NX3:2]", "Cl", "H", 0.95),
    ("Suzuki-type coupling", "biaryl C-C bond", "[c:1]-!@[c:2]", "SUZUKI", "SUZUKI", 0.95),
    ("alkylation / SNAr", "aryl C-N bond", "[c:1]-!@[NX3;!$(N[C,S]=O):2]", "ARX", "H", 0.85),
    ("reductive amination", "alkyl C-N bond", "[CX4;H2;$(C[#6]):1]-!@[NX3;!$(N[C,S]=O):2]", "=O", "H", 0.85),
    ("alkylation / SNAr", "alkyl C-N bond", "[CX4;H2,H1;$(C[#6]):1]-!@[NX3;!$(N[C,S]=O):2]", "Br", "H", 0.8),
    ("alkylation / SNAr", "ether C-O bond", "[CX4;H2:1]-!@[OX2:2]c", "Br", "H", 0.8),
    ("amide/ester coupling", "ester C-O bond", "[#6][CX3:1](=O)-!@[OX2:2][#6]", "OH", "H", 0.75),
]
_RULES = [(f, b, Chem.MolFromSmarts(s), c1, c2, w) for f, b, s, c1, c2, w in RULES]


def _cap(rw: Chem.RWMol, index: int, cap: str) -> None:
    atom = rw.GetAtomWithIdx(index)
    if cap == "H":
        # A bracket atom keeps its hydrogens explicitly; the lost bond becomes one.
        if atom.GetNoImplicit():
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
        return
    if cap == "=O":
        if atom.GetNoImplicit():
            atom.SetNumExplicitHs(max(0, atom.GetNumExplicitHs() - 1))
        oxygen = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(index, oxygen, Chem.BondType.DOUBLE)
        return
    if cap == "B(O)O":
        boron = rw.AddAtom(Chem.Atom(5))
        rw.AddBond(index, boron, Chem.BondType.SINGLE)
        for _ in range(2):
            rw.AddBond(boron, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
        return
    element = {"OH": 8, "Cl": 17, "Br": 35}[cap]
    rw.AddBond(index, rw.AddAtom(Chem.Atom(element)), Chem.BondType.SINGLE)


def _split(mol, a: int, b: int, cap_a: str, cap_b: str) -> tuple[str, ...] | None:
    rw = Chem.RWMol(mol)
    rw.RemoveBond(a, b)
    _cap(rw, a, cap_a)
    _cap(rw, b, cap_b)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        fragments = Chem.GetMolFrags(out, asMols=True)
        if len(fragments) != 2:
            return None
        return tuple(sorted(canonicalize_smiles(Chem.MolToSmiles(f)) for f in fragments))
    except Exception:
        return None


def _boronic_partner_size(reactants: tuple[str, ...]) -> int:
    return min(
        _mol(s).GetNumHeavyAtoms() for s in reactants if any(atom.GetSymbol() == "B" for atom in _mol(s).GetAtoms())
    )


def _heteroaryl(atom) -> bool:
    ring = atom.GetOwningMol().GetRingInfo()
    return any(
        atom.GetIdx() in r and any(atom.GetOwningMol().GetAtomWithIdx(i).GetSymbol() == "N" for i in r)
        for r in ring.AtomRings()
    )


@lru_cache(maxsize=16384)
def strategic_disconnections(product: str) -> tuple[Disconnection, ...]:
    """Plausible one-step cuts of ``product``, most attractive first.

    The score adds the rule's reliability to how evenly the cut splits the molecule,
    because chemists prefer reliable reactions and convergent routes. Bonds on a
    stereocentre are skipped, since re-attaching the cap there could flip it.
    """
    mol = _mol(canonicalize_smiles(product))
    if mol is None:
        return ()
    heavy = mol.GetNumHeavyAtoms()
    found: dict[tuple[str, ...], Disconnection] = {}
    for family, bond, pattern, cap_1, cap_2, reliability in _RULES:
        mapped = {atom.GetAtomMapNum(): atom.GetIdx() for atom in pattern.GetAtoms() if atom.GetAtomMapNum()}
        for match in mol.GetSubstructMatches(pattern):
            a, b = match[mapped[1]], match[mapped[2]]
            if any(mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for i in (a, b)):
                continue
            caps = [(cap_1, cap_2)]
            if cap_1 == "SUZUKI":
                caps = [("Br", "B(O)O"), ("B(O)O", "Br")]
            elif cap_1 == "ARX":
                caps = [("Cl" if _heteroaryl(mol.GetAtomWithIdx(a)) else "Br", cap_2)]
            options = [r for r in (_split(mol, a, b, x, y) for x, y in caps) if r]
            if cap_1 == "SUZUKI" and options:
                # The boronic acid goes on the smaller partner, as it usually would.
                options = [min(options, key=_boronic_partner_size)]
            for reactants in options:
                smallest = min(_mol(s).GetNumHeavyAtoms() for s in reactants)
                score = reliability + 0.6 * min(0.5, smallest / max(heavy, 1))
                key = reactants
                if key not in found or found[key].score < score:
                    found[key] = Disconnection(reactants, family, bond, round(score, 4))
    return tuple(sorted(found.values(), key=lambda d: (-d.score, d.reactants)))
