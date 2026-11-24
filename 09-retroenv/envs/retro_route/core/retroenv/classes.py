"""Deterministic, rule-based reaction classes for arbitrary (product, reactants) pairs.

The verifier, the reaction_class_lookup tool and the dataset builder all call
``classify_step``, so a forbidden-class constraint is judged by exactly the
labeller the agent can query. Rules compare functional-group counts between the
reactant side and the product with SMARTS; the first matching rule wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

CLASSIFIER_VERSION = "retroenv-classes-1"

EWG = "$([N+](=O)[O-]),$(C#N),$(C(F)(F)F),$([CX3]=O),$([SX4](=O)=O)"
ACTIVATION = ["$(c:n)", "$(c:a:a:n)", f"$(c:c-[{EWG}])", f"$(c:a:a:c-[{EWG}])"]
ACTIVATED = ",".join(ACTIVATION)
UNACTIVATED = ";".join(f"!{a}" for a in ACTIVATION)
ETHER_O = "[OX2;H0;!$(O[CX3]=[O,S,N]);!$(O[S,P,Si,B])]"
TRIFLATE = "OS(=O)(=O)C(F)(F)F"

GROUPS: dict[str, str | list[str]] = {
    "boc": "[CH3]C([CH3])([CH3])OC(=O)[#7]",
    "cbz_fmoc": ["[#7]C(=O)O[CH2]c1ccccc1", "[#7]C(=O)O[CH2]C1c2ccccc2-c2ccccc21"],
    "silyl": ["[#8]-[Si]", "[CX2]#[CX2]-[Si]"],
    "acetal_pg": [
        "[#7,#8]-!@[CH1]1OCCCC1",
        "[#7,#8]-!@[CH2]O[CH3]",
        "[#7,#8]-!@[CH2]OCC[Si]",
        "[CX4]1O[CX4][CX4]O1",
        "[CX4]1O[CX4][CX4][CX4]O1",
        "[CX4;!R;!$(C(~[OX2])(~[OX2])~[#7,#8]);!$(C([CH3])[CH3])](-[OX2][CH3,$([CH2][CH3])])-[OX2][CH3,$([CH2][CH3])]",
    ],
    "benzyl": [
        "[#7,#8;!$([#8][CX3]=O);!$([#7,#8]C(=O)O[CH2]c1ccccc1)]-!@[CH2]c1ccccc1",
        "[CX3;!$(C[#7])](=O)O[CH2]c1ccccc1",
        "[#7,#8]-!@[CX4](c1ccccc1)c1ccccc1",
    ],
    "n_sulfonyl": "[#7]-[SX4](=O)(=O)c",
    "thiocarbonyl": "[#6]=[SX1]",
    "acid": "[CX3](=O)[OX2H1,OX1-]",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "anhydride": "[CX3](=O)[OX2][CX3]=O",
    "ester": "[CX3;!$(C(=O)([#7,#8,#16])[#7,#8,#16])](=O)[OX2][#6;!$([CX3]=O)]",
    "amide": "[CX3;!$(C(=O)([#7,#8,#16])[#7,#8,#16])](=O)[#7]",
    "urea": "[#7][CX3](=[O,S])[#7]",
    "carbamate": "[#7][CX3](=O)[OX2][#6]",
    "amidine": "[CX3;!R;!$(C=O)](=[NX2])[#7]",
    "isocyanate": "N=C=[O,S]",
    "aldehyde": "[CX3;H1,H2;!$(C[#7,#8,#16])]=O",
    "ketone": "[#6][CX3](=O)[#6]",
    "aryl_carbonyl": "c-[CX3;!$(C[#7,#8,#16,F,Cl,Br,I])]=O",
    "nitrile": "[CX2]#[NX1]",
    "aryl_nitrile": "c-[CX2]#[NX1]",
    "imine": "[CX3]=[NX2;!R]",
    "sulfonamide": "[SX4](=O)(=O)[#7]",
    "sulfonate_ester": "[SX4](=O)(=O)[OX2][#6]",
    "aryl_N": "c-[#7]",
    "aryl_O": f"c-{ETHER_O}",
    "aryl_S": "c-[SX2]",
    "alkyl_N": "[CX4]-[#7]",
    "alkyl_O": f"[CX4]-{ETHER_O}",
    "alkyl_S": "[CX4]-[SX2]",
    "aryl_vinyl": "[c,$([CX3]=[CX3])]-[CX3]=[CX3]",
    "aryl_alkyne": "[c,$([CX3]=[CX3])]-[CX2]#[CX2]",
    "terminal_alkene": "[CH2]=[CX3]",
    "arx_activated": [
        f"[c;{ACTIVATED}]-[Cl,Br,I,$([SX4](=O)(=O)[CH3]),$([SX3](=O)[CH3]),$([SX2][CH3]),$([SX2H1]),$([N+](=O)[O-])]",
        "c-F",
    ],
    "arx": [f"[c;{UNACTIVATED}]-[Cl,Br,I]", f"c-{TRIFLATE}", f"[CX3]=[CX3]-[Br,I,$({TRIFLATE})]"],
    "alkyl_x": ["[CX4;!$(C([Cl,Br,I])[Cl,Br,I])]-[Cl,Br,I]", "[CX4]-OS(=O)(=O)[#6,$([OX2][CX4])]"],
    "epoxide": "[CX4]1[O,N][CX4]1",
    "michael": "[CX3]=[CX3]-[$([CX3]=O),$(C#N),$([SX4](=O)=O),$([N+](=O)[O-])]",
    "boron": "[#6]-[#5]",
    "tin_zinc": "[#6]-[Sn,Zn]",
    "organometal": "[#6]-[Mg,Li,Zn,Cu]",
    "alcohol": "[CX4]-[OX2H]",
    "phenol": "c-[OX2H]",
    "oxo_het": "c=[OX1]",
    "c_x": "[#6]-[F,Cl,Br,I]",
    "nitro": "[#6]-[#7+](=O)[O-]",
    "azide": "[#6]-N=[N+]=[N-]",
    "nh2": "[NX3;H2;!$(N[C,S]=[O,S,N])]-[#6]",
    "acyl_pg_ester": ["[CH3][CX3](=O)[OX2][#6]", "[CH3]C([CH3])([CH3])[CX3](=O)[OX2][CX4]"],
    "cc_double": "[CX3]=[CX3]",
    "cc_triple": "[CX2]#[CX2]",
    "s_ox": "[#16]=[OX1]",
    "n_oxide": "[#7+;!$([#7+]=O)]-[OX1-]",
    "phosphorus": "[#15]",
    "n_h": ["[#7;H1,H2,H3]", "[#7;H2,H3]", "[#7;H3]"],
}
_PATTERNS = {
    name: [Chem.MolFromSmarts(s) for s in ([smarts] if isinstance(smarts, str) else smarts)]
    for name, smarts in GROUPS.items()
}
AROMATIC, HETERO_AROMATIC, CARBON = (Chem.MolFromSmarts(s) for s in ("[a]", "[a;!#6]", "[#6]"))
# Halogen sources such as CBr4 or CHCl3 must not count as substrate C-X bonds.
SUBSTRATE_ONLY = {"c_x"}
REAGENT_MAX_HEAVY = 5


class _Side:
    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        frags = Chem.GetMolFrags(mol)
        self._reagent_atoms = frozenset(
            i for frag in frags if len(frags) > 1 and len(frag) <= REAGENT_MAX_HEAVY for i in frag
        )

    def count(self, name: str) -> int:
        skip = self._reagent_atoms if name in SUBSTRATE_ONLY else frozenset()
        return sum(
            match[0] not in skip
            for pattern in _PATTERNS[name]
            for match in self.mol.GetSubstructMatches(pattern, uniquify=True, maxMatches=10000)
        )

    def scalar(self) -> dict[str, int]:
        def atoms(pattern: Chem.Mol) -> set[int]:
            return {m[0] for m in self.mol.GetSubstructMatches(pattern, maxMatches=100000)}

        aromatic, hetero, carbons = atoms(AROMATIC), atoms(HETERO_AROMATIC), atoms(CARBON)
        rings = self.mol.GetRingInfo().AtomRings()
        return {
            "rings": rdMolDescriptors.CalcNumRings(self.mol),
            "het_aromatic_rings": sum(set(ring) <= aromatic and not hetero.isdisjoint(ring) for ring in rings),
            "aromatic_atoms": len(aromatic),
            "heavy": self.mol.GetNumHeavyAtoms(),
            "largest_carbon_piece": max(len(carbons.intersection(f)) for f in Chem.GetMolFrags(self.mol)),
        }


class _Delta:
    """Product-minus-reactants counts, computed lazily per functional group."""

    def __init__(self, product: _Side, reactants: _Side):
        self._product, self._reactants = product, reactants
        product_scalar, reactant_scalar = product.scalar(), reactants.scalar()
        self._cache = {key: (product_scalar[key], reactant_scalar[key]) for key in product_scalar}

    def __getattr__(self, name: str) -> int:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = (self._product.count(name), self._reactants.count(name))
        product, reactants = self._cache[name]
        return product - reactants


def _lost(d: _Delta, *names: str) -> bool:
    return any(getattr(d, n) < 0 for n in names)


def _gained(d: _Delta, *names: str) -> bool:
    return any(getattr(d, n) > 0 for n in names)


def _aryl_lg_lost(d: _Delta) -> bool:
    return d.arx + d.arx_activated < 0


ACYL_DONOR = ("acid", "acyl_halide", "ester", "anhydride")
CARBONYL = ("aldehyde", "ketone")
OH = ("alcohol", "phenol", "oxo_het", "acid")
HETERO_ALKYL = ("alkyl_N", "alkyl_O", "alkyl_S")
ARYL_HETERO = ("aryl_N", "aryl_O", "aryl_S")

# (rule, confidence, predicate); the first matching rule wins.
RULES: list[tuple[str, str, Callable[[_Delta], bool]]] = [
    (
        "heteroaromatic ring formation",
        "medium",
        lambda d: (
            (d.rings > max(d.acetal_pg, 0) or (d.rings == 0 and d.largest_carbon_piece > 0))
            and d.het_aromatic_rings > 0
        ),
    ),
    ("cyclisation", "medium", lambda d: d.rings > max(d.acetal_pg, 0)),
    ("Boc deprotection", "high", lambda d: d.boc < 0),
    ("Boc protection", "high", lambda d: d.boc > 0),
    ("Cbz/Fmoc deprotection", "high", lambda d: d.cbz_fmoc < 0),
    ("Cbz/Fmoc protection", "high", lambda d: d.cbz_fmoc > 0),
    ("acetal/THP/MOM/SEM deprotection", "high", lambda d: d.acetal_pg < 0),
    ("acetal/THP/MOM/SEM protection", "high", lambda d: d.acetal_pg > 0),
    ("silyl deprotection", "high", lambda d: d.silyl < 0),
    ("silyl protection", "high", lambda d: d.silyl > 0 and not _aryl_lg_lost(d)),
    ("benzyl deprotection", "high", lambda d: d.benzyl < 0),
    ("sulfonyl (Ts/Ns) deprotection", "medium", lambda d: d.n_sulfonyl < 0 and d.n_h > 0),
    ("Chan-Lam coupling", "medium", lambda d: d.boron < 0 and _gained(d, "aryl_N", "aryl_O") and not _aryl_lg_lost(d)),
    ("Suzuki coupling", "high", lambda d: d.boron < 0 and (_aryl_lg_lost(d) or _lost(d, "alkyl_x"))),
    ("borylation", "high", lambda d: d.boron > 0),
    ("Sonogashira coupling", "high", lambda d: d.aryl_alkyne > 0 and _aryl_lg_lost(d)),
    (
        "Heck reaction",
        "medium",
        lambda d: d.aryl_vinyl > 0 and _aryl_lg_lost(d) and _lost(d, "terminal_alkene", "michael"),
    ),
    ("cyanation", "medium", lambda d: d.aryl_nitrile > 0 and _aryl_lg_lost(d)),
    (
        "Stille/Negishi/Kumada coupling",
        "medium",
        lambda d: (
            _lost(d, "tin_zinc", "organometal")
            and _aryl_lg_lost(d)
            and not _lost(d, *CARBONYL, "nitrile", "ester", "amide")
        ),
    ),
    ("sulfonamide formation", "high", lambda d: d.sulfonamide > 0),
    ("O-sulfonylation", "high", lambda d: d.sulfonate_ester > 0),
    (
        "urea formation",
        "high",
        lambda d: d.urea > 0 and (d.n_h < 0 or _lost(d, "isocyanate", "acyl_halide", "carbamate", "acid")),
    ),
    (
        "carbamate formation",
        "high",
        lambda d: d.carbamate > 0 and (d.n_h < 0 or _lost(d, "isocyanate", "acyl_halide", "alcohol", "phenol", "acid")),
    ),
    ("amide coupling", "high", lambda d: d.amide > 0 and (d.n_h < 0 or _lost(d, *ACYL_DONOR))),
    ("esterification", "high", lambda d: d.ester > 0 and _lost(d, *ACYL_DONOR, "alcohol", "phenol")),
    ("amidine/guanidine formation", "medium", lambda d: d.amidine > 0),
    ("SNAr", "high", lambda d: _gained(d, *ARYL_HETERO) and _aryl_lg_lost(d) and d.arx_activated < 0),
    ("Buchwald-Hartwig/Ullmann coupling", "medium", lambda d: _gained(d, *ARYL_HETERO) and _aryl_lg_lost(d)),
    ("reductive amination", "high", lambda d: d.alkyl_N > 0 and _lost(d, *CARBONYL) and not _lost(d, "amide")),
    ("Michael addition", "medium", lambda d: _lost(d, "michael") and d.largest_carbon_piece > 0),
    ("N-alkylation", "high", lambda d: d.alkyl_N > 0 and _lost(d, "alkyl_x", "epoxide")),
    ("O-alkylation", "high", lambda d: d.alkyl_O > 0 and _lost(d, "alkyl_x", "epoxide")),
    ("S-alkylation", "high", lambda d: d.alkyl_S > 0 and _lost(d, "alkyl_x", "epoxide")),
    (
        "Mitsunobu / dehydrative alkylation",
        "medium",
        lambda d: _gained(d, *HETERO_ALKYL) and _lost(d, "alcohol") and not _lost(d, "amide", *ACYL_DONOR),
    ),
    (
        "Grignard/organolithium addition",
        "high",
        lambda d: _lost(d, "organometal") and _lost(d, *CARBONYL, "nitrile", "ester", "amide", "acyl_halide"),
    ),
    ("Wittig/HWE olefination", "high", lambda d: d.phosphorus < 0 and d.cc_double > 0),
    (
        "halogen-metal exchange addition",
        "low",
        lambda d: (
            _aryl_lg_lost(d)
            and d.largest_carbon_piece > 0
            and _lost(d, *CARBONYL, "ester", "amide", "nitrile")
            and not _gained(d, *ARYL_HETERO)
        ),
    ),
    (
        "other C-C coupling (aryl halide)",
        "low",
        lambda d: _aryl_lg_lost(d) and d.largest_carbon_piece > 0 and not _gained(d, *ARYL_HETERO, *HETERO_ALKYL),
    ),
    (
        "Friedel-Crafts acylation/formylation",
        "medium",
        lambda d: (
            d.aryl_carbonyl > 0 and _lost(d, "acyl_halide", "anhydride", "acid", "amide") and not _aryl_lg_lost(d)
        ),
    ),
    (
        "C-alkylation",
        "medium",
        lambda d: (
            _lost(d, "alkyl_x") and (d.largest_carbon_piece > 0 or d.nitrile > 0) and not _gained(d, *HETERO_ALKYL)
        ),
    ),
    (
        "aldol/Claisen/Knoevenagel",
        "low",
        lambda d: (
            d.largest_carbon_piece > 0
            and _lost(d, *CARBONYL, "ester", "nitrile", "acyl_halide")
            and not _gained(d, "alkyl_N", "alkyl_O", "aryl_N", "aryl_O", "imine", "c_x")
        ),
    ),
    ("Curtius rearrangement", "medium", lambda d: d.acid < 0 and d.nh2 > 0 and not _gained(d, "amide")),
    ("nitro reduction", "high", lambda d: d.nitro < 0 and d.n_h > 0),
    ("nitrile/azide reduction", "high", lambda d: _lost(d, "nitrile", "azide") and d.nh2 > 0),
    (
        "ester hydrolysis",
        "high",
        lambda d: (d.ester < 0 and d.acid > 0) or (d.acyl_pg_ester < 0 and _gained(d, "alcohol", "phenol")),
    ),
    (
        "amide/nitrile hydrolysis",
        "medium",
        lambda d: (
            (d.nitrile < 0 and _gained(d, "amide", "acid"))
            or (_lost(d, "amide", "carbamate", "urea") and (d.acid > 0 or d.n_h > 0) and d.alkyl_N <= 0)
        ),
    ),
    (
        "ester/acid reduction",
        "high",
        lambda d: _lost(d, "ester", "acid", "amide", "nitrile") and _gained(d, "alcohol", "aldehyde"),
    ),
    ("amide reduction", "medium", lambda d: d.amide < 0 and d.alkyl_N > 0),
    ("ketone/aldehyde reduction", "high", lambda d: _lost(d, *CARBONYL) and d.alcohol > 0),
    ("ether cleavage (O-dealkylation)", "medium", lambda d: d.alkyl_O < 0 and _gained(d, "phenol", "alcohol")),
    ("deoxyhalogenation", "high", lambda d: _lost(d, *OH) and d.c_x > 0),
    ("Sandmeyer/diazotisation", "medium", lambda d: d.nh2 < 0 and _gained(d, "c_x", "aryl_nitrile", "phenol")),
    ("halogenation", "medium", lambda d: d.c_x > 0),
    ("nitration", "high", lambda d: d.nitro > 0),
    ("thionation", "medium", lambda d: d.thiocarbonyl > 0),
    (
        "oxidation",
        "medium",
        lambda d: (
            (d.alcohol < 0 and _gained(d, *CARBONYL, "acid"))
            or (d.aldehyde < 0 and d.acid > 0)
            or _gained(d, "s_ox", "n_oxide", "epoxide")
            or (d.boron < 0 and d.phenol > 0)
            or (d.alkyl_x < 0 and _gained(d, *CARBONYL))
            or (d.cc_double < 0 and d.alcohol > 0)
            or (d.aromatic_atoms > 0 and d.rings == 0)
        ),
    ),
    ("imine/oxime/hydrazone formation", "medium", lambda d: _lost(d, *CARBONYL) and d.imine > 0),
    ("dehydration to nitrile", "medium", lambda d: d.nitrile > 0 and _lost(d, "amide", "imine", "aldehyde")),
    (
        "hydrogenation",
        "medium",
        lambda d: d.heavy == 0 and (_lost(d, "cc_double", "cc_triple", "imine") or d.aromatic_atoms < 0),
    ),
    ("dehalogenation", "medium", lambda d: d.c_x < 0 and d.largest_carbon_piece <= 0),
    ("deoxygenation", "low", lambda d: _lost(d, "alcohol", *CARBONYL) and d.heavy < 0),
]

PROTECTION = "protection/deprotection"
COUPLING = "metal-catalysed coupling"
ACYLATION = "acylation/sulfonylation"
HETERO = "heteroatom alkylation/arylation"
CC = "C-C bond formation"
REDUCTION = "reduction"
CLEAVAGE = "hydrolysis/cleavage"
HALO = "halogenation/nitration"
RING = "ring formation"
CONDENSATION = "condensation/dehydration"
OTHER_FGI = "other FGI"

OTHER_PROTECTION = "other protection (Cbz/Fmoc/silyl/acetal)"
OTHER_DEPROTECTION = "other deprotection (Cbz/Fmoc/silyl/acetal/sulfonyl)"
OTHER_CROSS = "other cross-coupling (Heck/Stille/Negishi/cyanation/borylation)"
ENOLATE = "enolate/Michael chemistry"
CARBONYL_REDUCTION = "carbonyl/nitrile/azide reduction"
CONDENSATIONS = "condensation (imine/oxime/amidine/guanidine)"
OTHER_FGI_CLASS = "other FGI (dehydration/thionation/Curtius)"
ARYL_AMINATION = "Buchwald-Hartwig/Ullmann/Chan-Lam coupling"

# rule -> (agent-facing class, family)
TAXONOMY: dict[str, tuple[str, str]] = {
    "heteroaromatic ring formation": ("heteroaromatic ring formation", RING),
    "cyclisation": ("cyclisation (non-aromatic ring)", RING),
    "Boc deprotection": ("Boc deprotection", PROTECTION),
    "Boc protection": ("Boc protection", PROTECTION),
    "Cbz/Fmoc deprotection": (OTHER_DEPROTECTION, PROTECTION),
    "Cbz/Fmoc protection": (OTHER_PROTECTION, PROTECTION),
    "acetal/THP/MOM/SEM deprotection": (OTHER_DEPROTECTION, PROTECTION),
    "acetal/THP/MOM/SEM protection": (OTHER_PROTECTION, PROTECTION),
    "silyl deprotection": (OTHER_DEPROTECTION, PROTECTION),
    "silyl protection": (OTHER_PROTECTION, PROTECTION),
    "benzyl deprotection": ("benzyl deprotection", PROTECTION),
    "sulfonyl (Ts/Ns) deprotection": (OTHER_DEPROTECTION, PROTECTION),
    "Chan-Lam coupling": (ARYL_AMINATION, COUPLING),
    "Suzuki coupling": ("Suzuki coupling", COUPLING),
    "borylation": (OTHER_CROSS, COUPLING),
    "Sonogashira coupling": ("Sonogashira coupling", COUPLING),
    "Heck reaction": (OTHER_CROSS, COUPLING),
    "cyanation": (OTHER_CROSS, COUPLING),
    "Stille/Negishi/Kumada coupling": (OTHER_CROSS, COUPLING),
    "other C-C coupling (aryl halide)": (OTHER_CROSS, COUPLING),
    "Buchwald-Hartwig/Ullmann coupling": (ARYL_AMINATION, COUPLING),
    "sulfonamide formation": ("sulfonamide formation", ACYLATION),
    "O-sulfonylation": ("O-sulfonylation (mesylate/tosylate/triflate)", ACYLATION),
    "urea formation": ("urea/carbamate formation", ACYLATION),
    "carbamate formation": ("urea/carbamate formation", ACYLATION),
    "amide coupling": ("amide coupling", ACYLATION),
    "esterification": ("esterification", ACYLATION),
    "amidine/guanidine formation": (CONDENSATIONS, CONDENSATION),
    "SNAr": ("SNAr", HETERO),
    "reductive amination": ("reductive amination", HETERO),
    "Michael addition": (ENOLATE, CC),
    "N-alkylation": ("N-alkylation", HETERO),
    "O-alkylation": ("O-/S-alkylation", HETERO),
    "S-alkylation": ("O-/S-alkylation", HETERO),
    "Mitsunobu / dehydrative alkylation": ("Mitsunobu / dehydrative alkylation", HETERO),
    "Grignard/organolithium addition": ("Grignard/organolithium addition", CC),
    "halogen-metal exchange addition": ("Grignard/organolithium addition", CC),
    "Wittig/HWE olefination": ("Wittig/HWE olefination", CC),
    "Friedel-Crafts acylation/formylation": ("Friedel-Crafts acylation/formylation", CC),
    "C-alkylation": (ENOLATE, CC),
    "aldol/Claisen/Knoevenagel": (ENOLATE, CC),
    "Curtius rearrangement": (OTHER_FGI_CLASS, OTHER_FGI),
    "nitro reduction": ("nitro reduction", REDUCTION),
    "nitrile/azide reduction": (CARBONYL_REDUCTION, REDUCTION),
    "ester hydrolysis": ("ester hydrolysis", CLEAVAGE),
    "amide/nitrile hydrolysis": ("amide/nitrile hydrolysis", CLEAVAGE),
    "ester/acid reduction": (CARBONYL_REDUCTION, REDUCTION),
    "amide reduction": (CARBONYL_REDUCTION, REDUCTION),
    "ketone/aldehyde reduction": (CARBONYL_REDUCTION, REDUCTION),
    "ether cleavage (O-dealkylation)": ("ether cleavage (O-dealkylation)", CLEAVAGE),
    "deoxyhalogenation": ("halogenation/nitration", HALO),
    "Sandmeyer/diazotisation": ("halogenation/nitration", HALO),
    "halogenation": ("halogenation/nitration", HALO),
    "nitration": ("halogenation/nitration", HALO),
    "thionation": (OTHER_FGI_CLASS, OTHER_FGI),
    "oxidation": ("oxidation", "oxidation"),
    "imine/oxime/hydrazone formation": (CONDENSATIONS, CONDENSATION),
    "dehydration to nitrile": (OTHER_FGI_CLASS, CONDENSATION),
    "hydrogenation": ("hydrogenation (C=C/C#C/arene)", REDUCTION),
    "dehalogenation": ("other reduction (dehalogenation/deoxygenation)", REDUCTION),
    "deoxygenation": ("other reduction (dehalogenation/deoxygenation)", REDUCTION),
    "other": ("other", "other"),
}
REACTION_CLASSES: tuple[str, ...] = tuple(sorted({name for name, _ in TAXONOMY.values()}))
# Classes whose labels matched reagent evidence well enough to be enforced as constraints.
CONSTRAINABLE_CLASSES: tuple[str, ...] = (
    "amide coupling",
    "Suzuki coupling",
    "Sonogashira coupling",
    ARYL_AMINATION,
    "SNAr",
    "reductive amination",
    "nitro reduction",
    "Boc protection",
    "Boc deprotection",
    "ester hydrolysis",
    "sulfonamide formation",
    "urea/carbamate formation",
    "esterification",
)


@dataclass(frozen=True)
class ReactionClass:
    name: str
    family: str
    rule: str
    confidence: str

    def to_dict(self) -> dict[str, str]:
        return {"reaction_class": self.name, "family": self.family, "rule": self.rule, "confidence": self.confidence}


OTHER = ReactionClass("other", "other", "other", "none")


@lru_cache(maxsize=200_000)
def _classify(product: str, reactants: tuple[str, ...]) -> ReactionClass:
    product_mol = Chem.MolFromSmiles(product)
    reactant_mol = Chem.MolFromSmiles(".".join(reactants))
    if product_mol is None or reactant_mol is None or not reactant_mol.GetNumAtoms():
        return OTHER
    delta = _Delta(_Side(product_mol), _Side(reactant_mol))
    for rule, confidence, predicate in RULES:
        if predicate(delta):
            name, family = TAXONOMY[rule]
            return ReactionClass(name, family, rule, confidence)
    return OTHER


def classify_step(product: str, reactants: Iterable[str]) -> ReactionClass:
    """Classify one step from plain SMILES; never raises."""
    try:
        return _classify(product, tuple(sorted(reactants)))
    except Exception:
        return OTHER
