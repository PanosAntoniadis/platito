from typing import Dict

MAX_RES_ID = 23

AA_3_TO_1: Dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "HSP": "H",
    "HSD": "H",
    "HSE": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PYL": "O",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "SEC": "U",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}


AA_1_TO_ID: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "U": 18,
    "V": 19,
    "W": 20,
    "Y": 21,
    "X": 22,  # "X" = unknown
}


AA_3_TO_ID: Dict[str, int] = {
    aa3: AA_1_TO_ID[aa1] for aa3, aa1 in AA_3_TO_1.items()
}
