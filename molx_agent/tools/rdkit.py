"""RDKit-based chemistry tools."""

from typing import Any, Optional

from langchain_core.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from molx_agent.tools.utils import tanimoto


class MolSimilarity(BaseTool):
    """Calculate Tanimoto similarity between two molecules."""

    name: str = "MolSimilarity"
    description: str = (
        "Input two molecule SMILES (separated by '.'), returns Tanimoto similarity."
    )

    def _run(self, smiles_pair: str) -> str:
        smi_list = smiles_pair.split(".")
        if len(smi_list) != 2:
            return "Input error, please input two smiles strings separated by '.'"
        else:
            smiles1, smiles2 = smi_list

        similarity = tanimoto(smiles1, smiles2)

        if isinstance(similarity, str):
            return similarity

        sim_score = {
            0.9: "very similar",
            0.8: "similar",
            0.7: "somewhat similar",
            0.6: "not very similar",
            0: "not similar",
        }
        if similarity == 1:
            return "Error: Input Molecules Are Identical"
        else:
            val = sim_score[
                max(key for key in sim_score.keys() if key <= round(similarity, 1))
            ]
            message = (
                f"The Tanimoto similarity between {smiles1} and {smiles2} is "
                f"{round(similarity, 4)}, indicating that the two molecules "
                f"are {val}."
            )
        return message


class SMILES2Weight(BaseTool):
    """Convert SMILES to molecular weight."""

    name: str = "SMILES2Weight"
    description: str = "Input SMILES, returns molecular weight."

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        return str(mol_weight)


class FuncGroups(BaseTool):
    """Identify functional groups in a molecule."""

    name: str = "FunctionalGroups"
    description: str = (
        "Input SMILES, return list of functional groups in the molecule."
    )
    dict_fgs: Optional[dict[str, str]] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dict_fgs = {
            "furan": "o1cccc1",
            "aldehydes": " [CX3H1](=O)[#6]",
            "esters": " [#6][CX3](=O)[OX2H0][#6]",
            "ketones": " [#6][CX3](=O)[#6]",
            "amides": " C(=O)-N",
            "thiol groups": " [SH]",
            "alcohol groups": " [OH]",
            "carboxylic acids": "*-C(=O)[O;D1]",
            "nitro": "*-[N;D3](=[O;D1])[O;D1]",
            "cyano": "*-[C;D2]#[N;D1]",
            "halogens": "*-[#9,#17,#35,#53]",
            "methoxy": "*-[O;D2]-[C;D1;H3]",
            "primary amines": "*-[N;D1]",
        }

    def _is_fg_in_mol(self, mol_smiles: str, fg: str) -> bool:
        fgmol = Chem.MolFromSmarts(fg)
        mol = Chem.MolFromSmiles(mol_smiles.strip())
        if mol is None or fgmol is None:
            return False
        return len(Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True)) > 0

    def _run(self, smiles: str) -> str:
        try:
            fgs_in_molec = [
                name
                for name, fg in self.dict_fgs.items()
                if self._is_fg_in_mol(smiles, fg)
            ]
            if len(fgs_in_molec) > 1:
                return (
                    f"This molecule contains {', '.join(fgs_in_molec[:-1])}, "
                    f"and {fgs_in_molec[-1]}."
                )
            elif len(fgs_in_molec) == 1:
                return f"This molecule contains {fgs_in_molec[0]}."
            else:
                return "No common functional groups detected."
        except Exception:
            return "Wrong argument. Please input a valid molecular SMILES."
