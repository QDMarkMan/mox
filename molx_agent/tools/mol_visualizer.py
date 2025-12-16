"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-16].
*  @Description Molecule visualization utilities using RDKit.
**************************************************************************
"""

import base64
import logging
from io import BytesIO
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


def smiles_to_svg(
    smiles: str,
    width: int = 250,
    height: int = 180,
    highlight_atoms: Optional[list[int]] = None,
    highlight_bonds: Optional[list[int]] = None,
) -> str:
    """Generate an inline SVG image from a SMILES string.

    Args:
        smiles: SMILES string of the molecule.
        width: Width of the SVG in pixels.
        height: Height of the SVG in pixels.
        highlight_atoms: Optional list of atom indices to highlight.
        highlight_bonds: Optional list of bond indices to highlight.

    Returns:
        SVG string that can be embedded directly in HTML.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _placeholder_svg(width, height, "Invalid SMILES")

        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        
        # Set drawing options
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.addAtomIndices = False
        opts.bondLineWidth = 2.0

        # Draw with or without highlights
        if highlight_atoms or highlight_bonds:
            colors = {}
            if highlight_atoms:
                for idx in highlight_atoms:
                    colors[idx] = (0.3, 0.7, 0.3, 0.5)  # Green highlight
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms or [],
                highlightBonds=highlight_bonds or [],
                highlightAtomColors=colors if highlight_atoms else {},
            )
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        # Clean up SVG for inline use
        svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
        svg = svg.replace("\n", " ")

        return svg

    except Exception as e:
        logger.warning(f"Failed to generate SVG for {smiles[:30]}: {e}")
        return _placeholder_svg(width, height, "Error")


def smiles_to_png_base64(smiles: str, width: int = 250, height: int = 180) -> str:
    """Generate a base64-encoded PNG image from a SMILES string.

    Args:
        smiles: SMILES string of the molecule.
        width: Width of the image in pixels.
        height: Height of the image in pixels.

    Returns:
        Base64-encoded PNG data URL for embedding in HTML.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(width, height))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        logger.warning(f"Failed to generate PNG for {smiles[:30]}: {e}")
        return ""


def generate_mol_grid_svg(
    smiles_list: list[str],
    legends: Optional[list[str]] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (200, 150),
) -> str:
    """Generate a grid of molecule SVG images.

    Args:
        smiles_list: List of SMILES strings.
        legends: Optional list of labels for each molecule.
        mols_per_row: Number of molecules per row.
        sub_img_size: Size of each molecule image as (width, height).

    Returns:
        SVG string containing the molecule grid.
    """
    if not smiles_list:
        return ""

    try:
        mols = []
        valid_legends = []

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
                if legends and i < len(legends):
                    valid_legends.append(legends[i])
                else:
                    valid_legends.append("")

        if not mols:
            return ""

        # Calculate grid dimensions
        n_rows = (len(mols) + mols_per_row - 1) // mols_per_row
        grid_width = mols_per_row * sub_img_size[0]
        grid_height = n_rows * sub_img_size[1]

        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DSVG(grid_width, grid_height, sub_img_size[0], sub_img_size[1])
        drawer.DrawMolecules(mols, legends=valid_legends if any(valid_legends) else None)
        drawer.FinishDrawing()

        svg = drawer.GetDrawingText()
        svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")

        return svg

    except Exception as e:
        logger.warning(f"Failed to generate molecule grid: {e}")
        return ""


def highlight_substructure_svg(
    smiles: str,
    pattern_smarts: str,
    width: int = 250,
    height: int = 180,
) -> str:
    """Generate an SVG with a substructure pattern highlighted.

    Args:
        smiles: SMILES string of the molecule.
        pattern_smarts: SMARTS pattern to highlight.
        width: Width of the SVG.
        height: Height of the SVG.

    Returns:
        SVG string with highlighted substructure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _placeholder_svg(width, height, "Invalid SMILES")

        pattern = Chem.MolFromSmarts(pattern_smarts)
        if pattern is None:
            return smiles_to_svg(smiles, width, height)

        match = mol.GetSubstructMatch(pattern)
        if not match:
            return smiles_to_svg(smiles, width, height)

        # Get bonds in the match
        highlight_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                highlight_bonds.append(bond.GetIdx())

        return smiles_to_svg(
            smiles,
            width,
            height,
            highlight_atoms=list(match),
            highlight_bonds=highlight_bonds,
        )

    except Exception as e:
        logger.warning(f"Failed to highlight substructure: {e}")
        return smiles_to_svg(smiles, width, height)


def _placeholder_svg(width: int, height: int, text: str = "?") -> str:
    """Generate a placeholder SVG for invalid molecules."""
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f3f4f6" rx="8"/>
        <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" 
              fill="#9ca3af" font-family="sans-serif" font-size="12">{text}</text>
    </svg>'''


def get_scaffold_smiles(smiles: str) -> Optional[str]:
    """Extract the Murcko scaffold from a molecule.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        SMILES of the Murcko scaffold, or None if extraction fails.
    """
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)

    except Exception:
        return None


def get_functional_groups(smiles: str) -> list[dict]:
    """Identify functional groups in a molecule.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        List of dicts with functional group info.
    """
    FG_PATTERNS = {
        "hydroxyl": "[OX2H]",
        "methoxy": "[OX2][CH3]",
        "amino": "[NX3;H2,H1;!$(NC=O)]",
        "amide": "[NX3][CX3](=[OX1])",
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "ester": "[CX3](=O)[OX2H0]",
        "ketone": "[CX3](=O)[#6]",
        "halogen": "[F,Cl,Br,I]",
        "nitro": "[NX3+](=O)[O-]",
        "cyano": "[CX2]#[NX1]",
        "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
        "aromatic": "a",
    }

    found = []

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        for fg_name, smarts in FG_PATTERNS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                found.append({
                    "name": fg_name,
                    "count": len(matches),
                    "atom_indices": [list(m) for m in matches],
                })

    except Exception:
        pass

    return found
