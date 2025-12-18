"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-18].
*  @Description Example MCP server for chemistry-related tools.
*               Demonstrates how to create custom MCP tools.
**************************************************************************

Usage:
    Run directly: python examples/example_mcp_server.py
    
    Or configure in config/mcp_servers.json:
    {
        "chemistry": {
            "command": "python",
            "args": ["./examples/example_mcp_server.py"],
            "transport": "stdio"
        }
    }
"""

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("MCP not installed. Run: pip install mcp")
    exit(1)

# Create an MCP server instance
mcp = FastMCP("ChemistryTools")


@mcp.tool()
def calculate_molecular_weight(smiles: str) -> dict:
    """Calculate molecular weight from SMILES string.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Dictionary with molecular weight and formula
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}
        
        mw = Descriptors.MolWt(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        
        return {
            "smiles": smiles,
            "molecular_weight": round(mw, 2),
            "formula": formula,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def count_atoms(smiles: str) -> dict:
    """Count atoms in a molecule by element.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Dictionary with atom counts by element
    """
    try:
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}
        
        # Count atoms
        atom_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        
        # Add hydrogens
        mol_with_h = Chem.AddHs(mol)
        h_count = sum(1 for a in mol_with_h.GetAtoms() if a.GetSymbol() == "H")
        if h_count > 0:
            atom_counts["H"] = h_count
            
        return {
            "smiles": smiles,
            "atom_counts": atom_counts,
            "total_atoms": sum(atom_counts.values()),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def check_drug_likeness(smiles: str) -> dict:
    """Check Lipinski's Rule of 5 for drug-likeness.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Dictionary with drug-likeness evaluation
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        violations = 0
        rules = []
        
        if mw > 500:
            violations += 1
            rules.append(f"MW ({mw:.1f}) > 500")
        else:
            rules.append(f"MW ({mw:.1f}) ≤ 500 ✓")
            
        if logp > 5:
            violations += 1
            rules.append(f"LogP ({logp:.2f}) > 5")
        else:
            rules.append(f"LogP ({logp:.2f}) ≤ 5 ✓")
            
        if hbd > 5:
            violations += 1
            rules.append(f"HBD ({hbd}) > 5")
        else:
            rules.append(f"HBD ({hbd}) ≤ 5 ✓")
            
        if hba > 10:
            violations += 1
            rules.append(f"HBA ({hba}) > 10")
        else:
            rules.append(f"HBA ({hba}) ≤ 10 ✓")
        
        return {
            "smiles": smiles,
            "drug_like": violations == 0,
            "violations": violations,
            "properties": {
                "molecular_weight": round(mw, 2),
                "logP": round(logp, 2),
                "h_bond_donors": hbd,
                "h_bond_acceptors": hba,
            },
            "rules_check": rules,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_functional_groups(smiles: str) -> dict:
    """Identify functional groups in a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Dictionary with identified functional groups
    """
    try:
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}
        
        # Common functional group patterns
        patterns = {
            "alcohol": "[OX2H]",
            "aldehyde": "[CX3H1](=O)[#6]",
            "ketone": "[#6][CX3](=O)[#6]",
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "ester": "[#6][CX3](=O)[OX2H0][#6]",
            "amine_primary": "[NX3;H2;!$(NC=O)]",
            "amine_secondary": "[NX3;H1;!$(NC=O)]",
            "amine_tertiary": "[NX3;H0;!$(NC=O)]",
            "amide": "[NX3][CX3](=[OX1])[#6]",
            "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
            "halide": "[F,Cl,Br,I]",
            "nitrile": "[NX1]#[CX2]",
            "sulfoxide": "[SX3](=O)[#6]",
            "sulfone": "[SX4](=O)(=O)[#6]",
            "thiol": "[#16X2H]",
            "ether": "[OD2]([#6])[#6]",
            "phenol": "[OX2H][cX3]:[c]",
            "aromatic_ring": "c1ccccc1",
        }
        
        found_groups = []
        for name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = len(mol.GetSubstructMatches(pattern))
                found_groups.append({"name": name, "count": matches})
        
        return {
            "smiles": smiles,
            "functional_groups": found_groups,
            "total_groups": len(found_groups),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the MCP server with stdio transport
    mcp.run(transport="stdio")
