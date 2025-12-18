"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-15].
*  @Description HTML builder for SAR reports - templates and rendering.
**************************************************************************
"""

import os
from datetime import datetime
from typing import Any

REPORT_DIR = os.path.join(os.getcwd(), "output", "reports")

# =============================================================================
# CSS Styles
# =============================================================================

SAR_REPORT_CSS = """
:root {
    --primary: #0f172a; /* Slate 900 - Deep, Trustworthy */
    --primary-light: #334155; /* Slate 700 */
    --accent: #0ea5e9; /* Sky 500 - Focused, Modern */
    --accent-soft: #e0f2fe; /* Sky 50 */
    --success: #059669; /* Emerald 600 */
    --warning: #d97706; /* Amber 600 */
    --danger: #dc2626; /* Red 600 */
    --background: #f8fafc; /* Slate 50 */
    --surface: #ffffff;
    --text-main: #334155; /* Slate 700 - Softer than black */
    --text-strong: #0f172a; /* Slate 900 */
    --text-muted: #64748b; /* Slate 500 */
    --border: #e2e8f0; /* Slate 200 */
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
    --radius: 0.5rem;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: var(--background);
    color: var(--text-main);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

.container {
    max-width: 1100px; /* Slightly narrower for better reading focus */
    margin: 0 auto;
    padding: 3rem 2rem;
}

/* Header - Minimalist & Editorial */
.header {
    background: transparent;
    color: var(--text-strong);
    padding: 0 0 2rem 0;
    margin-bottom: 3rem;
    border-bottom: 1px solid var(--border);
    box-shadow: none;
    border-radius: 0;
    position: relative;
}

.header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 60px;
    height: 3px;
    background: var(--accent);
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--primary);
    margin-bottom: 0.75rem;
}

.header .meta {
    display: flex;
    gap: 2rem;
    font-size: 0.875rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

/* Cards - Clean & Focused */
.card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

.card h2 {
    color: var(--primary);
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    letter-spacing: -0.01em;
}

.card h3 {
    color: var(--text-strong);
    font-size: 1rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem;
}

/* Tables - Professional Data */
.table-container {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: var(--radius);
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

th {
    background: var(--background);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    padding: 1rem 1.5rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

td {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-main);
    vertical-align: middle;
}

tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--background); }

/* Badges - Subtle & Trustworthy */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.125rem 0.625rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid transparent;
}

.badge-essential { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
.badge-beneficial { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
.badge-tolerated { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.badge-detrimental { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1rem;
}

.stat-card {
    background: var(--surface);
    padding: 1.5rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    text-align: left; /* More professional alignment */
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.2;
    margin-bottom: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.05em;
}

.stat-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Molecule Grid */
.mol-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.mol-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1rem;
    border: 1px solid var(--border);
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.mol-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
    border-color: var(--accent);
}

.mol-svg {
    background: white;
    padding: 0.5rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.mol-id {
    font-weight: 600;
    color: var(--text-strong);
    font-size: 0.875rem;
    font-family: 'JetBrains Mono', monospace;
}

.mol-activity {
    color: var(--accent);
    font-weight: 600;
    font-size: 0.75rem;
    margin-top: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
}

/* R-Group Table Specifics */
.rgroup-table .mol-cell { min-width: 180px; }
.rgroup-table .rgroup-cell {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.75rem;
    color: var(--text-strong);
    background: #f8fafc;
}
.rgroup-table .activity-cell {
    font-weight: 600;
    color: var(--primary);
    font-family: 'JetBrains Mono', monospace;
}

/* Conclusion Box - Editorial Style */
.conclusion {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    padding: 1.25rem 1.5rem;
    margin-top: 1.5rem;
    border-radius: 0 var(--radius) var(--radius) 0;
    color: var(--text-strong);
    font-size: 0.95rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    line-height: 1.6;
}

.conclusion strong { color: var(--accent); }

/* Footer */
.footer {
    text-align: center;
    padding: 3rem 0;
    color: var(--text-muted);
    font-size: 0.75rem;
    border-top: 1px solid var(--border);
    margin-top: 4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Utilities */
.text-center { text-align: center; }
.text-right { text-align: right; }
.font-mono { font-family: 'JetBrains Mono', monospace; }

/* R-Group Legend */
.rgroup-legend {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.5rem;
    border-radius: var(--radius);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
}

.rgroup-legend .core-scaffold {
    background: white;
    padding: 0.5rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
}

.rgroup-legend .strategy-badge {
    background: var(--accent-soft);
    color: var(--accent);
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
"""


# =============================================================================
# Section Builders
# =============================================================================

def build_rgroup_decomposition_table_section(
    r_group_data: dict,
    scaffold: str = None,
    scaffold_strategy: str = None,
) -> str:
    """Build R-group decomposition table with molecule SVGs and core highlighting.
    
    Args:
        r_group_data: Dict containing decomposed_compounds and ocat_pairs.
        scaffold: SMARTS/SMILES of the core scaffold.
        scaffold_strategy: Strategy used (mcs, murcko, custom).
    
    Returns:
        HTML string for the R-group decomposition section.
    """
    decomposed = r_group_data.get("decomposed_compounds", [])
    
    if not decomposed:
        return ""
    
    # Import RDKit for molecule rendering
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import AllChem
    
    def _render_mol_with_highlight(smiles: str, core_smarts: str = None, width: int = 140, height: int = 100) -> str:
        """Render molecule SVG with core scaffold highlighted."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f'<svg width="{width}" height="{height}"><rect width="100%" height="100%" fill="#f3f4f6"/><text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="10">Invalid</text></svg>'
            
            AllChem.Compute2DCoords(mol)
            
            highlight_atoms = []
            highlight_bonds = []
            atom_colors = {}
            bond_colors = {}
            
            # Highlight core scaffold if provided
            if core_smarts:
                core = Chem.MolFromSmarts(core_smarts)
                if core is None:
                    core = Chem.MolFromSmiles(core_smarts)
                
                if core:
                    match = mol.GetSubstructMatch(core)
                    if match:
                        highlight_atoms = list(match)
                        # Get bonds within the match
                        for bond in mol.GetBonds():
                            if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                                highlight_bonds.append(bond.GetIdx())
                        # Color core atoms BRIGHT ORANGE for high visibility
                        for idx in highlight_atoms:
                            atom_colors[idx] = (1.0, 0.5, 0.0, 0.7)
                        for idx in highlight_bonds:
                            bond_colors[idx] = (1.0, 0.5, 0.0, 0.8)
            
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().bondLineWidth = 1.5
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds,
                highlightAtomColors=atom_colors if atom_colors else {},
                highlightBondColors=bond_colors if bond_colors else {},
            )
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").replace("\n", " ")
        except Exception:
            return f'<svg width="{width}" height="{height}"><rect width="100%" height="100%" fill="#f3f4f6"/><text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="10">Error</text></svg>'
    
    def _render_rgroup(rgroup_smi: str, width: int = 80, height: int = 60) -> str:
        """Render R-group fragment SVG."""
        if not rgroup_smi or rgroup_smi.startswith("[*"):
            return f'<span style="color:#6b7280">-</span>'
        try:
            mol = Chem.MolFromSmiles(rgroup_smi)
            if mol is None:
                return f'<code>{rgroup_smi[:15]}</code>'
            
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().bondLineWidth = 1.0
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").replace("\n", " ")
        except Exception:
            return f'<code>{rgroup_smi[:15]}</code>'
    
    # Get all R-group positions
    all_r_positions = set()
    for cpd in decomposed:
        all_r_positions.update(cpd.get("r_groups", {}).keys())
    r_positions = sorted(all_r_positions)
    
    # Build HTML
    html = '<div class="card"><h2>R-Group Decomposition Table</h2>'
    
    # Show scaffold legend with visualization and strategy reason
    strategy_names = {"mcs": "MCS", "murcko": "Murcko Scaffold", "custom": "Custom", "none": "None"}
    strategy_reasons = {
        "mcs": "Based on 80% threshold MCS detection.",
        "murcko": "Extracted Murcko scaffold (core ring system).",
        "custom": "User-defined core scaffold.",
        "none": "No common scaffold identified.",
    }
    
    if scaffold:
        # Render scaffold as molecule SVG
        def _render_scaffold_svg(scaffold_str: str, width: int = 150, height: int = 100) -> str:
            """Render scaffold pattern as SVG."""
            try:
                # Try as SMARTS first, then SMILES
                mol = Chem.MolFromSmarts(scaffold_str)
                if mol is None:
                    mol = Chem.MolFromSmiles(scaffold_str)
                if mol is None:
                    return f'<code style="font-size:0.7rem">{scaffold_str[:50]}...</code>'
                
                AllChem.Compute2DCoords(mol)
                drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
                drawer.drawOptions().bondLineWidth = 2.0
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
                return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").replace("\n", " ")
            except Exception:
                return f'<code style="font-size:0.7rem">{scaffold_str[:50]}...</code>'
        
        scaffold_svg = _render_scaffold_svg(scaffold, 150, 100)
        reason = strategy_reasons.get(scaffold_strategy, "Automatic selection")
        
        html += f'''
        <div class="rgroup-legend">
            <div style="flex:1">
                <div style="margin-bottom:0.5rem">
                    <strong style="font-size:1.1rem;color:var(--primary)">Core Scaffold</strong>
                    <span class="strategy-badge">{strategy_names.get(scaffold_strategy, scaffold_strategy or "auto")}</span>
                </div>
                <div style="font-size:0.875rem;color:var(--text-muted)">{reason}</div>
            </div>
            <div class="core-scaffold">
                {scaffold_svg}
            </div>
        </div>
        '''
    
    html += f'<p style="margin-bottom:1rem;color:var(--text-secondary)">Decomposed <strong>{len(decomposed)}</strong> compounds, identified <strong>{len(r_positions)}</strong> R-group positions.</p>'
    
    # Build table
    html += '<div class="table-container"><table class="rgroup-table">'
    
    # Header
    html += '<thead><tr>'
    html += '<th>ID</th>'
    html += '<th class="mol-cell">Structure (Core Highlighted)</th>'
    for pos in r_positions:
        html += f'<th class="rgroup-cell">{pos}</th>'
    html += '<th class="activity-cell">Name</th>'
    html += '<th class="activity-cell">Activity</th>'
    html += '</tr></thead>'
    
    # Body
    html += '<tbody>'
    max_display = min(len(decomposed), 30)  # Limit to 30 rows
    
    for cpd in decomposed[:max_display]:
        cpd_id = cpd.get("compound_id", "")
        smiles = cpd.get("smiles", "")
        activity = cpd.get("activity")
        # Use name from original data, fallback to compound_id
        name = cpd.get("name") or cpd.get("Name") or cpd_id
        r_groups = cpd.get("r_groups", {})
        
        mol_svg = _render_mol_with_highlight(smiles, scaffold, 140, 100)
        
        html += '<tr>'
        html += f'<td><strong>{cpd_id}</strong></td>'
        html += f'<td class="mol-cell">{mol_svg}</td>'
        
        for pos in r_positions:
            rg = r_groups.get(pos, "")
            rg_svg = _render_rgroup(rg)
            html += f'<td class="rgroup-cell">{rg_svg}</td>'
        
        html += f'<td class="activity-cell">{name}</td>'
        act_str = f"{activity}" if activity is not None else "-"
        html += f'<td class="activity-cell">{act_str}</td>'
        html += '</tr>'
    
    html += '</tbody></table></div>'
    
    if len(decomposed) > max_display:
        html += f'<p style="color:var(--text-muted);font-size:0.875rem;margin-top:1rem;text-align:center">Showing first {max_display} of {len(decomposed)} records</p>'
    
    html += '</div>'
    return html


def build_stats_section(sar_data: dict) -> str:
    """Build summary statistics section."""
    stats = []
    
    # 1. Basic Counts
    total_compounds = sar_data.get("total_compounds", 0)
    stats.append(("Total Compounds", total_compounds))
    
    # 2. Activity Statistics
    compounds = sar_data.get("compounds", [])
    activities = []
    mols = []
    
    if compounds:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        for c in compounds:
            act = c.get("activity")
            if act is not None:
                try:
                    activities.append(float(act))
                except (ValueError, TypeError):
                    pass
            
            smi = c.get("smiles")
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mols.append(mol)
    
    if activities:
        stats.append(("Activity Range", f"{min(activities):.2f} - {max(activities):.2f}"))
        stats.append(("Mean Activity", f"{sum(activities)/len(activities):.2f}"))
    
    # 3. Molecular Properties (MW, LogP)
    if mols:
        avg_mw = sum(Descriptors.MolWt(m) for m in mols) / len(mols)
        avg_logp = sum(Descriptors.MolLogP(m) for m in mols) / len(mols)
        stats.append(("Avg MW", f"{avg_mw:.1f}"))
        stats.append(("Avg LogP", f"{avg_logp:.1f}"))
        
    # 4. Scaffold Info
    scaffold_strategy = sar_data.get("scaffold_strategy")
    if scaffold_strategy:
        stats.append(("Scaffold Strategy", scaffold_strategy.upper()))
        
    # 5. Other Counts
    cliffs = sar_data.get("activity_cliffs", {})
    if cliffs.get("activity_cliffs_found"):
        stats.append(("Activity Cliffs", f"{cliffs['activity_cliffs_found']} pairs"))

    fg = sar_data.get("functional_group_sar", {})
    if fg.get("functional_group_sar"):
        essential = len([f for f in fg["functional_group_sar"] if f.get("effect") == "essential"])
        if essential > 0:
            stats.append(("Essential Groups", essential))

    html = '<div class="card"><h2>Summary Statistics</h2><div class="stats-grid">'
    for label, value in stats:
        html += f'''
        <div class="stat-card">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>'''
    html += '</div></div>'
    return html


def build_compound_gallery_section(compounds: list[dict], max_display: int = 24) -> str:
    """Build compound gallery section with molecule SVG images.
    
    Args:
        compounds: List of compound dicts with 'smiles', 'compound_id', 'activity'.
        max_display: Maximum number of compounds to display.
    
    Returns:
        HTML string for the compound gallery section.
    """
    if not compounds:
        return ""
    
    # Import at function level to avoid circular imports
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import AllChem
    
    def _smiles_to_svg(smiles: str, width: int = 180, height: int = 140) -> str:
        """Generate inline SVG from SMILES (local function to avoid circular import)."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#f3f4f6" rx="8"/><text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="12">Invalid</text></svg>'
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().bondLineWidth = 2.0
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").replace("\n", " ")
        except Exception:
            return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#f3f4f6" rx="8"/><text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="12">Error</text></svg>'
    
    html = '<div class="card"><h2>Compound Gallery</h2>'
    html += f'<p style="margin-bottom:1.5rem;color:var(--text-secondary)">Total {len(compounds)} compounds'
    if len(compounds) > max_display:
        html += f', showing first {max_display}'
    html += '</p>'
    
    html += '<div class="mol-grid">'
    
    for i, cpd in enumerate(compounds[:max_display]):
        smiles = cpd.get("smiles", "")
        compound_id = cpd.get("compound_id", f"Cpd-{i+1}")
        activity = cpd.get("activity")
        
        svg = _smiles_to_svg(smiles, width=200, height=150)
        
        activity_str = f"Act: {activity}" if activity is not None else ""
        
        html += f'''
        <div class="mol-card">
            <div class="mol-svg">{svg}</div>
            <div class="mol-id">{compound_id}</div>
            <div class="mol-activity">{activity_str}</div>
        </div>'''
    
    html += '</div></div>'
    return html


def build_rgroup_section(rgroup_data: dict) -> str:
    """Build R-group analysis section."""
    if not rgroup_data or "error" in rgroup_data:
        return ""

    recommendations = rgroup_data.get("recommendations", [])
    r_group_analysis = rgroup_data.get("r_group_analysis", {})

    if not recommendations and not r_group_analysis:
        return ""

    html = '<div class="card"><h2>R-Group SAR Analysis</h2>'

    # Activity range
    activity_range = rgroup_data.get("activity_range", {})
    if activity_range.get("min") is not None:
        html += f'''
        <p style="margin-bottom:1rem"><strong>Activity Range:</strong> {activity_range.get("min")} - {activity_range.get("max")}</p>
        '''

    # Recommendations table
    if recommendations:
        html += '''
        <h3>Optimal Substituents</h3>
        <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Position</th>
                    <th>Best Substituent</th>
                    <th>Best Activity</th>
                    <th>Worst Substituent</th>
                    <th>Worst Activity</th>
                </tr>
            </thead>
            <tbody>'''

        for rec in recommendations:
            html += f'''
                <tr>
                    <td><strong>{rec.get('position', '')}</strong></td>
                    <td><span class="badge badge-essential">{rec.get('best_substituent', '')}</span></td>
                    <td>{rec.get('best_activity', '')}</td>
                    <td><span class="badge badge-detrimental">{rec.get('worst_substituent', '')}</span></td>
                    <td>{rec.get('worst_activity', '')}</td>
                </tr>'''

        html += '</tbody></table></div>'

    # Detailed R-group analysis
    if r_group_analysis:
        html += '<h3>Detailed Analysis</h3>'
        for r_name, values in r_group_analysis.items():
            html += f'<h4>{r_name}</h4><div class="table-container"><table><thead><tr><th>Substituent</th><th>Count</th><th>Mean Activity</th><th>Range</th></tr></thead><tbody>'
            for r_value, stats in values.items():
                html += f'''
                    <tr>
                        <td>{r_value}</td>
                        <td>{stats.get('count', '')}</td>
                        <td>{stats.get('mean_activity', '')}</td>
                        <td>{stats.get('min_activity', '')} - {stats.get('max_activity', '')}</td>
                    </tr>'''
            html += '</tbody></table></div>'

    # Conclusion
    if recommendations:
        best_parts = [f"{r['position']}={r['best_substituent']}" for r in recommendations[:3]]
        html += f'<div class="conclusion"><strong>SAR Conclusion:</strong> Optimal combination appears to be {", ".join(best_parts)}</div>'

    html += '</div>'
    return html


def build_functional_group_section(fg_data: dict) -> str:
    """Build functional group SAR section."""
    if not fg_data or "error" in fg_data:
        return ""

    # Handle both direct list and dict with "functional_group_sar" key
    if isinstance(fg_data, list):
        fg_list = fg_data
    else:
        fg_list = fg_data.get("functional_group_sar", [])
    
    if not fg_list:
        return ""

    html = '<div class="card"><h2>Functional Group SAR</h2>'
    html += '''
    <div class="table-container">
    <table>
        <thead>
            <tr>
                <th>Functional Group</th>
                <th>Effect</th>
                <th>Fold Change</th>
                <th>Avg Activity (With / Without)</th>
            </tr>
        </thead>
        <tbody>'''

    badge_map = {
        "essential": ("badge-essential", "Essential"),
        "beneficial": ("badge-beneficial", "Beneficial"),
        "tolerated": ("badge-tolerated", "Neutral"),
        "detrimental": ("badge-detrimental", "Detrimental"),
    }

    for fg in fg_list:
        effect = fg.get("effect", "")
        badge_class, effect_en = badge_map.get(effect, ("", effect))
        html += f'''
            <tr>
                <td><strong>{fg.get('functional_group', '')}</strong></td>
                <td><span class="badge {badge_class}">{effect_en}</span></td>
                <td>{fg.get('fold_change', '')}x</td>
                <td>{fg.get('avg_activity_with', '')} / {fg.get('avg_activity_without', '')}</td>
            </tr>'''

    html += '</tbody></table></div>'

    # Conclusion
    essential_fgs = [f["functional_group"] for f in fg_list if f.get("effect") == "essential"]
    if essential_fgs:
        html += f'<div class="conclusion"><strong>Key Findings:</strong> {", ".join(essential_fgs)} are essential for activity.</div>'

    html += '</div>'
    return html


def build_conformational_section(conf_data: dict) -> str:
    """Build conformational SAR section."""
    if not conf_data or "error" in conf_data:
        return ""
    
    # Check if there's any meaningful data
    if not conf_data.get("planarity") and not conf_data.get("rigidity") and not conf_data.get("conclusions"):
        return ""

    html = '<div class="card"><h2>Conformational SAR</h2>'

    if conf_data.get("planarity"):
        p = conf_data["planarity"]
        html += f'''
        <h3>Planarity Analysis</h3>
        <div class="table-container">
        <table>
            <tr><th>Planar Molecules</th><td>{p.get('planar_count', 'N/A')}</td></tr>
            <tr><th>Non-planar Molecules</th><td>{p.get('nonplanar_count', 'N/A')}</td></tr>
            <tr><th>Preference</th><td><strong>{p.get('preference', 'N/A')}</strong></td></tr>
        </table></div>'''

    if conf_data.get("rigidity"):
        r = conf_data["rigidity"]
        html += f'''
        <h3>Rigidity Analysis</h3>
        <div class="table-container">
        <table>
            <tr><th>Rigid Molecules</th><td>{r.get('rigid_count', 'N/A')}</td></tr>
            <tr><th>Flexible Molecules</th><td>{r.get('flexible_count', 'N/A')}</td></tr>
            <tr><th>Preference</th><td><strong>{r.get('preference', 'N/A')}</strong></td></tr>
        </table></div>'''

    if conf_data.get("conclusions"):
        for c in conf_data["conclusions"]:
            html += f'<div class="conclusion"><strong>Insight:</strong> {c}</div>'

    html += '</div>'
    return html


def build_activity_cliffs_section(cliffs_data: dict) -> str:
    """Build activity cliffs section with molecule SVGs."""
    if not cliffs_data or "error" in cliffs_data:
        return ""

    # Import at function level to avoid circular imports
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import AllChem
    
    def _cliff_smiles_to_svg(smiles: str, width: int = 180, height: int = 140) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#f3f4f6" rx="8"/><text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="12">Invalid</text></svg>'
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().bondLineWidth = 2.0
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").replace("\n", " ")
        except Exception:
            return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#f3f4f6" rx="8"/><text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="12">Error</text></svg>'

    # Handle both direct list and dict with "cliffs" key
    if isinstance(cliffs_data, list):
        cliffs = cliffs_data
        count = len(cliffs)
    else:
        cliffs = cliffs_data.get("cliffs", [])
        count = cliffs_data.get("activity_cliffs_found", 0)

    html = f'<div class="card"><h2>Activity Cliffs</h2>'
    html += f'<p>Found <strong>{count}</strong> pairs of activity cliffs (Similarity &gt; 0.7, Activity Diff &gt; 10x)</p>'

    if cliffs:
        # Show visual comparison for top cliffs
        html += '<h3>Structural Comparison</h3>'
        for cliff in cliffs[:5]:
            smi1 = cliff.get('mol1', '')
            smi2 = cliff.get('mol2', '')
            act1 = cliff.get('activity1', 'N/A')
            act2 = cliff.get('activity2', 'N/A')
            fold = cliff.get('fold_change', 'N/A')
            sim = cliff.get('similarity', 'N/A')
            
            svg1 = _cliff_smiles_to_svg(smi1, width=180, height=140)
            svg2 = _cliff_smiles_to_svg(smi2, width=180, height=140)
            
            html += f'''
            <div class="mol-pair">
                <div style="text-align:center">
                    <div class="mol-svg">{svg1}</div>
                    <div style="font-size:0.875rem;margin-top:0.5rem">Activity: <strong>{act1}</strong></div>
                </div>
                <div class="arrow">↔️<br><span style="font-size:0.75rem">{fold}x</span></div>
                <div style="text-align:center">
                    <div class="mol-svg">{svg2}</div>
                    <div style="font-size:0.875rem;margin-top:0.5rem">Activity: <strong>{act2}</strong></div>
                </div>
                <div style="font-size:0.75rem;color:var(--text-secondary)">
                    Similarity: {sim}
                </div>
            </div>'''

        # Also show a summary table
        html += '''
        <h3>Detailed Data</h3>
        <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Molecule 1 (SMILES)</th>
                    <th>Molecule 2 (SMILES)</th>
                    <th>Similarity</th>
                    <th>Activity 1</th>
                    <th>Activity 2</th>
                    <th>Fold Change</th>
                </tr>
            </thead>
            <tbody>'''

        for cliff in cliffs[:10]:
            smi1 = cliff.get('mol1', '')
            smi2 = cliff.get('mol2', '')
            html += f'''
                <tr>
                    <td class="smiles">{smi1[:40]}{"..." if len(smi1) > 40 else ""}</td>
                    <td class="smiles">{smi2[:40]}{"..." if len(smi2) > 40 else ""}</td>
                    <td>{cliff.get('similarity', '')}</td>
                    <td>{cliff.get('activity1', '')}</td>
                    <td>{cliff.get('activity2', '')}</td>
                    <td><strong>{cliff.get('fold_change', '')}x</strong></td>
                </tr>'''

        html += '</tbody></table></div>'

        if count > 0:
            html += '<div class="conclusion"><strong>Insight:</strong> Small structural changes leading to large activity differences indicate critical SAR regions.</div>'

    html += '</div>'
    return html


def build_scaffold_section(scaffold_data: dict) -> str:
    """Build scaffold SAR section."""
    if not scaffold_data or "error" in scaffold_data:
        return ""

    html = '<div class="card"><h2>Scaffold SAR</h2>'

    core = scaffold_data.get("core_scaffold", "N/A")
    essential = "Yes" if scaffold_data.get("scaffold_essential") else "No"

    html += f'''
    <div class="table-container">
    <table>
        <tr><th>Core Scaffold</th><td class="smiles">{core}</td></tr>
        <tr><th>Essential?</th><td>{essential}</td></tr>
        <tr><th>Conserved Count</th><td>{scaffold_data.get('scaffold_conserved', 'N/A')}</td></tr>
        <tr><th>Varied Count</th><td>{scaffold_data.get('scaffold_varied', 'N/A')}</td></tr>
    </table></div>'''

    if scaffold_data.get("conclusion"):
        html += f'<div class="conclusion"><strong>Conclusion:</strong> {scaffold_data["conclusion"]}</div>'

    html += '</div>'
    return html


def build_positional_section(pos_data: dict) -> str:
    """Build positional SAR section."""
    if not pos_data or "error" in pos_data:
        return ""

    positions = pos_data.get("positional_sar", [])
    if not positions:
        return ""

    html = '<div class="card"><h2>Positional SAR</h2>'

    for pos in positions:
        html += f'''
        <h3>{pos.get('position', 'Unknown')}</h3>
        <div class="table-container">
        <table>
            <tr><th>Best Substituent</th><td>{pos.get('best_substituent', 'N/A')}</td></tr>
            <tr><th>Best Activity</th><td>{pos.get('best_activity', 'N/A')}</td></tr>
            <tr><th>Size Preference</th><td>{pos.get('size_preference', 'N/A')}</td></tr>
        </table></div>'''

    html += '</div>'
    return html


# =============================================================================
# Main Report Builder
# =============================================================================

def build_sar_html_report(sar_data: dict, title: str = "SAR Analysis Report") -> str:
    """Build complete HTML SAR report.

    Args:
        sar_data: Dictionary containing all SAR analysis results.
        title: Report title.

    Returns:
        Complete HTML string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_compounds = sar_data.get("total_compounds", 0)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{SAR_REPORT_CSS}</style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">
                <span>Generated: {timestamp}</span>
                <span>Total Compounds: {total_compounds}</span>
            </div>
        </div>
'''

    # =========================================================================
    # Generate Visualizations (if advanced data is available)
    # =========================================================================
    plots = {}
    
    # Note: Advanced visualizations using SARVisualizerAdvanced require DataFrame 
    # with pActivity. For basic reports, we skip these plots.
    # Use generate_sar_visualizations() separately for full interactive charts.

    # =========================================================================
    # Build HTML Sections
    # =========================================================================

    # 1. Summary Stats
    html += build_stats_section(sar_data)

    # Pass the raw tool output to each section builder
    # Each tool returns a dict that the section builder knows how to parse
    
    # Add R-group decomposition table with molecule visualization
    
    # 3. Property Analysis (New)
    if plots.get("prop_act"):
        html += f'<div class="card"><h2>⚗️ Property-Activity Landscape</h2><img src="{plots["prop_act"]}" style="width:100%;border-radius:0.5rem;"></div>'

    # 4. R-Group Analysis (Enhanced)
    if "r_group_analysis" in sar_data:
        # Insert Heatmap and Distribution before the table
        if plots.get("pos_heatmap") or plots.get("act_dist"):
            html += '<div class="card"><h2>🧪 R-Group SAR Visualization</h2><div class="mol-grid" style="grid-template-columns: 1fr 1fr;">'
            if plots.get("pos_heatmap"):
                html += f'<div><img src="{plots["pos_heatmap"]}" style="width:100%;border-radius:0.5rem;"></div>'
            if plots.get("act_dist"):
                html += f'<div><img src="{plots["act_dist"]}" style="width:100%;border-radius:0.5rem;"></div>'
            html += '</div></div>'
            
        html += build_rgroup_decomposition_table_section(
            sar_data["r_group_analysis"],
            sar_data.get("scaffold"),
            sar_data.get("scaffold_strategy")
        )
        html += build_rgroup_section(sar_data["r_group_analysis"])

    # 5. Functional Groups (Enhanced)
    if "functional_group_sar" in sar_data:
        if plots.get("fg_matrix"):
             html += f'<div class="card"><h2>🧩 Functional Group Necessity</h2><img src="{plots["fg_matrix"]}" style="width:100%;border-radius:0.5rem;margin-bottom:1.5rem;"></div>'
        html += build_functional_group_section(sar_data["functional_group_sar"])

    # 6. Conformational SAR
    if "conformational_sar" in sar_data:
        html += build_conformational_section(sar_data["conformational_sar"])

    # 7. Activity Cliffs (Enhanced)
    if "activity_cliffs" in sar_data:
        if plots.get("mmp_diff"):
             html += f'<div class="card"><h2>📉 MMP Activity Distribution</h2><img src="{plots["mmp_diff"]}" style="width:100%;border-radius:0.5rem;margin-bottom:1.5rem;"></div>'
        html += build_activity_cliffs_section(sar_data["activity_cliffs"])

    # 8. Scaffold SAR (Enhanced)
    if "scaffold_sar" in sar_data:
        if plots.get("scaffold_anno"):
             html += f'<div class="card"><h2>🎯 Scaffold Annotation</h2><img src="{plots["scaffold_anno"]}" style="width:100%;max_width:500px;display:block;margin:0 auto;border-radius:0.5rem;margin-bottom:1.5rem;"></div>'
        html += build_scaffold_section(sar_data["scaffold_sar"])

    # 9. Positional SAR
    if "positional_sar" in sar_data:
        html += build_positional_section(sar_data["positional_sar"])
        
    # 10. Compound Gallery
    if "compounds" in sar_data:
        html += build_compound_gallery_section(sar_data["compounds"])

    html += f'''
        <div class="footer">
            <p>Generated by MolX Agent • {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
    </div>
</body>
</html>'''
    
    return html


def build_advanced_sar_section(vis_results: dict) -> str:
    """Build advanced SAR visualization section with interactive Plotly charts.
    
    Args:
        vis_results: Dictionary from SARVisualizerAdvanced.generate_all()
    
    Returns:
        HTML string for the advanced SAR section.
    """
    if not vis_results:
        return ""
    
    html = '<div class="card"><h2>📊 Advanced SAR Visualizations</h2>'
    html += '<p style="color:var(--text-muted);margin-bottom:1.5rem;">Interactive plots - hover for details, zoom and pan available.</p>'
    
    # List of visualization types with titles
    plot_titles = {
        "position_heatmap": ("🎯 Position-wise SAR Heatmap", "Activity patterns across R-group positions"),
        "mmp_analysis": ("🔗 Matched Molecular Pair Analysis", "Single-change structure-activity pairs"),
        "fg_matrix": ("🧪 Functional Group Necessity", "Impact of functional groups on activity"),
        "property_activity": ("📈 Property-Activity Relationships", "Correlations with molecular properties"),
        "activity_distribution": ("📊 Activity Distribution", "Activity spread per R-group position"),
        "lead_radar": ("🎖️ Lead vs Backup Comparison", "Multi-criteria candidate comparison"),
        "timeline": ("📅 SAR Evolution Timeline", "Activity improvement over iterations"),
        "scaffold_annotation": ("🧬 Scaffold Annotation", "Core scaffold with position classifications"),
    }
    
    for key, (title, description) in plot_titles.items():
        result = vis_results.get(key, {})
        html_div = result.get("html_div", "")
        
        if html_div:
            html += f'''
            <div style="margin-bottom:2rem;">
                <h3 style="margin-bottom:0.5rem;">{title}</h3>
                <p style="color:var(--text-muted);font-size:0.875rem;margin-bottom:1rem;">{description}</p>
                <div style="background:white;border-radius:var(--radius);padding:1rem;border:1px solid var(--border);">
                    {html_div}
                </div>
            </div>
            '''
    
    html += '</div>'
    return html


def save_html_report(html: str, filename: str = None) -> str:
    """Save HTML report to file.

    Args:
        html: HTML content.
        filename: Optional filename (auto-generated if not provided).

    Returns:
        Path to saved file.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sar_report_{timestamp}.html"

    filepath = os.path.join(REPORT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath

