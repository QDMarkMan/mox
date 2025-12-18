"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description SAR Visualizer - 8 types of decision-valuable SAR plots.
*               Uses Plotly for interactive HTML charts.
**************************************************************************
"""

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)

# Output directory for visualization files
VIS_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "visualizations")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SARVisualizerConfig:
    """Configuration parameters for SAR visualizations."""
    min_count: int = 3  # Minimum samples to show in heatmap
    delta_activity_cutoff: float = 0.5  # pActivity difference threshold
    property_ranges: Dict[str, Tuple[float, float]] = None
    mmp_similarity_threshold: float = 0.7
    top_k_timeline: int = 3
    
    def __post_init__(self):
        if self.property_ranges is None:
            self.property_ranges = {
                "cLogP": (-2, 7),
                "MW": (100, 800),
                "TPSA": (0, 200),
            }


# =============================================================================
# Data Preprocessor
# =============================================================================

class SARDataPreprocessor:
    """Preprocessor for SAR data - handles unit conversion and property calculation."""
    
    REQUIRED_COLUMNS = ["compound_id", "smiles", "activity_value", "activity_unit"]
    OPTIONAL_COLUMNS = ["series_id", "iteration", "key_change"]
    PROPERTY_COLUMNS = ["cLogP", "MW", "TPSA", "HBD", "HBA", "RB"]
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with raw DataFrame."""
        self.raw_df = df.copy()
        self.df = None
        self._validate_columns()
    
    def _validate_columns(self):
        """Validate required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.raw_df.columns]
        
        # Check for alternative column names
        alt_names = {
            "compound_id": ["id", "compound_name", "name", "cpd_id"],
            "smiles": ["SMILES", "smi", "structure"],
            "activity_value": ["activity", "IC50", "EC50", "Ki"],
            "activity_unit": ["unit", "units"],
        }
        
        for col in missing[:]:
            for alt in alt_names.get(col, []):
                if alt in self.raw_df.columns:
                    self.raw_df[col] = self.raw_df[alt]
                    missing.remove(col)
                    break
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def preprocess(self) -> pd.DataFrame:
        """Run full preprocessing pipeline."""
        self.df = self.raw_df.copy()
        
        # Convert activity to pActivity
        self._convert_to_pactivity()
        
        # Calculate missing molecular properties
        self._calculate_properties()
        
        # Handle duplicate measurements (take median)
        self._handle_duplicates()
        
        # Identify R-group columns
        self._identify_rgroups()
        
        return self.df
    
    def _convert_to_pactivity(self):
        """Convert activity values to pActivity (-log10 of molar concentration)."""
        def to_pactivity(row):
            value = row["activity_value"]
            unit = str(row["activity_unit"]).lower().strip()
            
            if pd.isna(value) or value <= 0:
                return np.nan
            
            # Convert to molar
            if unit in ["nm", "nanomolar"]:
                molar = value * 1e-9
            elif unit in ["um", "µm", "micromolar"]:
                molar = value * 1e-6
            elif unit in ["mm", "millimolar"]:
                molar = value * 1e-3
            elif unit in ["m", "molar"]:
                molar = value
            else:
                # Assume nM if unit unclear
                logger.warning(f"Unknown unit '{unit}', assuming nM")
                molar = value * 1e-9
            
            return -np.log10(molar)
        
        self.df["pActivity"] = self.df.apply(to_pactivity, axis=1)
        logger.info(f"Converted {self.df['pActivity'].notna().sum()} activity values to pActivity")
    
    def _calculate_properties(self):
        """Calculate missing molecular properties using RDKit."""
        property_funcs = {
            "MW": Descriptors.MolWt,
            "cLogP": Descriptors.MolLogP,
            "TPSA": Descriptors.TPSA,
            "HBD": Descriptors.NumHDonors,
            "HBA": Descriptors.NumHAcceptors,
            "RB": Descriptors.NumRotatableBonds,
        }
        
        # Cache molecules
        if "_mol" not in self.df.columns:
            self.df["_mol"] = self.df["smiles"].apply(
                lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None
            )
        
        for prop, func in property_funcs.items():
            if prop not in self.df.columns or self.df[prop].isna().all():
                self.df[prop] = self.df["_mol"].apply(
                    lambda m: func(m) if m else np.nan
                )
                logger.info(f"Calculated {prop} for {self.df[prop].notna().sum()} molecules")
    
    def _handle_duplicates(self):
        """Handle duplicate compound measurements by taking median."""
        if self.df.duplicated(subset=["smiles"]).any():
            # Group by SMILES and aggregate
            agg_dict = {"pActivity": "median", "compound_id": "first"}
            for col in self.PROPERTY_COLUMNS:
                if col in self.df.columns:
                    agg_dict[col] = "first"
            
            # Preserve R-group columns
            r_cols = [c for c in self.df.columns if c.startswith("R") and c[1:].isdigit()]
            for col in r_cols:
                agg_dict[col] = "first"
            
            original_count = len(self.df)
            self.df = self.df.groupby("smiles", as_index=False).agg(agg_dict)
            logger.info(f"Reduced {original_count} to {len(self.df)} after deduplication")
    
    def _identify_rgroups(self):
        """Identify R-group columns in the DataFrame."""
        self.rgroup_columns = [
            c for c in self.df.columns 
            if c.startswith("R") and len(c) <= 3 and c[1:].isdigit()
        ]
        logger.info(f"Identified R-group columns: {self.rgroup_columns}")
    
    def get_rgroup_columns(self) -> List[str]:
        """Get list of R-group column names."""
        return getattr(self, "rgroup_columns", [])


# =============================================================================
# Molecule Rendering Utilities
# =============================================================================

def mol_to_svg(smiles: str, width: int = 300, height: int = 200, 
               highlight_atoms: List[int] = None) -> str:
    """Convert SMILES to SVG string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle">Invalid</text></svg>'
        
        AllChem.Compute2DCoords(mol)
        
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        svg = drawer.GetDrawingText()
        return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").strip()
    except Exception as e:
        logger.error(f"Error rendering molecule: {e}")
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle">Error</text></svg>'


def mol_to_base64_png(smiles: str, width: int = 300, height: int = 200) -> str:
    """Convert SMILES to base64-encoded PNG."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(width, height))
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Error rendering molecule: {e}")
        return ""


# =============================================================================
# SAR Visualizer Advanced
# =============================================================================

class SARVisualizerAdvanced:
    """Advanced SAR Visualizer with 8 interactive plot types."""
    
    def __init__(self, df: pd.DataFrame, config: SARVisualizerConfig = None, 
                 activity_column: str = None):
        """
        Initialize visualizer with preprocessed DataFrame.
        
        Args:
            df: DataFrame with pActivity and molecular properties calculated.
            config: Configuration parameters.
            activity_column: Specific activity column to use for visualizations.
                           If provided, will remap this column to pActivity.
        """
        self.df = df.copy()
        self.config = config or SARVisualizerConfig()
        self.output_dir = VIS_OUTPUT_DIR
        self.activity_column = activity_column
        os.makedirs(self.output_dir, exist_ok=True)
        
        # If an activity column is specified, remap it to pActivity
        if activity_column and activity_column in self.df.columns:
            # Store original pActivity if it exists
            if "pActivity" in self.df.columns:
                self.df["pActivity_original"] = self.df["pActivity"]
            # Use the specified activity column as pActivity
            self.df["pActivity"] = self.df[activity_column]
            logger.info(f"Using activity column '{activity_column}' for visualizations")
        
        # Identify R-group columns
        self.rgroup_columns = [
            c for c in df.columns 
            if c.startswith("R") and len(c) <= 3 and c[1:].isdigit()
        ]
    
    def _save_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to CSV and return path."""
        path = os.path.join(self.output_dir, filename)
        data.to_csv(path, index=False)
        return path
    
    def _save_html(self, fig: go.Figure, filename: str) -> str:
        """Save Plotly figure to HTML and return path."""
        path = os.path.join(self.output_dir, filename)
        fig.write_html(path, include_plotlyjs='cdn')
        return path
    
    def _fig_to_html_div(self, fig: go.Figure) -> str:
        """Convert Plotly figure to embeddable HTML div."""
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # =========================================================================
    # 1. Position-wise SAR Heatmap
    # =========================================================================
    
    def plot_position_sar_heatmap(self) -> Tuple[str, str, str]:
        """
        Generate Position × Substituent SAR heatmap.
        
        Returns:
            Tuple of (html_div, html_file_path, csv_file_path)
        """
        if not self.rgroup_columns:
            logger.warning("No R-group columns found for heatmap")
            return "", "", ""
        
        # Collect data for all positions
        data = []
        for pos in self.rgroup_columns:
            for sub, group in self.df.groupby(pos):
                if pd.notna(sub) and str(sub).strip():
                    median_act = group["pActivity"].median()
                    n = len(group)
                    data.append({
                        "Position": pos,
                        "Substituent": str(sub)[:20],  # Truncate long names
                        "Median_pActivity": median_act,
                        "N": n,
                    })
        
        if not data:
            return "", "", ""
        
        heatmap_df = pd.DataFrame(data)
        
        # Filter low-count cells
        heatmap_df.loc[heatmap_df["N"] < self.config.min_count, "Median_pActivity"] = np.nan
        
        # Pivot for heatmap
        pivot = heatmap_df.pivot_table(
            index="Substituent", 
            columns="Position", 
            values="Median_pActivity",
            aggfunc="first"
        )
        
        # Also create count matrix for annotations
        count_pivot = heatmap_df.pivot_table(
            index="Substituent",
            columns="Position",
            values="N",
            aggfunc="first"
        ).fillna(0).astype(int)
        
        # Create annotation text
        annotations = []
        for i, row in enumerate(pivot.index):
            for j, col in enumerate(pivot.columns):
                val = pivot.loc[row, col]
                n = count_pivot.loc[row, col] if row in count_pivot.index else 0
                if pd.notna(val):
                    text = f"{val:.1f}<br>n={n}"
                else:
                    text = f"n={n}" if n > 0 else ""
                annotations.append(text)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=np.array(annotations).reshape(len(pivot.index), len(pivot.columns)),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="Viridis",
            colorbar=dict(title="pActivity"),
            hoverongaps=False,
        ))
        
        fig.update_layout(
            title="Position-wise SAR Heatmap",
            xaxis_title="R-Group Position",
            yaxis_title="Substituent",
            height=max(400, len(pivot.index) * 30),
            width=max(600, len(pivot.columns) * 100),
        )
        
        # Save outputs
        csv_path = self._save_csv(heatmap_df, "sar_position_map.csv")
        html_path = self._save_html(fig, "sar_position_map.html")
        html_div = self._fig_to_html_div(fig)
        
        return html_div, html_path, csv_path
    
    # =========================================================================
    # 2. MMP Delta Activity Plot
    # =========================================================================
    
    def plot_mmp_delta_activity(self) -> Tuple[str, str, str]:
        """
        Generate Matched Molecular Pair (MMP) delta activity plot.
        
        Returns:
            Tuple of (html_div, html_file_path, csv_file_path)
        """
        # Find MMPs using fingerprint similarity
        mols = []
        for idx, row in self.df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                mols.append({
                    "idx": idx,
                    "mol": mol,
                    "smiles": row["smiles"],
                    "pActivity": row["pActivity"],
                    "compound_id": row.get("compound_id", str(idx)),
                })
        
        # Find pairs with high similarity
        pairs = []
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                try:
                    # Use MCS to find similarity
                    mcs = rdFMCS.FindMCS([mols[i]["mol"], mols[j]["mol"]], timeout=1)
                    if mcs.numAtoms > 0:
                        # Calculate similarity based on MCS coverage
                        size_i = mols[i]["mol"].GetNumAtoms()
                        size_j = mols[j]["mol"].GetNumAtoms()
                        similarity = mcs.numAtoms / max(size_i, size_j)
                        
                        if similarity >= self.config.mmp_similarity_threshold:
                            delta = mols[i]["pActivity"] - mols[j]["pActivity"]
                            pairs.append({
                                "Compound_1": mols[i]["compound_id"],
                                "Compound_2": mols[j]["compound_id"],
                                "SMILES_1": mols[i]["smiles"],
                                "SMILES_2": mols[j]["smiles"],
                                "pActivity_1": mols[i]["pActivity"],
                                "pActivity_2": mols[j]["pActivity"],
                                "Delta_pActivity": abs(delta),
                                "Similarity": similarity,
                                "MCS_Atoms": mcs.numAtoms,
                            })
                except Exception as e:
                    continue
        
        if not pairs:
            logger.warning("No MMP pairs found")
            return "", "", ""
        
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values("Delta_pActivity", ascending=False)
        
        # Create figure with subplots for charts + structure gallery
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=["ΔpActivity Distribution", "Similarity vs ΔpActivity"]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=pairs_df["Delta_pActivity"], nbinsx=20, name="Count", marker_color="#0ea5e9"),
            row=1, col=1
        )
        
        # Scatter with hover info
        fig.add_trace(
            go.Scatter(
                x=pairs_df["Similarity"],
                y=pairs_df["Delta_pActivity"],
                mode="markers",
                marker=dict(size=10, color=pairs_df["Delta_pActivity"], colorscale="Reds", showscale=True),
                text=pairs_df.apply(lambda r: f"{r['Compound_1']} vs {r['Compound_2']}", axis=1),
                hovertemplate="<b>%{text}</b><br>Similarity: %{x:.2f}<br>ΔpActivity: %{y:.2f}<extra></extra>",
                name="MMP Pairs",
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Matched Molecular Pair (MMP) Activity Differences",
            height=500,
            showlegend=False,
        )
        fig.update_xaxes(title_text="ΔpActivity", row=1, col=1)
        fig.update_xaxes(title_text="Similarity", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="ΔpActivity", row=1, col=2)
        
        main_chart_html = self._fig_to_html_div(fig)
        
        # Generate structure pair gallery for top pairs
        top_pairs = pairs_df.head(min(5, len(pairs_df)))
        structure_html = '<div style="margin-top:2rem;"><h3>Top Activity Cliff Pairs</h3>'
        structure_html += '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(400px, 1fr));gap:1.5rem;margin-top:1rem;">'
        
        for _, pair in top_pairs.iterrows():
            svg1 = mol_to_svg(pair["SMILES_1"], width=180, height=140)
            svg2 = mol_to_svg(pair["SMILES_2"], width=180, height=140)
            
            structure_html += f'''
            <div style="background:#f8fafc;border-radius:0.5rem;padding:1rem;border:1px solid #e2e8f0;">
                <div style="display:flex;justify-content:space-around;align-items:center;">
                    <div style="text-align:center;">
                        <div style="background:white;padding:0.5rem;border-radius:4px;display:inline-block;">{svg1}</div>
                        <div style="font-size:0.75rem;color:#64748b;margin-top:0.25rem;">{pair["Compound_1"]}</div>
                        <div style="font-weight:600;">pAct: {pair["pActivity_1"]:.2f}</div>
                    </div>
                    <div style="font-size:1.5rem;color:#0ea5e9;">→</div>
                    <div style="text-align:center;">
                        <div style="background:white;padding:0.5rem;border-radius:4px;display:inline-block;">{svg2}</div>
                        <div style="font-size:0.75rem;color:#64748b;margin-top:0.25rem;">{pair["Compound_2"]}</div>
                        <div style="font-weight:600;">pAct: {pair["pActivity_2"]:.2f}</div>
                    </div>
                </div>
                <div style="text-align:center;margin-top:0.5rem;">
                    <span style="background:#fef2f2;color:#dc2626;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.875rem;">
                        ΔpActivity: {pair["Delta_pActivity"]:.2f}
                    </span>
                    <span style="background:#f0f9ff;color:#0ea5e9;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.875rem;margin-left:0.5rem;">
                        Similarity: {pair["Similarity"]:.2%}
                    </span>
                </div>
            </div>
            '''
        
        structure_html += '</div></div>'
        
        # Combine chart and structures
        html_div = main_chart_html + structure_html
        
        # Save outputs
        csv_path = self._save_csv(pairs_df, "mmp_pairs.csv")
        
        # Save full HTML with structures
        full_html = f"""<!DOCTYPE html>
<html><head><title>MMP Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body style="font-family:sans-serif;padding:2rem;">
<h1>Matched Molecular Pair (MMP) Analysis</h1>
{html_div}
</body></html>"""
        html_path = os.path.join(self.output_dir, "mmp_delta_activity.html")
        with open(html_path, "w") as f:
            f.write(full_html)
        
        return html_div, html_path, csv_path
    
    # =========================================================================
    # 3. Functional Group Necessity Matrix
    # =========================================================================
    
    def plot_functional_group_matrix(self) -> Tuple[str, str, str]:
        """
        Generate Functional Group Necessity Matrix.
        
        Returns:
            Tuple of (html_div, html_file_path, csv_file_path)
        """
        # Define functional group SMARTS
        fg_smarts = {
            "Hydroxyl": "[OX2H]",
            "Methoxy": "[OX2][CH3]",
            "Amino": "[NX3;H2,H1;!$(NC=O)]",
            "Amide": "[NX3][CX3](=[OX1])",
            "Carboxylic": "[CX3](=O)[OX2H1]",
            "Ester": "[CX3](=O)[OX2H0]",
            "Ketone": "[CX3](=O)[#6]",
            "Halogen": "[F,Cl,Br,I]",
            "Nitro": "[NX3+](=O)[O-]",
            "Cyano": "[CX2]#[NX1]",
            "Sulfone": "[SX4](=[OX1])(=[OX1])",
            "Aromatic": "a",
        }
        
        fg_effects = []
        
        for fg_name, smarts in fg_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            
            with_fg = []
            without_fg = []
            
            for _, row in self.df.iterrows():
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is None:
                    continue
                
                pact = row["pActivity"]
                if pd.isna(pact):
                    continue
                
                if mol.HasSubstructMatch(pattern):
                    with_fg.append(pact)
                else:
                    without_fg.append(pact)
            
            if with_fg and without_fg:
                avg_with = np.mean(with_fg)
                avg_without = np.mean(without_fg)
                delta = avg_with - avg_without
                
                # Classify effect
                if delta > self.config.delta_activity_cutoff:
                    effect = "Essential"
                elif delta > 0:
                    effect = "Beneficial"
                elif delta > -self.config.delta_activity_cutoff:
                    effect = "Tolerated"
                else:
                    effect = "Detrimental"
                
                fg_effects.append({
                    "Functional_Group": fg_name,
                    "Count_With": len(with_fg),
                    "Count_Without": len(without_fg),
                    "Avg_pActivity_With": avg_with,
                    "Avg_pActivity_Without": avg_without,
                    "Delta_pActivity": delta,
                    "Effect": effect,
                })
        
        if not fg_effects:
            return "", "", ""
        
        fg_df = pd.DataFrame(fg_effects)
        fg_df = fg_df.sort_values("Delta_pActivity", ascending=False)
        
        # Color mapping
        color_map = {
            "Essential": "#059669",
            "Beneficial": "#0ea5e9",
            "Tolerated": "#64748b",
            "Detrimental": "#dc2626",
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=fg_df["Functional_Group"],
            x=fg_df["Delta_pActivity"],
            orientation="h",
            marker_color=[color_map.get(e, "#64748b") for e in fg_df["Effect"]],
            text=fg_df["Effect"],
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "ΔpActivity: %{x:.2f}<br>"
                "With: %{customdata[0]} compounds (avg: %{customdata[1]:.2f})<br>"
                "Without: %{customdata[2]} compounds (avg: %{customdata[3]:.2f})<extra></extra>"
            ),
            customdata=fg_df[["Count_With", "Avg_pActivity_With", "Count_Without", "Avg_pActivity_Without"]].values,
        ))
        
        # Add reference line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Functional Group Necessity Matrix",
            xaxis_title="ΔpActivity (With - Without)",
            yaxis_title="Functional Group",
            height=max(400, len(fg_df) * 40),
        )
        
        # Save outputs
        csv_path = self._save_csv(fg_df, "functional_group_matrix.csv")
        html_path = self._save_html(fig, "functional_group_matrix.html")
        html_div = self._fig_to_html_div(fig)
        
        return html_div, html_path, csv_path
    
    # =========================================================================
    # 4. Property-Activity Plots
    # =========================================================================
    
    def plot_property_activity(self) -> Tuple[str, str, str]:
        """
        Generate Property vs Activity correlation plots.
        
        Returns:
            Tuple of (html_div, html_file_path, csv_file_path)
        """
        properties = ["cLogP", "MW", "TPSA"]
        available_props = [p for p in properties if p in self.df.columns]
        
        if not available_props:
            logger.warning("No property columns found")
            return "", "", ""
        
        # Calculate correlations
        stats = []
        for prop in available_props:
            valid = self.df[[prop, "pActivity"]].dropna()
            if len(valid) > 2:
                from scipy import stats as sp_stats
                pearson_r, pearson_p = sp_stats.pearsonr(valid[prop], valid["pActivity"])
                spearman_r, spearman_p = sp_stats.spearmanr(valid[prop], valid["pActivity"])
                stats.append({
                    "Property": prop,
                    "Pearson_R": pearson_r,
                    "Pearson_P": pearson_p,
                    "Spearman_R": spearman_r,
                    "Spearman_P": spearman_p,
                    "N": len(valid),
                })
        
        stats_df = pd.DataFrame(stats)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(available_props),
            subplot_titles=[f"pActivity vs {p}" for p in available_props]
        )
        
        # Color by series if available
        has_series = "series_id" in self.df.columns and self.df["series_id"].notna().any()
        
        for i, prop in enumerate(available_props, 1):
            valid = self.df[["compound_id", "smiles", prop, "pActivity"]].dropna()
            
            if has_series:
                valid = valid.join(self.df[["series_id"]])
                fig.add_trace(
                    go.Scatter(
                        x=valid[prop],
                        y=valid["pActivity"],
                        mode="markers",
                        marker=dict(size=8),
                        text=valid["compound_id"],
                        hovertemplate=f"<b>%{{text}}</b><br>{prop}: %{{x:.2f}}<br>pActivity: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1, col=i
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=valid[prop],
                        y=valid["pActivity"],
                        mode="markers",
                        marker=dict(size=8, color="#0ea5e9"),
                        text=valid["compound_id"],
                        hovertemplate=f"<b>%{{text}}</b><br>{prop}: %{{x:.2f}}<br>pActivity: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1, col=i
                )
            
            # Add correlation annotation using subplot index
            stat_row = stats_df[stats_df["Property"] == prop]
            if not stat_row.empty:
                r = stat_row.iloc[0]["Pearson_R"]
                # Use subplot reference (first subplot is x, second is x2, etc.)
                x_ref = "x" if i == 1 else f"x{i}"
                y_ref = "y" if i == 1 else f"y{i}"
                fig.add_annotation(
                    x=0.02, y=0.98, 
                    xref=f"{x_ref} domain", 
                    yref=f"{y_ref} domain",
                    text=f"r = {r:.2f}",
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="white",
                )
        
        fig.update_layout(
            title="Property-Activity Relationships",
            height=500,
            showlegend=False,
        )
        
        for i, prop in enumerate(available_props, 1):
            fig.update_xaxes(title_text=prop, row=1, col=i)
            fig.update_yaxes(title_text="pActivity" if i == 1 else "", row=1, col=i)
        
        # Save outputs
        csv_path = self._save_csv(stats_df, "property_activity_stats.csv")
        html_path = self._save_html(fig, "property_activity.html")
        html_div = self._fig_to_html_div(fig)
        
        return html_div, html_path, csv_path
    
    # =========================================================================
    # 5. SAR Scaffold Annotation
    # =========================================================================
    
    def plot_scaffold_annotation(
        self, 
        scaffold_smiles: str,
        position_rules: Dict[str, str] = None
    ) -> Tuple[str, str]:
        """
        Generate 2D scaffold annotation with R-position highlighting.
        
        Args:
            scaffold_smiles: Core scaffold SMILES/SMARTS.
            position_rules: Dict mapping position to classification 
                           (essential/beneficial/tolerated/do_not_go).
        
        Returns:
            Tuple of (html_div, html_file_path)
        """
        mol = Chem.MolFromSmiles(scaffold_smiles)
        if mol is None:
            mol = Chem.MolFromSmarts(scaffold_smiles)
        
        if mol is None:
            logger.error(f"Could not parse scaffold: {scaffold_smiles}")
            return "", ""
        
        AllChem.Compute2DCoords(mol)
        
        # Color mapping for positions
        color_map = {
            "essential": (0.02, 0.59, 0.41),      # Green
            "beneficial": (0.05, 0.65, 0.91),     # Blue
            "tolerated": (0.39, 0.45, 0.55),      # Gray
            "do_not_go": (0.86, 0.15, 0.15),      # Red
        }
        
        # Find atoms to highlight based on rules
        highlight_atoms = []
        atom_colors = {}
        
        if position_rules:
            # Try to match R-group positions to atom indices
            # This is simplified - real implementation would need SMARTS matching
            for i, atom in enumerate(mol.GetAtoms()):
                symbol = atom.GetSymbol()
                for pos, rule in position_rules.items():
                    if pos in symbol or f"[{pos}]" in scaffold_smiles:
                        highlight_atoms.append(i)
                        atom_colors[i] = color_map.get(rule.lower(), (0.5, 0.5, 0.5))
        
        # Draw molecule
        drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors)
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        # Create legend
        legend_html = "<div style='margin-top:1rem;display:flex;gap:1rem;flex-wrap:wrap;'>"
        for rule, color in color_map.items():
            hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            legend_html += f"<span style='display:flex;align-items:center;gap:0.5rem;'>"
            legend_html += f"<span style='width:16px;height:16px;background:{hex_color};border-radius:4px;'></span>"
            legend_html += f"<span>{rule.replace('_', ' ').title()}</span></span>"
        legend_html += "</div>"
        
        # Wrap in HTML
        html_div = f"""
        <div class="scaffold-annotation">
            <h3 style="margin-bottom:1rem;">Core Scaffold SAR Annotation</h3>
            <div style="background:white;padding:1rem;border-radius:8px;display:inline-block;">
                {svg}
            </div>
            {legend_html}
        </div>
        """
        
        # Save HTML file
        full_html = f"""<!DOCTYPE html>
<html><head><title>Scaffold Annotation</title></head>
<body style="font-family:sans-serif;padding:2rem;">{html_div}</body></html>"""
        
        path = os.path.join(self.output_dir, "sar_scaffold_annotation.html")
        with open(path, "w") as f:
            f.write(full_html)
        
        return html_div, path
    
    # =========================================================================
    # 6. Per-position Activity Distribution
    # =========================================================================
    
    def plot_activity_distribution(self, min_samples: int = 3) -> Tuple[str, str, str]:
        """
        Generate per-position activity distribution plots.
        
        Args:
            min_samples: Minimum samples to show individual substituent; 
                        smaller groups merged to "Others".
        
        Returns:
            Tuple of (html_div, html_file_path, csv_file_path)
        """
        if not self.rgroup_columns:
            return "", "", ""
        
        n_positions = len(self.rgroup_columns)
        fig = make_subplots(
            rows=1, cols=n_positions,
            subplot_titles=[f"{pos} Distribution" for pos in self.rgroup_columns]
        )
        
        all_data = []
        
        for i, pos in enumerate(self.rgroup_columns, 1):
            pos_data = self.df[[pos, "pActivity", "compound_id"]].dropna()
            
            # Count substituents and merge rare ones
            sub_counts = pos_data[pos].value_counts()
            rare_subs = sub_counts[sub_counts < min_samples].index.tolist()
            
            pos_data = pos_data.copy()
            pos_data.loc[pos_data[pos].isin(rare_subs), pos] = "Others"
            
            # Collect for CSV
            for _, row in pos_data.iterrows():
                all_data.append({
                    "Position": pos,
                    "Substituent": row[pos],
                    "pActivity": row["pActivity"],
                    "compound_id": row["compound_id"],
                })
            
            # Box plot
            for sub in pos_data[pos].unique():
                sub_vals = pos_data[pos_data[pos] == sub]["pActivity"]
                fig.add_trace(
                    go.Box(
                        y=sub_vals,
                        name=str(sub)[:15],
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(size=5),
                        showlegend=False,
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            title="Per-position Activity Distribution",
            height=500,
        )
        
        for i in range(1, n_positions + 1):
            fig.update_yaxes(title_text="pActivity" if i == 1 else "", row=1, col=i)
        
        # Save
        dist_df = pd.DataFrame(all_data)
        csv_path = self._save_csv(dist_df, "activity_distribution.csv")
        html_path = self._save_html(fig, "activity_distribution.html")
        html_div = self._fig_to_html_div(fig)
        
        return html_div, html_path, csv_path
    
    # =========================================================================
    # 7. Lead vs Backup Radar Chart
    # =========================================================================
    
    def plot_lead_radar(
        self, 
        candidate_ids: List[str] = None,
        metrics: List[str] = None
    ) -> Tuple[str, str, str]:
        """
        Generate Lead vs Backup comparison radar chart.
        
        Args:
            candidate_ids: List of compound IDs to compare. If None, uses top 2-5.
            metrics: List of metrics to compare. If None, uses defaults.
        
        Returns:
            Tuple of (html_div, html_file_path, csv_file_path)
        """
        # Default metrics
        if metrics is None:
            metrics = ["pActivity", "MW", "cLogP", "TPSA", "HBD", "HBA"]
        
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        if not available_metrics:
            return "", "", ""
        
        # Select candidates
        if candidate_ids:
            candidates = self.df[self.df["compound_id"].isin(candidate_ids)]
        else:
            # Select top 3 by pActivity
            candidates = self.df.nlargest(min(3, len(self.df)), "pActivity")
        
        if len(candidates) < 2:
            return "", "", ""
        
        # Normalize values to 0-1 scale
        normalized = pd.DataFrame()
        normalized["compound_id"] = candidates["compound_id"]
        
        for metric in available_metrics:
            col = candidates[metric]
            min_val, max_val = col.min(), col.max()
            if max_val > min_val:
                # For activity, higher is better; for MW, lower is better
                if metric in ["MW", "HBD", "HBA", "RB"]:
                    normalized[metric] = 1 - (col - min_val) / (max_val - min_val)
                else:
                    normalized[metric] = (col - min_val) / (max_val - min_val)
            else:
                normalized[metric] = 0.5
        
        # Create radar chart
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2[:len(candidates)]
        
        for idx, (_, row) in enumerate(normalized.iterrows()):
            values = [row[m] for m in available_metrics]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                fill="toself",
                name=row["compound_id"],
                line_color=colors[idx % len(colors)],
                opacity=0.6,
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Lead vs Backup Candidate Comparison",
            showlegend=True,
            height=500,
        )
        
        main_chart_html = self._fig_to_html_div(fig)
        
        # Create score table
        score_df = candidates[["compound_id", "smiles"] + available_metrics].copy()
        score_df["Total_Score"] = normalized[available_metrics].sum(axis=1)
        score_df = score_df.sort_values("Total_Score", ascending=False)
        
        # Generate structure gallery for candidates
        structure_html = '<div style="margin-top:2rem;"><h3>Candidate Structures</h3>'
        structure_html += '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));gap:1rem;margin-top:1rem;">'
        
        for idx, (_, cand) in enumerate(candidates.iterrows()):
            smi = cand.get("smiles", "")
            svg = mol_to_svg(smi, width=200, height=150) if smi else "<div>No structure</div>"
            pact = cand.get("pActivity", 0)
            mw = cand.get("MW", 0)
            clogp = cand.get("cLogP", 0)
            
            # Determine rank badge color
            rank = idx + 1
            badge_color = "#059669" if rank == 1 else "#0ea5e9" if rank == 2 else "#64748b"
            
            structure_html += f'''
            <div style="background:white;border-radius:0.5rem;padding:1rem;border:1px solid #e2e8f0;text-align:center;">
                <div style="margin-bottom:0.5rem;">
                    <span style="background:{badge_color};color:white;padding:0.25rem 0.75rem;border-radius:1rem;font-size:0.75rem;font-weight:600;">
                        #{rank} {cand["compound_id"]}
                    </span>
                </div>
                <div style="background:#f8fafc;padding:0.5rem;border-radius:4px;display:inline-block;">{svg}</div>
                <div style="margin-top:0.5rem;font-size:0.75rem;color:#64748b;">
                    pActivity: <b>{pact:.2f}</b> | MW: {mw:.0f} | cLogP: {clogp:.1f}
                </div>
            </div>
            '''
        
        structure_html += '</div></div>'
        
        # Combine chart and structures
        html_div = main_chart_html + structure_html
        
        # Save
        csv_path = self._save_csv(score_df[["compound_id"] + available_metrics + ["Total_Score"]], "lead_score_table.csv")
        
        # Save full HTML with structures
        full_html = f"""<!DOCTYPE html>
<html><head><title>Lead Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body style="font-family:sans-serif;padding:2rem;">
<h1>Lead vs Backup Candidate Comparison</h1>
{html_div}
</body></html>"""
        html_path = os.path.join(self.output_dir, "lead_radar.html")
        with open(html_path, "w") as f:
            f.write(full_html)
        
        return html_div, html_path, csv_path
    
    # =========================================================================
    # 8. SAR Timeline
    # =========================================================================
    
    def plot_sar_timeline(self) -> Tuple[str, str, str]:
        """
        Generate SAR iteration timeline plot.
        
        Returns:
            Tuple of (html_div, html_file_path, csv_file_path)
        """
        # Check for iteration column
        has_iteration = "iteration" in self.df.columns and self.df["iteration"].notna().any()
        
        if has_iteration:
            timeline_df = self.df.dropna(subset=["iteration", "pActivity"])
            timeline_df = timeline_df.sort_values("iteration")
            x_col = "iteration"
            x_title = "Iteration"
        else:
            # Use index as proxy
            timeline_df = self.df.dropna(subset=["pActivity"]).reset_index(drop=True)
            timeline_df["_index"] = range(len(timeline_df))
            x_col = "_index"
            x_title = "Compound Sequence"
        
        if len(timeline_df) < 2:
            return "", "", ""
        
        # Get top-k per iteration/group
        if has_iteration:
            top_df = (timeline_df.groupby("iteration")
                      .apply(lambda g: g.nlargest(self.config.top_k_timeline, "pActivity"))
                      .reset_index(drop=True))
        else:
            top_df = timeline_df.copy()
        
        # Create figure
        fig = go.Figure()
        
        # Line for trajectory
        fig.add_trace(go.Scatter(
            x=top_df[x_col],
            y=top_df["pActivity"],
            mode="lines+markers",
            marker=dict(size=10, color="#0ea5e9"),
            line=dict(width=2, color="#0f172a"),
            text=top_df["compound_id"],
            hovertemplate="<b>%{text}</b><br>pActivity: %{y:.2f}<extra></extra>",
            name="Top Compounds",
        ))
        
        # Highlight significant improvements
        if len(top_df) > 1:
            improvements = []
            prev_val = top_df.iloc[0]["pActivity"]
            for i, row in top_df.iloc[1:].iterrows():
                delta = row["pActivity"] - prev_val
                if delta > self.config.delta_activity_cutoff:
                    improvements.append({
                        x_col: row[x_col],
                        "pActivity": row["pActivity"],
                        "compound_id": row["compound_id"],
                        "delta": delta,
                        "key_change": row.get("key_change", ""),
                    })
                prev_val = row["pActivity"]
            
            if improvements:
                imp_df = pd.DataFrame(improvements)
                fig.add_trace(go.Scatter(
                    x=imp_df[x_col],
                    y=imp_df["pActivity"],
                    mode="markers+text",
                    marker=dict(size=15, color="#059669", symbol="star"),
                    text=imp_df["compound_id"],
                    textposition="top center",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "pActivity: %{y:.2f}<br>"
                        "Improvement: +%{customdata[0]:.2f}<br>"
                        "%{customdata[1]}<extra></extra>"
                    ),
                    customdata=imp_df[["delta", "key_change"]].values,
                    name="Significant Improvements",
                ))
        
        fig.update_layout(
            title="SAR Evolution Timeline",
            xaxis_title=x_title,
            yaxis_title="pActivity",
            height=450,
            showlegend=True,
        )
        
        main_chart_html = self._fig_to_html_div(fig)
        
        # Summary stats
        best_idx = timeline_df["pActivity"].idxmax()
        best_row = timeline_df.loc[best_idx]
        summary = {
            "Total_Compounds": len(timeline_df),
            "Best_pActivity": timeline_df["pActivity"].max(),
            "Best_Compound": best_row["compound_id"],
            "Activity_Improvement": timeline_df["pActivity"].max() - timeline_df["pActivity"].iloc[0] if len(timeline_df) > 1 else 0,
        }
        summary_df = pd.DataFrame([summary])
        
        # Generate milestone structure gallery (first, best, latest)
        milestones = []
        
        # First compound
        first_row = timeline_df.iloc[0]
        milestones.append(("🚀 First", first_row))
        
        # Best compound
        milestones.append(("⭐ Best", best_row))
        
        # Latest compound (if different from best)
        last_row = timeline_df.iloc[-1]
        if last_row["compound_id"] != best_row["compound_id"]:
            milestones.append(("📍 Latest", last_row))
        
        structure_html = '<div style="margin-top:2rem;"><h3>Key Milestone Compounds</h3>'
        structure_html += '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:1rem;margin-top:1rem;">'
        
        for label, row in milestones:
            smi = row.get("smiles", "")
            svg = mol_to_svg(smi, width=180, height=140) if smi else "<div>No structure</div>"
            pact = row.get("pActivity", 0)
            cpd_id = row.get("compound_id", "Unknown")
            
            structure_html += f'''
            <div style="background:white;border-radius:0.5rem;padding:1rem;border:1px solid #e2e8f0;text-align:center;">
                <div style="margin-bottom:0.5rem;font-weight:600;color:#0f172a;">{label}</div>
                <div style="background:#f8fafc;padding:0.5rem;border-radius:4px;display:inline-block;">{svg}</div>
                <div style="margin-top:0.5rem;">
                    <span style="font-size:0.75rem;color:#64748b;">{cpd_id}</span>
                </div>
                <div style="font-weight:600;color:#059669;">pActivity: {pact:.2f}</div>
            </div>
            '''
        
        structure_html += '</div></div>'
        
        # Combine chart and structures
        html_div = main_chart_html + structure_html
        
        # Save
        csv_path = self._save_csv(summary_df, "sar_timeline_summary.csv")
        
        # Save full HTML with structures
        full_html = f"""<!DOCTYPE html>
<html><head><title>SAR Timeline</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body style="font-family:sans-serif;padding:2rem;">
<h1>SAR Evolution Timeline</h1>
{html_div}
</body></html>"""
        html_path = os.path.join(self.output_dir, "sar_timeline.html")
        with open(html_path, "w") as f:
            f.write(full_html)
        
        return html_div, html_path, csv_path
    
    # =========================================================================
    # Generate All Plots
    # =========================================================================
    
    def generate_all(
        self, 
        scaffold_smiles: str = None,
        position_rules: Dict[str, str] = None,
        candidate_ids: List[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate all 8 visualization types.
        
        Args:
            scaffold_smiles: Core scaffold for annotation.
            position_rules: Position classification for scaffold annotation.
            candidate_ids: Candidate IDs for lead comparison.
        
        Returns:
            Dictionary with plot names as keys and {html_div, html_path, csv_path} as values.
        """
        results = {}
        
        # 1. Position Heatmap
        logger.info("Generating Position SAR Heatmap...")
        html, path, csv = self.plot_position_sar_heatmap()
        results["position_heatmap"] = {"html_div": html, "html_path": path, "csv_path": csv}
        
        # 2. MMP Analysis
        logger.info("Generating MMP Analysis...")
        html, path, csv = self.plot_mmp_delta_activity()
        results["mmp_analysis"] = {"html_div": html, "html_path": path, "csv_path": csv}
        
        # 3. Functional Group Matrix
        logger.info("Generating Functional Group Matrix...")
        html, path, csv = self.plot_functional_group_matrix()
        results["fg_matrix"] = {"html_div": html, "html_path": path, "csv_path": csv}
        
        # 4. Property-Activity
        logger.info("Generating Property-Activity Plots...")
        html, path, csv = self.plot_property_activity()
        results["property_activity"] = {"html_div": html, "html_path": path, "csv_path": csv}
        
        # 5. Scaffold Annotation
        if scaffold_smiles:
            logger.info("Generating Scaffold Annotation...")
            html, path = self.plot_scaffold_annotation(scaffold_smiles, position_rules)
            results["scaffold_annotation"] = {"html_div": html, "html_path": path, "csv_path": ""}
        
        # 6. Activity Distribution
        logger.info("Generating Activity Distribution...")
        html, path, csv = self.plot_activity_distribution()
        results["activity_distribution"] = {"html_div": html, "html_path": path, "csv_path": csv}
        
        # 7. Lead Radar
        logger.info("Generating Lead Radar...")
        html, path, csv = self.plot_lead_radar(candidate_ids)
        results["lead_radar"] = {"html_div": html, "html_path": path, "csv_path": csv}
        
        # 8. Timeline
        logger.info("Generating SAR Timeline...")
        html, path, csv = self.plot_sar_timeline()
        results["timeline"] = {"html_div": html, "html_path": path, "csv_path": csv}
        
        logger.info(f"Generated {len(results)} visualizations")
        return results


# =============================================================================
# Convenience Function
# =============================================================================

def generate_sar_visualizations(
    df: pd.DataFrame,
    scaffold_smiles: str = None,
    position_rules: Dict[str, str] = None,
    candidate_ids: List[str] = None,
    config: SARVisualizerConfig = None
) -> Dict[str, Dict[str, str]]:
    """
    Convenience function to preprocess data and generate all SAR visualizations.
    
    Args:
        df: Raw DataFrame with compound data.
        scaffold_smiles: Core scaffold for annotation.
        position_rules: Position classification for scaffold annotation.
        candidate_ids: Candidate IDs for lead comparison.
        config: Visualization configuration.
    
    Returns:
        Dictionary with visualization results.
    """
    # Preprocess
    preprocessor = SARDataPreprocessor(df)
    processed_df = preprocessor.preprocess()
    
    # Generate visualizations
    visualizer = SARVisualizerAdvanced(processed_df, config)
    return visualizer.generate_all(scaffold_smiles, position_rules, candidate_ids)
