
import base64
import io
import logging
from typing import Any, List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, QED

# Set backend to Agg for headless environments
plt.switch_backend('Agg')
sns.set_theme(style="whitegrid")

logger = logging.getLogger(__name__)

class SARVisualizer:
    """Tool for generating SAR visualization plots."""

    def __init__(self):
        self.colors = sns.color_palette("viridis", as_cmap=True)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    def plot_position_sar_heatmap(self, r_group_data: Dict[str, Any]) -> str:
        """1. Position-wise SAR Map (Heatmap)."""
        try:
            analysis = r_group_data.get("r_group_analysis", {})
            if not analysis:
                return ""

            # Prepare data for heatmap
            data = []
            for pos, subs in analysis.items():
                for sub, stats in subs.items():
                    data.append({
                        "Position": pos,
                        "Substituent": sub,
                        "Activity": stats.get("mean_activity", 0)
                    })
            
            if not data:
                return ""

            df = pd.DataFrame(data)
            pivot_table = df.pivot(index="Substituent", columns="Position", values="Activity")
            
            # Filter to top 15 substituents by frequency if too many
            if len(pivot_table) > 15:
                top_subs = df["Substituent"].value_counts().head(15).index
                pivot_table = pivot_table.loc[pivot_table.index.intersection(top_subs)]

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar_kws={'label': 'Activity'})
            ax.set_title("Position-wise SAR Heatmap")
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error plotting position SAR heatmap: {e}")
            return ""

    def plot_mmp_activity_diff(self, cliffs_data: Dict[str, Any]) -> str:
        """2. Matched Molecular Pair (MMP) Delta Activity Plot."""
        try:
            cliffs = cliffs_data.get("cliffs", [])
            if not cliffs:
                return ""

            fold_changes = [c.get("fold_change", 1.0) for c in cliffs]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(fold_changes, bins=20, kde=True, color="skyblue", ax=ax)
            ax.set_title("MMP Activity Differences Distribution")
            ax.set_xlabel("Fold Change")
            ax.set_ylabel("Count")
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error plotting MMP activity diff: {e}")
            return ""

    def plot_fg_necessity_matrix(self, fg_data: Dict[str, Any]) -> str:
        """3. Functional Group Necessity Matrix."""
        try:
            fg_list = fg_data.get("functional_group_sar", []) if isinstance(fg_data, dict) else fg_data
            if not fg_list:
                return ""

            # Data: FG vs Fold Change (Impact)
            fgs = [item["functional_group"] for item in fg_list]
            impacts = [item.get("fold_change", 1.0) for item in fg_list]
            effects = [item.get("effect", "neutral") for item in fg_list]
            
            # Map effects to colors
            color_map = {
                "essential": "green",
                "beneficial": "blue",
                "tolerated": "gray",
                "detrimental": "red"
            }
            colors = [color_map.get(e, "gray") for e in effects]

            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(fgs))
            ax.barh(y_pos, impacts, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(fgs)
            ax.set_xlabel("Fold Change (Impact)")
            ax.set_title("Functional Group Necessity")
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=c, label=l) for l, c in color_map.items()]
            ax.legend(handles=legend_elements, loc='lower right')
            
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error plotting FG necessity matrix: {e}")
            return ""

    def plot_property_activity(self, compounds: List[Dict[str, Any]]) -> str:
        """4. Property-Activity Plot (Activity vs MW/LogP)."""
        try:
            data = []
            for c in compounds:
                smi = c.get("smiles")
                act = c.get("activity")
                if smi and act is not None:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        data.append({
                            "Activity": float(act),
                            "MW": Descriptors.MolWt(mol),
                            "LogP": Descriptors.MolLogP(mol),
                            "TPSA": Descriptors.TPSA(mol)
                        })
            
            if not data:
                return ""
                
            df = pd.DataFrame(data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            sns.scatterplot(data=df, x="MW", y="Activity", hue="LogP", size="TPSA", sizes=(20, 200), ax=ax1, palette="viridis")
            ax1.set_title("Activity vs MW (colored by LogP)")
            
            sns.scatterplot(data=df, x="LogP", y="Activity", hue="MW", size="TPSA", sizes=(20, 200), ax=ax2, palette="magma")
            ax2.set_title("Activity vs LogP (colored by MW)")
            
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error plotting property-activity: {e}")
            return ""

    def plot_scaffold_annotation(self, scaffold_smiles: str, r_positions: List[str]) -> str:
        """5. 2D Scaffold Annotation (SAR Heatmap Schematic)."""
        try:
            if not scaffold_smiles:
                return ""
                
            mol = Chem.MolFromSmiles(scaffold_smiles)
            if not mol:
                return ""
            
            AllChem.Compute2DCoords(mol)
            
            # For now, just draw the scaffold with R-group labels if possible
            # This is a simplified version. A full heatmap requires mapping R-groups to atoms.
            # We will just draw the scaffold clearly.
            
            dopts = Draw.MolDrawOptions()
            dopts.addAtomIndices = True
            dopts.bondLineWidth = 2
            
            img = Draw.MolToImage(mol, size=(400, 300), options=dopts)
            
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error plotting scaffold annotation: {e}")
            return ""

    def plot_activity_distribution(self, r_group_data: Dict[str, Any]) -> str:
        """6. Per-position Activity Distribution."""
        try:
            analysis = r_group_data.get("r_group_analysis", {})
            if not analysis:
                return ""
                
            data = []
            for pos, subs in analysis.items():
                for sub, stats in subs.items():
                    # We don't have raw values here, only stats. 
                    # If we had raw values it would be better.
                    # Let's use mean_activity repeated 'count' times as a proxy, or just plot means.
                    count = stats.get("count", 1)
                    mean = stats.get("mean_activity", 0)
                    for _ in range(count):
                        data.append({"Position": pos, "Activity": mean})
            
            if not data:
                return ""
                
            df = pd.DataFrame(data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x="Position", y="Activity", ax=ax, inner="stick", palette="pastel")
            sns.stripplot(data=df, x="Position", y="Activity", ax=ax, color="black", alpha=0.5)
            ax.set_title("Activity Distribution by R-Group Position")
            
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error plotting activity distribution: {e}")
            return ""

    def plot_radar_chart(self, compounds: List[Dict[str, Any]]) -> str:
        """7. Lead vs Backup Radar Chart."""
        try:
            # Select top 2 compounds by activity
            valid_compounds = [c for c in compounds if c.get("activity") is not None and c.get("smiles")]
            valid_compounds.sort(key=lambda x: float(x["activity"]), reverse=True) # Assuming higher is better? Or lower? Usually lower IC50 is better.
            # Let's assume input is activity (higher better) or pIC50. If IC50, we should invert or handle.
            # Let's assume the user provided normalized activity or pIC50.
            
            if len(valid_compounds) < 2:
                return ""
                
            top2 = valid_compounds[:2]
            
            # Metrics: Activity, QED, LogP (normalized), MW (normalized), TPSA (normalized)
            categories = ['Activity', 'QED', 'Lipophilicity', 'Size Efficiency', 'Polarity']
            
            data = []
            for c in top2:
                mol = Chem.MolFromSmiles(c["smiles"])
                act = float(c["activity"])
                qed = QED.qed(mol)
                logp = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = Descriptors.TPSA(mol)
                
                # Normalize (very rough normalization for visualization)
                # Ideally should be based on project criteria
                norm_act = min(act / 10.0, 1.0) # Assume max activity ~10
                norm_logp = 1.0 - min(abs(logp - 3) / 5.0, 1.0) # Optimal ~3
                norm_mw = 1.0 - min(mw / 800.0, 1.0) # Lower is better, max 800
                norm_tpsa = 1.0 - min(abs(tpsa - 100) / 100.0, 1.0) # Optimal ~100
                
                data.append([norm_act, qed, norm_logp, norm_mw, norm_tpsa])
                
            # Plot
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            
            colors = ['#0ea5e9', '#64748b']
            labels = ['Lead', 'Backup']
            
            for i, d in enumerate(data):
                val = d + d[:1]
                ax.plot(angles, val, linewidth=2, linestyle='solid', label=labels[i], color=colors[i])
                ax.fill(angles, val, color=colors[i], alpha=0.25)
                
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            ax.set_title("Lead vs Backup Candidate Profile")
            
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error plotting radar chart: {e}")
            return ""

    def plot_sar_timeline(self, compounds: List[Dict[str, Any]]) -> str:
        """8. SAR Iteration Timeline."""
        try:
            # Check if we have date or just use index as proxy for time
            # Assuming compounds are ordered by synthesis/testing time
            
            activities = []
            ids = []
            for c in compounds:
                act = c.get("activity")
                if act is not None:
                    activities.append(float(act))
                    ids.append(c.get("compound_id", ""))
            
            if not activities:
                return ""
                
            df = pd.DataFrame({"Index": range(len(activities)), "Activity": activities, "ID": ids})
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x="Index", y="Activity", marker="o", ax=ax, color="#0f172a")
            
            # Highlight top compounds
            top_indices = df.nlargest(3, "Activity").index
            for idx in top_indices:
                row = df.loc[idx]
                ax.annotate(row["ID"], (row["Index"], row["Activity"]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_title("SAR Evolution Timeline")
            ax.set_xlabel("Iteration (Compound Sequence)")
            ax.set_ylabel("Activity")
            
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error plotting timeline: {e}")
            return ""
