
import os
import sys
import json
from datetime import datetime

import importlib.util

spec = importlib.util.spec_from_file_location("html_builder", "/data/worksapce/molx_agent/molx_agent/tools/html_builder.py")
html_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(html_builder)

build_sar_html_report = html_builder.build_sar_html_report
save_html_report = html_builder.save_html_report

def test_generate_report():
    # Dummy data
    sar_results = {
        "total_compounds": 5,
        "generated_at": datetime.now().isoformat(),
        "compounds": [
            {"compound_id": "Cpd-1", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "activity": 0.5},
            {"compound_id": "Cpd-2", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "activity": 1.2},
            {"compound_id": "Cpd-3", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "activity": 10.5},
        ],
        "r_group_analysis": {
            "decomposed_compounds": [
                {
                    "compound_id": "Cpd-1",
                    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "activity": 0.5,
                    "r_groups": {"R1": "CH3", "R2": "OH"}
                },
                {
                    "compound_id": "Cpd-2",
                    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "activity": 1.2,
                    "r_groups": {"R1": "CH2CH3", "R2": "OH"}
                }
            ],
            "recommendations": [
                {"position": "R1", "best_substituent": "CH3", "best_activity": 0.5, "worst_substituent": "Ph", "worst_activity": 100}
            ],
            "r_group_analysis": {
                "R1": {
                    "CH3": {"count": 5, "mean_activity": 0.5, "min_activity": 0.1, "max_activity": 1.0}
                }
            }
        },
        "scaffold": "c1ccccc1",
        "scaffold_strategy": "mcs",
        "functional_group_sar": {
            "functional_group_sar": [
                {"functional_group": "hydroxyl", "effect": "essential", "fold_change": 0.1, "avg_activity_with": 0.5, "avg_activity_without": 5.0}
            ]
        },
        "activity_cliffs": {
            "activity_cliffs_found": 1,
            "cliffs": [
                {
                    "mol1": "CC(=O)Oc1ccccc1C(=O)O",
                    "mol2": "CC(=O)Oc1ccccc1C(=O)OC",
                    "similarity": 0.85,
                    "activity1": 0.5,
                    "activity2": 50.0,
                    "fold_change": 100.0
                }
            ]
        }
    }

    print("Building report...")
    html = build_sar_html_report(sar_results, "Test SAR Report")
    
    print("Saving report...")
    path = save_html_report(html, "test_report_v2.html")
    print(f"Report saved to: {path}")

if __name__ == "__main__":
    test_generate_report()
