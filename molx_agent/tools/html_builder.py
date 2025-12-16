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
    --primary: #2563eb;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-600: #4b5563;
    --gray-800: #1f2937;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--gray-50);
    color: var(--gray-800);
    line-height: 1.6;
}
.container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
.header {
    background: linear-gradient(135deg, var(--primary), #1e40af);
    color: white;
    padding: 3rem 2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
}
.header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
.header .meta { opacity: 0.9; }
.card {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.card h2 {
    color: var(--primary);
    border-bottom: 2px solid var(--gray-100);
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
}
.card h3 { color: var(--gray-600); margin: 1rem 0 0.5rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--gray-200); }
th { background: var(--gray-100); font-weight: 600; color: var(--gray-600); }
tr:hover { background: var(--gray-50); }
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
}
.badge-essential { background: #dcfce7; color: #166534; }
.badge-beneficial { background: #dbeafe; color: #1e40af; }
.badge-tolerated { background: #fef3c7; color: #92400e; }
.badge-detrimental { background: #fee2e2; color: #991b1b; }
.conclusion {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-left: 4px solid var(--primary);
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 0 0.5rem 0.5rem 0;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.stat-card {
    background: var(--gray-50);
    padding: 1.5rem;
    border-radius: 0.5rem;
    text-align: center;
}
.stat-value { font-size: 2rem; font-weight: 700; color: var(--primary); }
.stat-label { color: var(--gray-600); font-size: 0.875rem; margin-top: 0.25rem; }
.smiles { font-family: monospace; font-size: 0.75rem; word-break: break-all; }
.footer { text-align: center; padding: 2rem; color: var(--gray-600); }
"""


# =============================================================================
# Section Builders
# =============================================================================

def build_stats_section(sar_data: dict) -> str:
    """Build summary statistics section."""
    stats = [("总化合物数", sar_data.get("total_compounds", 0))]

    cliffs = sar_data.get("activity_cliffs", {})
    if cliffs.get("activity_cliffs_found"):
        stats.append(("活性悬崖", cliffs["activity_cliffs_found"]))

    fg = sar_data.get("functional_group_sar", {})
    if fg.get("functional_group_sar"):
        essential = len([f for f in fg["functional_group_sar"] if f.get("effect") == "essential"])
        if essential > 0:
            stats.append(("关键官能团", essential))

    html = '<div class="card"><h2>📈 Summary Statistics</h2><div class="stats-grid">'
    for label, value in stats:
        html += f'''
        <div class="stat-card">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
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

    html = '<div class="card"><h2>🧪 R-group 分析 (R-Group SAR)</h2>'

    # Activity range
    activity_range = rgroup_data.get("activity_range", {})
    if activity_range.get("min") is not None:
        html += f'''
        <p><strong>活性范围:</strong> {activity_range.get("min")} - {activity_range.get("max")}</p>
        '''

    # Recommendations table
    if recommendations:
        html += '''
        <h3>位点最优取代基</h3>
        <table>
            <thead>
                <tr>
                    <th>位点</th>
                    <th>最佳取代基</th>
                    <th>最佳活性</th>
                    <th>最差取代基</th>
                    <th>最差活性</th>
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

        html += '</tbody></table>'

    # Detailed R-group analysis
    if r_group_analysis:
        html += '<h3>R-group 详细分析</h3>'
        for r_name, values in r_group_analysis.items():
            html += f'<h4>{r_name}</h4><table><thead><tr><th>取代基</th><th>数量</th><th>平均活性</th><th>活性范围</th></tr></thead><tbody>'
            for r_value, stats in values.items():
                html += f'''
                    <tr>
                        <td>{r_value}</td>
                        <td>{stats.get('count', '')}</td>
                        <td>{stats.get('mean_activity', '')}</td>
                        <td>{stats.get('min_activity', '')} - {stats.get('max_activity', '')}</td>
                    </tr>'''
            html += '</tbody></table>'

    # Conclusion
    if recommendations:
        best_parts = [f"{r['position']}={r['best_substituent']}" for r in recommendations[:3]]
        html += f'<div class="conclusion">💡 <strong>SAR 结论:</strong> 最优组合为 {", ".join(best_parts)}</div>'

    html += '</div>'
    return html


def build_functional_group_section(fg_data: dict) -> str:
    """Build functional group SAR section."""
    if not fg_data or "error" in fg_data:
        return ""

    fg_list = fg_data.get("functional_group_sar", [])
    if not fg_list:
        return ""

    html = '<div class="card"><h2>⚗️ 官能团 SAR (Functional Group SAR)</h2>'
    html += '''
    <table>
        <thead>
            <tr>
                <th>官能团</th>
                <th>SAR 结论</th>
                <th>倍数变化</th>
                <th>有/无平均活性</th>
            </tr>
        </thead>
        <tbody>'''

    badge_map = {
        "essential": ("badge-essential", "必需"),
        "beneficial": ("badge-beneficial", "有利"),
        "tolerated": ("badge-tolerated", "中性"),
        "detrimental": ("badge-detrimental", "有害"),
    }

    for fg in fg_list:
        effect = fg.get("effect", "")
        badge_class, effect_cn = badge_map.get(effect, ("", effect))
        html += f'''
            <tr>
                <td><strong>{fg.get('functional_group', '')}</strong></td>
                <td><span class="badge {badge_class}">{effect_cn}</span></td>
                <td>{fg.get('fold_change', '')}x</td>
                <td>{fg.get('avg_activity_with', '')} / {fg.get('avg_activity_without', '')}</td>
            </tr>'''

    html += '</tbody></table>'

    # Conclusion
    essential_fgs = [f["functional_group"] for f in fg_list if f.get("effect") == "essential"]
    if essential_fgs:
        html += f'<div class="conclusion">💡 <strong>关键发现:</strong> {", ".join(essential_fgs)} 为必需官能团，对活性至关重要。</div>'

    html += '</div>'
    return html


def build_conformational_section(conf_data: dict) -> str:
    """Build conformational SAR section."""
    if not conf_data or "error" in conf_data:
        return ""

    html = '<div class="card"><h2>🧬 构象 SAR (Conformational SAR)</h2>'

    if conf_data.get("planarity"):
        p = conf_data["planarity"]
        html += f'''
        <h3>平面性分析</h3>
        <table>
            <tr><th>平面分子数</th><td>{p.get('planar_count', 'N/A')}</td></tr>
            <tr><th>非平面分子数</th><td>{p.get('nonplanar_count', 'N/A')}</td></tr>
            <tr><th>偏好</th><td><strong>{p.get('preference', 'N/A')}</strong></td></tr>
        </table>'''

    if conf_data.get("rigidity"):
        r = conf_data["rigidity"]
        html += f'''
        <h3>刚性分析</h3>
        <table>
            <tr><th>刚性分子数</th><td>{r.get('rigid_count', 'N/A')}</td></tr>
            <tr><th>柔性分子数</th><td>{r.get('flexible_count', 'N/A')}</td></tr>
            <tr><th>偏好</th><td><strong>{r.get('preference', 'N/A')}</strong></td></tr>
        </table>'''

    if conf_data.get("conclusions"):
        for c in conf_data["conclusions"]:
            html += f'<div class="conclusion">💡 {c}</div>'

    html += '</div>'
    return html


def build_activity_cliffs_section(cliffs_data: dict) -> str:
    """Build activity cliffs section."""
    if not cliffs_data or "error" in cliffs_data:
        return ""

    cliffs = cliffs_data.get("cliffs", [])
    count = cliffs_data.get("activity_cliffs_found", 0)

    html = f'<div class="card"><h2>⚠️ 活性悬崖 (Activity Cliffs)</h2>'
    html += f'<p>发现 <strong>{count}</strong> 对活性悬崖 (结构相似度 &gt; 0.7, 活性差异 &gt; 10倍)</p>'

    if cliffs:
        html += '''
        <table>
            <thead>
                <tr>
                    <th>分子1</th>
                    <th>分子2</th>
                    <th>相似度</th>
                    <th>活性1</th>
                    <th>活性2</th>
                    <th>倍数差</th>
                </tr>
            </thead>
            <tbody>'''

        for cliff in cliffs[:10]:
            smi1 = cliff.get('mol1', '')
            smi2 = cliff.get('mol2', '')
            html += f'''
                <tr>
                    <td class="smiles">{smi1[:50]}{"..." if len(smi1) > 50 else ""}</td>
                    <td class="smiles">{smi2[:50]}{"..." if len(smi2) > 50 else ""}</td>
                    <td>{cliff.get('similarity', '')}</td>
                    <td>{cliff.get('activity1', '')}</td>
                    <td>{cliff.get('activity2', '')}</td>
                    <td><strong>{cliff.get('fold_change', '')}x</strong></td>
                </tr>'''

        html += '</tbody></table>'

        if count > 0:
            html += '<div class="conclusion">💡 活性悬崖表明小的结构变化可导致显著的活性差异，这些位点值得深入研究。</div>'

    html += '</div>'
    return html


def build_scaffold_section(scaffold_data: dict) -> str:
    """Build scaffold SAR section."""
    if not scaffold_data or "error" in scaffold_data:
        return ""

    html = '<div class="card"><h2>🔬 骨架 SAR (Scaffold SAR)</h2>'

    core = scaffold_data.get("core_scaffold", "N/A")
    essential = "是" if scaffold_data.get("scaffold_essential") else "否"

    html += f'''
    <table>
        <tr><th>核心骨架</th><td class="smiles">{core}</td></tr>
        <tr><th>骨架必需性</th><td>{essential}</td></tr>
        <tr><th>保守骨架化合物</th><td>{scaffold_data.get('scaffold_conserved', 'N/A')}</td></tr>
        <tr><th>变异骨架化合物</th><td>{scaffold_data.get('scaffold_varied', 'N/A')}</td></tr>
    </table>'''

    if scaffold_data.get("conclusion"):
        html += f'<div class="conclusion">💡 {scaffold_data["conclusion"]}</div>'

    html += '</div>'
    return html


def build_positional_section(pos_data: dict) -> str:
    """Build positional SAR section."""
    if not pos_data or "error" in pos_data:
        return ""

    positions = pos_data.get("positional_sar", [])
    if not positions:
        return ""

    html = '<div class="card"><h2>📍 位点 SAR (Positional SAR)</h2>'

    for pos in positions:
        html += f'''
        <h3>{pos.get('position', 'Unknown')}</h3>
        <table>
            <tr><th>最佳取代基</th><td>{pos.get('best_substituent', 'N/A')}</td></tr>
            <tr><th>最佳活性</th><td>{pos.get('best_activity', 'N/A')}</td></tr>
            <tr><th>尺寸偏好</th><td>{pos.get('size_preference', 'N/A')}</td></tr>
        </table>'''

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
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <div class="meta">
                <span>Generated: {timestamp}</span> |
                <span>Total Compounds: {total_compounds}</span>
            </div>
        </div>
'''

    # Add sections
    html += build_stats_section(sar_data)

    if "r_group_analysis" in sar_data:
        html += build_rgroup_section(sar_data["r_group_analysis"])

    if "scaffold_sar" in sar_data:
        html += build_scaffold_section(sar_data["scaffold_sar"])

    if "functional_group_sar" in sar_data:
        html += build_functional_group_section(sar_data["functional_group_sar"])

    if "positional_sar" in sar_data:
        html += build_positional_section(sar_data["positional_sar"])

    if "conformational_sar" in sar_data:
        html += build_conformational_section(sar_data["conformational_sar"])

    if "activity_cliffs" in sar_data:
        html += build_activity_cliffs_section(sar_data["activity_cliffs"])

    # Footer
    html += '''
        <div class="footer">
            <p>Generated by MolX SAR Agent | Powered by RDKit</p>
        </div>
    </div>
</body>
</html>'''

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
