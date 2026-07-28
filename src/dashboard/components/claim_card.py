from __future__ import annotations
import streamlit as st
from src.consistency.aggregator import VERDICT_COLORS, VERDICT_EMOJI
ESG_COLORS = {'E': '#22c55e', 'S': '#3b82f6', 'G': '#a855f7', 'MIXED': '#94a3b8'}
FLAG_COLORS = {'LIKELY_CONSISTENT': '#22c55e', 'NEEDS_REVIEW': '#f59e0b', 'HIGH_RISK': '#ef4444'}

def render_claim_card(item: dict, idx: int) -> None:
    claim = item['claim']
    result = item['classification']
    agg = item['agg_result']
    verdict_color = VERDICT_COLORS.get(agg.verdict, '#94a3b8')
    verdict_emoji = VERDICT_EMOJI.get(agg.verdict, '❓')
    esg_color = ESG_COLORS.get(result.esg_label, '#94a3b8')
    flag_color = FLAG_COLORS.get(result.consistency_flag, '#94a3b8')
    html_content = f"""\n<div style="background: #1e293b; border: 1px solid #334155; border-left: 4px solid {verdict_color}; border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 0.5rem;">\n<div style="display:flex; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap;">\n<span style="background: {esg_color}22; color: {esg_color}; border: 1px solid {esg_color}; padding: 2px 10px; border-radius: 999px; font-size: 0.8em; font-weight: 700;">{result.esg_label}</span>\n<span style="background: {flag_color}22; color: {flag_color}; border: 1px solid {flag_color}; padding: 2px 10px; border-radius: 999px; font-size: 0.8em;">{result.consistency_flag.replace('_', ' ')}</span>\n<span style="background: {verdict_color}22; color: {verdict_color}; border: 1px solid {verdict_color}; padding: 2px 10px; border-radius: 999px; font-size: 0.8em; font-weight: 600;">{verdict_emoji} {agg.verdict.replace('_', ' ')}</span>\n<span style="color:#64748b; font-size:0.8em;">Section: {claim.source_section.title()} &nbsp;|&nbsp; NLP Conf: {result.confidence:.0%} &nbsp;|&nbsp; Audit Risk: {agg.risk_score:.2f}</span>\n</div>\n<p style="color:#e2e8f0; font-size:0.95em; margin:0; line-height:1.6;"><b>Atomic Claim:</b> {claim.text}</p>\n<p style="color:#cbd5e1; font-size:0.9em; margin-top:8px; line-height:1.5;"><b>Original Sentence:</b> {getattr(claim, 'evidence_sentence', '') or claim.text}</p>\n{(f'<p style="color:#94a3b8; font-size:0.85em; margin-top:8px; line-height:1.5;"><b>Context Window:</b> <i>"{claim.evidence}"</i></p>' if getattr(claim, 'evidence', None) else '')}\n{(f'<p style="color:#f87171; font-size:0.8em; margin-top:8px; margin-bottom:0;">⚠️ {agg.summary}</p>' if agg.verdict != 'CONSISTENT' else '')}\n</div>\n"""
    st.markdown(html_content, unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric('Claim Type', claim.claim_type.title())
    col2.metric('ESG Label', result.esg_label)
    col3.metric('Audit Risk', f'{agg.risk_score:.3f}')
    col4.metric('L1 Check', item['l1_result'].status if item.get('l1_result') else '—')
    col5.metric('L2 (Historical)', item['l2_result'].status if item.get('l2_result') else '—')
    col6.metric('L3 (SBTi)', item['l3_result'].status if item.get('l3_result') else '—')