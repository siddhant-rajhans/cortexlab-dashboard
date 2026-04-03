"""Futuristic theme injection for Streamlit."""

import streamlit as st
from pathlib import Path


def inject_theme():
    """Inject custom CSS and hide default Streamlit chrome."""
    css_path = Path(__file__).parent / "assets" / "theme.css"
    if css_path.exists():
        css = css_path.read_text()
    else:
        css = ""

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero_header(title, subtitle="", github_url="https://github.com/siddhant-rajhans/cortexlab"):
    """Render a futuristic hero header with gradient title."""
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 40%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
            letter-spacing: -0.04em;
        ">{title}</h1>
        <p style="color: #94A3B8; font-size: 1.1rem; margin-bottom: 0.8rem;">{subtitle}</p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <a href="{github_url}" target="_blank" style="
                display: inline-flex; align-items: center; gap: 0.4rem;
                padding: 0.5rem 1.2rem;
                background: rgba(124, 58, 237, 0.15);
                border: 1px solid rgba(124, 58, 237, 0.3);
                border-radius: 8px;
                color: #C4B5FD;
                text-decoration: none;
                font-size: 0.85rem;
                font-weight: 500;
                transition: all 0.3s ease;
            ">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
                GitHub
            </a>
            <a href="https://huggingface.co/SID2000/cortexlab" target="_blank" style="
                display: inline-flex; align-items: center; gap: 0.4rem;
                padding: 0.5rem 1.2rem;
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 8px;
                color: #93C5FD;
                text-decoration: none;
                font-size: 0.85rem;
                font-weight: 500;
            ">HuggingFace</a>
            <a href="https://huggingface.co/spaces/SID2000/cortexlab-dashboard" target="_blank" style="
                display: inline-flex; align-items: center; gap: 0.4rem;
                padding: 0.5rem 1.2rem;
                background: rgba(6, 182, 212, 0.15);
                border: 1px solid rgba(6, 182, 212, 0.3);
                border-radius: 8px;
                color: #67E8F9;
                text-decoration: none;
                font-size: 0.85rem;
                font-weight: 500;
            ">Live Demo</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def glow_card(title, value, subtitle="", color="#06B6D4"):
    """Render a glowing metric card."""
    st.markdown(f"""
    <div style="
        background: rgba(15, 15, 40, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid {color}33;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 0 20px {color}15;
        transition: all 0.3s ease;
    ">
        <div style="color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;">{title}</div>
        <div style="color: {color}; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;">{value}</div>
        <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.2rem;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, description=""):
    """Render a styled section header with optional description."""
    st.markdown(f"""
    <div style="margin: 1.5rem 0 0.8rem 0;">
        <h2 style="
            color: #F1F5F9;
            font-size: 1.3rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(100, 100, 255, 0.15);
        ">{title}</h2>
        {"<p style='color: #94A3B8; font-size: 0.85rem; margin-top: 0;'>" + description + "</p>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def feature_card(icon, title, description, color="#7C3AED"):
    """Render a feature card for the home page."""
    return f"""
    <div style="
        background: rgba(15, 15, 40, 0.4);
        backdrop-filter: blur(8px);
        border: 1px solid {color}25;
        border-radius: 14px;
        padding: 1.3rem;
        height: 100%;
        transition: all 0.3s ease;
        cursor: pointer;
    ">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="color: {color}; font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem;">{title}</div>
        <div style="color: #94A3B8; font-size: 0.8rem; line-height: 1.5;">{description}</div>
    </div>
    """
