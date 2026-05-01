"""Futuristic theme injection for Streamlit."""

import streamlit as st
from pathlib import Path


THEME_MODES = ("black", "white")


def get_theme_mode() -> str:
    """Read the active theme from session state. Defaults to ``"black"``."""
    mode = st.session_state.get("ui_theme_mode", "black")
    return mode if mode in THEME_MODES else "black"


def set_theme_mode(mode: str) -> None:
    """Set the active theme. Use ``"black"`` for pure-black background,
    ``"white"`` for pure-white background. Anything else is normalized."""
    st.session_state["ui_theme_mode"] = mode if mode in THEME_MODES else "black"


def inject_theme(mode: str | None = None):
    """Inject custom CSS for the chosen theme and hide Streamlit chrome.

    When ``mode`` is ``None`` we read the active mode from session state
    via :func:`get_theme_mode`. The base ``assets/theme.css`` is always
    included; theme-specific overrides are appended on top.
    """
    if mode is None:
        mode = get_theme_mode()

    base_css_path = Path(__file__).parent / "assets" / "theme.css"
    overlay_css_path = Path(__file__).parent / "assets" / f"theme_{mode}.css"

    css = base_css_path.read_text() if base_css_path.exists() else ""
    if overlay_css_path.exists():
        css += "\n\n" + overlay_css_path.read_text()

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
    """Render a glowing metric card.

    Layout uses the ``cl-stat-card`` CSS class so background / border /
    text colors come from theme variables and flip automatically when
    the user switches between Black and White modes. Only the accent
    color (the big number) stays per-card.
    """
    st.markdown(
        f"""
        <div class="cl-stat-card" style="--card-accent: {color};">
            <div class="cl-stat-label">{title}</div>
            <div class="cl-stat-value">{value}</div>
            <div class="cl-stat-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, description=""):
    """Render a styled section header with optional description.

    Colors come from theme variables (``--text-primary`` for the title,
    ``--text-secondary`` for the description, ``--border-glass`` for
    the underline) so the same markup reads correctly on dark and
    light backgrounds.
    """
    desc_html = (
        f'<p class="cl-section-desc">{description}</p>' if description else ""
    )
    st.markdown(
        f"""
        <div class="cl-section">
            <h2 class="cl-section-title">{title}</h2>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


_FEATURE_ICONS: dict[str, str] = {
    # Heroicons-style monochrome strokes; small currentColor SVGs so the
    # accent color flows from the parent `--card-accent` variable.
    "target": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="12" cy="12" r="9"/>'
        '<circle cx="12" cy="12" r="5"/>'
        '<circle cx="12" cy="12" r="1.5" fill="currentColor"/>'
        '</svg>'
    ),
    "bars": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M5 21V11"/><path d="M12 21V4"/><path d="M19 21V14"/>'
        '<path d="M3 21h18"/></svg>'
    ),
    "clock": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2.5"/></svg>'
    ),
    "graph": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="6" cy="7" r="2.2"/><circle cx="18" cy="7" r="2.2"/>'
        '<circle cx="6" cy="17" r="2.2"/><circle cx="18" cy="17" r="2.2"/>'
        '<circle cx="12" cy="12" r="2.4"/>'
        '<path d="M8 7h2.5M13.5 7H16M8 17h2.5M13.5 17H16M6 9.2v5.6M18 9.2v5.6"/></svg>'
    ),
    "brain": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M9 5a3 3 0 0 0-3 3v0a2.5 2.5 0 0 0-2 4 2.5 2.5 0 0 0 1 4.5A3 3 0 0 0 9 19V5z"/>'
        '<path d="M15 5a3 3 0 0 1 3 3v0a2.5 2.5 0 0 1 2 4 2.5 2.5 0 0 1-1 4.5A3 3 0 0 1 15 19V5z"/>'
        '<path d="M12 5v14"/></svg>'
    ),
    "broadcast": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="12" cy="12" r="2.5" fill="currentColor"/>'
        '<path d="M8.5 8.5a5 5 0 0 0 0 7"/><path d="M15.5 15.5a5 5 0 0 0 0-7"/>'
        '<path d="M5.5 5.5a9 9 0 0 0 0 13"/><path d="M18.5 18.5a9 9 0 0 0 0-13"/></svg>'
    ),
    "arrow": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/></svg>'
    ),
}


def feature_icon(name: str) -> str:
    """Return inline SVG markup for a named card icon. Unknown names
    fall back to a hollow circle so layout still works."""
    return _FEATURE_ICONS.get(
        name,
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.6" aria-hidden="true"><circle cx="12" cy="12" r="9"/></svg>',
    )


def feature_card(icon, title, description, color="#7C3AED"):
    """Render a feature card for the home page (no inline CTA).

    Kept for backwards compatibility. New callers should prefer
    :func:`feature_card_link` which produces a uniform-height card
    with a built-in "Open ..." link, so a row of cards lines up.

    ``icon`` may be either a name registered in ``_FEATURE_ICONS``
    (e.g. ``"target"``) or raw SVG / HTML markup.
    """
    icon_html = _FEATURE_ICONS.get(icon, icon)
    return f"""
    <span class="cl-feature-card" style="--card-accent: {color};">
        <span class="cl-feature-icon">{icon_html}</span>
        <span class="cl-feature-title">{title}</span>
        <span class="cl-feature-desc">{description}</span>
    </span>
    """


def feature_card_link(icon, title, description, href: str, color="#7C3AED"):
    """Render a feature card as a single clickable anchor element.

    Inner blocks are rendered as ``<span>`` elements with
    ``display: block`` in CSS — anchors only allow phrasing-content
    children in HTML5, and Streamlit's HTML sanitizer hoists ``<div>``
    children out of ``<a>``, splitting the card visually. Spans are
    safe.

    ``icon`` may be either a name registered in ``_FEATURE_ICONS``
    (``"target"``, ``"bars"``, ``"clock"``, ``"graph"``, ``"brain"``,
    ``"broadcast"``) or raw SVG markup.
    ``href`` is the multipage URL Streamlit exposes (e.g.
    ``./Brain_Alignment``).
    """
    icon_html = _FEATURE_ICONS.get(icon, icon)
    arrow = _FEATURE_ICONS["arrow"]
    return f"""
    <a class="cl-feature-card cl-feature-link" href="{href}"
       style="--card-accent: {color};" target="_self">
        <span class="cl-feature-icon">{icon_html}</span>
        <span class="cl-feature-title">{title}</span>
        <span class="cl-feature-desc">{description}</span>
        <span class="cl-feature-cta">
            <span class="cl-feature-cta-label">Open</span>
            <span class="cl-feature-arrow">{arrow}</span>
        </span>
    </a>
    """
