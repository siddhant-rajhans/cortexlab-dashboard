"""TRIBE v2 demo-styled UI components.

Mirrors the visual structure of `aidemos.atmeta.com/tribev2`:

* A branded top header with the project name on the left, parent
  attribution centered, and navigation links on the right.
* A two-column hero with the architecture explanation (numbered
  steps + flow diagram) on the left and the interactive demo
  (3D brain + controls + tabs + video) on the right.
* Segmented toggle controls (True / Compare / Predicted, etc.)
  styled as pill groups.
* Tab strip with thin underline indicator.

All components are pure HTML+CSS injected via ``st.markdown``;
no third-party Streamlit components needed. The CSS classes are
prefixed ``tr-`` so they don't collide with the existing theme.
"""

from __future__ import annotations

import streamlit as st


def _consume_theme_query_param() -> None:
    """Apply ``?theme=black|white`` from the URL, then strip it.

    The HTML toggle renders an anchor with ``href=?theme=<next>``;
    clicking it triggers a navigation that Streamlit converts into
    a rerun with ``st.query_params["theme"]`` set. We honor it here
    and clear the param so the URL stays clean for sharing.
    """
    from theme import get_theme_mode, set_theme_mode  # local: avoid cycle
    qp = st.query_params
    requested = qp.get("theme")
    if requested in ("black", "white") and requested != get_theme_mode():
        set_theme_mode(requested)
        try:
            del qp["theme"]
        except KeyError:
            pass
        st.rerun()


def top_header(
    title: str = "CortexLab",
    subtitle: str = "AI research toolkit",
    parent: str = "Stevens · Meta TRIBE v2",
    links: list[tuple[str, str]] | None = None,
    theme_toggle: bool = True,
) -> None:
    """Render the TRIBE-style top header bar.

    Layout: title block on the far left, parent attribution centered,
    a sliding sun/moon toggle and a row of text links on the far right.
    The toggle is rendered as an inline anchor with a CSS-only sliding
    pill; clicking it navigates to ``?theme=<next>``, which Streamlit
    serves as a rerun. :func:`_consume_theme_query_param` (called at
    the top of the script) then applies the new mode and strips the
    query param.
    """
    from theme import get_theme_mode  # local: keep theme import deferred

    if theme_toggle:
        _consume_theme_query_param()

    if links is None:
        links = [
            ("Code",  "https://github.com/siddhant-rajhans/cortexlab"),
            ("Demo",  "https://huggingface.co/spaces/SID2000/cortexlab-dashboard"),
            ("Paper", "https://github.com/siddhant-rajhans/cortexlab"),
            ("Hub",   "https://huggingface.co/SID2000/cortexlab"),
        ]
    link_html = " ".join(
        f'<a class="tr-nav-link" href="{href}" target="_blank">{label} <span class="tr-arrow">↗</span></a>'
        for label, href in links
    )

    toggle_html = ""
    if theme_toggle:
        cur_mode = get_theme_mode()
        next_mode = "white" if cur_mode == "black" else "black"
        is_on = cur_mode == "white"
        sun = (
            '<svg class="tr-toggle-icon" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
            'stroke-linejoin="round" aria-hidden="true">'
            '<circle cx="12" cy="12" r="4.2"/>'
            '<path d="M12 3v2"/><path d="M12 19v2"/>'
            '<path d="M3 12h2"/><path d="M19 12h2"/>'
            '<path d="M5.6 5.6l1.4 1.4"/><path d="M17 17l1.4 1.4"/>'
            '<path d="M5.6 18.4l1.4-1.4"/><path d="M17 7l1.4-1.4"/></svg>'
        )
        moon = (
            '<svg class="tr-toggle-icon" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
            'stroke-linejoin="round" aria-hidden="true">'
            '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
        )
        toggle_html = (
            f'<a class="tr-toggle" href="?theme={next_mode}" target="_self" '
            f'role="switch" aria-checked="{str(is_on).lower()}" '
            'aria-label="Toggle light/dark theme">'
            f'<span class="tr-toggle-track {"on" if is_on else "off"}">'
            f'  <span class="tr-toggle-side tr-toggle-left">{moon}</span>'
            f'  <span class="tr-toggle-side tr-toggle-right">{sun}</span>'
            f'  <span class="tr-toggle-knob"></span>'
            f'</span>'
            '</a>'
        )

    st.markdown(
        f"""
        <div class="tr-header">
          <div class="tr-header-left">
            <div class="tr-header-title">{title}</div>
            <div class="tr-header-subtitle">{subtitle}</div>
          </div>
          <div class="tr-header-center">{parent}</div>
          <div class="tr-header-right">{toggle_html}{link_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def architecture_steps(
    title: str = "CortexLab: a multimodal encoding pipeline",
    intro: str = "CortexLab predicts brain activity through a three-stage pipeline:",
    steps: list[tuple[str, str]] | None = None,
) -> None:
    """Numbered architecture description card.

    ``steps`` is a list of (heading, body_html) tuples. The body may
    include inline markup (e.g. underlined modality names).
    """
    if steps is None:
        steps = [
            (
                "Multimodal feature extraction",
                "Pretrained CLIP, V-JEPA 2, and CLIP-text encoders extract <u>vision</u>, <u>motion</u>, "
                "and <u>language</u> embeddings from short video clips and machine-generated captions.",
            ),
            (
                "Voxelwise ridge encoding",
                "A fused Triton kernel solves <u>327,684</u> independent voxelwise ridge regressions in seconds, "
                "mapping each modality's features to per-voxel BOLD responses on the fsaverage cortical surface.",
            ),
            (
                "Causal modality lesion",
                "Each modality is ablated at test time and per-voxel ΔR² is permutation-tested across "
                "1,000 shuffles, BH-FDR corrected, and rendered onto the inflated cortex.",
            ),
        ]
    items_html = "".join(
        f"""
        <li class="tr-step">
          <span class="tr-step-num">{i}</span>
          <div class="tr-step-body">
            <strong class="tr-step-title">{title_}</strong>
            <p class="tr-step-text">{body}</p>
          </div>
        </li>
        """
        for i, (title_, body) in enumerate(steps, start=1)
    )
    st.markdown(
        f"""
        <div class="tr-arch">
          <h2 class="tr-arch-title">{title}</h2>
          <p class="tr-arch-intro">{intro}</p>
          <ul class="tr-step-list">{items_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_diagram() -> None:
    """Flow diagram: stimulus rows → per-modality encoders → fusion → brain.

    Mirrors TRIBE's "Video → V-JEPA2", "Audio → wav2vec 2.0",
    "Text → Llama 3.2" diagram visually but with our model lineup.
    """
    st.markdown(
        """
        <div class="tr-pipeline">
          <div class="tr-pipe-rows">
            <div class="tr-pipe-row">
              <div class="tr-pipe-source"><span class="tr-pipe-icon">▶</span> Video frame</div>
              <span class="tr-pipe-arrow">→</span>
              <div class="tr-pipe-model">CLIP ViT-L/14</div>
            </div>
            <div class="tr-pipe-row">
              <div class="tr-pipe-source"><span class="tr-pipe-icon">▤</span> Motion</div>
              <span class="tr-pipe-arrow">→</span>
              <div class="tr-pipe-model">V-JEPA 2</div>
            </div>
            <div class="tr-pipe-row">
              <div class="tr-pipe-source"><span class="tr-pipe-icon">T</span> Caption</div>
              <span class="tr-pipe-arrow">→</span>
              <div class="tr-pipe-model">CLIP text</div>
            </div>
          </div>
          <span class="tr-pipe-arrow tr-pipe-arrow-h">→</span>
          <div class="tr-pipe-fuse">
            <div class="tr-pipe-icon-lg">⚙</div>
            <div class="tr-pipe-label">Voxelwise<br/>ridge</div>
          </div>
          <span class="tr-pipe-arrow tr-pipe-arrow-h">→</span>
          <div class="tr-pipe-fuse">
            <div class="tr-pipe-icon-lg">𓆑</div>
            <div class="tr-pipe-label">Subject<br/>block</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def segmented(
    label: str,
    options: list[str],
    *,
    default_index: int = 0,
    key: str | None = None,
) -> str:
    """Pill-style segmented control. Returns the selected value.

    Built on top of ``st.radio`` with custom CSS that hides the radio
    dots and styles each option as a pill. The active option gets a
    subtle background and white text; inactive options stay muted.
    """
    return st.radio(
        label,
        options,
        index=default_index,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )


def demo_card_open(title: str = "Live encoder · sample stimulus") -> None:
    """Open the rounded card that wraps the demo column. Pair with
    :func:`demo_card_close`."""
    st.markdown(
        f"""
        <div class="tr-demo-card">
          <div class="tr-demo-header">
            <span class="tr-demo-title">{title}</span>
            <span class="tr-demo-cbar">
              <span class="tr-demo-cbar-label">Low</span>
              <span class="tr-demo-cbar-grad"></span>
              <span class="tr-demo-cbar-label">High</span>
              <div class="tr-demo-cbar-caption">Activity</div>
            </span>
          </div>
        """,
        unsafe_allow_html=True,
    )


def demo_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def tr_footer(
    title: str = "CortexLab",
    tagline: str = "Multimodal fMRI brain encoding · open source",
) -> None:
    """Render a polished four-column footer.

    Mirrors Meta-style research-page footers: a brand block on the
    left, three columns of grouped links, and a thin bottom rule with
    legal / build credits. All color-mode aware via the theme variables.
    """
    columns = [
        (
            "Product",
            [
                ("Live demo",   "https://huggingface.co/spaces/SID2000/cortexlab-dashboard"),
                ("Toolkit",     "https://github.com/siddhant-rajhans/cortexlab"),
                ("Dashboard",   "https://github.com/siddhant-rajhans/cortexlab-dashboard"),
                ("Releases",    "https://github.com/siddhant-rajhans/cortexlab/releases"),
            ],
        ),
        (
            "Research",
            [
                ("Paper",        "https://github.com/siddhant-rajhans/cortexlab"),
                ("BOLD Moments", "https://www.nature.com/articles/s41467-024-50310-3"),
                ("HCP-MMP 1.0",  "https://www.nature.com/articles/nature18933"),
                ("CLIP",         "https://arxiv.org/abs/2103.00020"),
            ],
        ),
        (
            "Resources",
            [
                ("HuggingFace",  "https://huggingface.co/SID2000/cortexlab"),
                ("API docs",     "https://github.com/siddhant-rajhans/cortexlab#readme"),
                ("Changelog",    "https://github.com/siddhant-rajhans/cortexlab/commits/master"),
                ("Issues",       "https://github.com/siddhant-rajhans/cortexlab/issues"),
            ],
        ),
    ]
    cols_html = ""
    for label, links in columns:
        items = "".join(
            f'<li><a class="tr-foot-link" href="{href}" target="_blank">{name}</a></li>'
            for name, href in links
        )
        cols_html += f"""
        <div class="tr-foot-col">
          <div class="tr-foot-col-title">{label}</div>
          <ul class="tr-foot-list">{items}</ul>
        </div>
        """

    st.markdown(
        f"""
        <div class="tr-foot">
          <div class="tr-foot-grid">
            <div class="tr-foot-brand">
              <div class="tr-foot-brand-logo">
                <span class="tr-foot-dot tr-foot-dot-a"></span>
                <span class="tr-foot-dot tr-foot-dot-b"></span>
                <span class="tr-foot-dot tr-foot-dot-c"></span>
              </div>
              <div class="tr-foot-brand-title">{title}</div>
              <div class="tr-foot-brand-tagline">{tagline}</div>
              <div class="tr-foot-built">
                Built on <a href="https://github.com/facebookresearch/tribev2" target="_blank">Meta's TRIBE v2</a>
              </div>
            </div>
            {cols_html}
          </div>
          <div class="tr-foot-rule"></div>
          <div class="tr-foot-legal">
            <span>© CortexLab Contributors · CC BY-NC 4.0</span>
            <span class="tr-foot-legal-right">
              <a href="https://github.com/siddhant-rajhans/cortexlab/blob/master/LICENSE" target="_blank">License</a>
              <span>·</span>
              <a href="https://github.com/siddhant-rajhans/cortexlab/blob/master/CODE_OF_CONDUCT.md" target="_blank">Conduct</a>
              <span>·</span>
              <a href="https://github.com/siddhant-rajhans/cortexlab" target="_blank">Source</a>
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = [
    "top_header",
    "architecture_steps",
    "pipeline_diagram",
    "segmented",
    "demo_card_open",
    "demo_card_close",
    "tr_footer",
]
