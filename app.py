"""Eval Dashboard - AI Agent Safety Evaluation Analysis Tool."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import get_settings
from utils.data_processing import (
    load_and_clean_csv,
    calculate_stats,
    get_display_stats,
    calculate_comparison,
    get_attack_distribution,
    get_failure_distribution,
    filter_dataframe,
    calculate_stats_with_mapping,
    format_stats_for_display,
)
from utils.mappings import (
    load_mappings,
    save_mappings,
    find_similar_categories,
    apply_mappings,
)
from utils.insights import (
    get_category_insights,
    get_comparison_insights,
    analyze_failure_patterns,
    analyze_specific_failures,
    generate_recommendations,
    generate_executive_summary,
)
from utils.taxonomy import (
    TAXONOMY_LEVELS,
    get_hierarchy_path,
    get_children,
    get_child_column,
    get_parent_column,
    calculate_category_stats,
    get_worst_category,
    get_best_category,
    get_available_taxonomy_columns,
)
from utils.column_mapper import (
    COLUMN_ROLES,
    auto_detect_columns,
    validate_mapping,
    apply_mapping_to_df,
    get_hierarchy_columns,
    save_column_mapping,
    load_column_mapping,
    list_saved_mappings,
    delete_column_mapping,
)
from utils.heatmap import (
    create_heatmap_data,
    create_pass_rate_heatmap_data,
    create_heatmap_figure,
    create_pass_rate_heatmap_figure,
    create_comparison_heatmap,
    create_pass_rate_comparison,
    get_distribution_summary,
    get_coverage_gaps,
)
from utils.chart_theme import (
    COLORS,
    SEVERITY_COLORS,
    style_plotly_chart,
    get_risk_color,
)
import pandas as pd
from pathlib import Path

# Load settings
settings = get_settings()

# Page configuration
st.set_page_config(
    page_title="AIUC-1 Eval Dashboard",
    page_icon="https://www.aiuc-1.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css():
    """Load custom CSS styling from assets/custom.css or inline fallback."""
    css_file = Path(__file__).parent / "assets" / "custom.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Inline fallback CSS for AIUC-1 theme
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp { font-family: 'Inter', sans-serif; }

        h1, h2, h3, h4, h5, h6 {
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetric"] {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 1rem;
        }

        div[data-testid="stMetric"] label {
            color: #888 !important;
            font-size: 0.875rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border: none;
            border-radius: 8px;
            font-weight: 500;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #1a1a1a;
            border-radius: 12px;
            padding: 4px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #6366f1 !important;
            color: white !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            overflow: hidden;
        }

        section[data-testid="stSidebar"] {
            background-color: #0a0a0a;
            border-right: 1px solid #1a1a1a;
        }

        div[data-testid="stFileUploader"] {
            background-color: #1a1a1a;
            border: 2px dashed #333;
            border-radius: 12px;
        }

        div[data-testid="stFileUploader"]:hover {
            border-color: #6366f1;
        }
        </style>
        """, unsafe_allow_html=True)


# Load custom styling
load_custom_css()


def df_to_clipboard_format(df) -> str:
    """Convert dataframe to tab-separated string for clipboard.

    Args:
        df: Dataframe to convert

    Returns:
        Tab-separated string representation
    """
    return df.to_csv(sep='\t', index=False)


# ============================================================================
# GLOBAL FILTERING SYSTEM
# ============================================================================

def get_cascading_filter_options(
    df: pd.DataFrame,
    mapping: dict,
    current_filters: dict,
    hierarchy_type: str = 'risk'
) -> dict:
    """Get filter options that respect hierarchy - selecting L1 limits L2 options, etc.

    Args:
        df: Full dataframe (before any filters applied)
        mapping: Column mapping dict
        current_filters: Currently applied filters
        hierarchy_type: 'risk' or 'attack'

    Returns:
        Dict mapping level columns to their available options
    """
    # Get hierarchy columns
    if hierarchy_type == 'risk':
        hierarchy_cols = mapping.get('risk_hierarchy', [])
        level_names = [f'Risk L{i+1}' for i in range(len(hierarchy_cols))]
    else:
        hierarchy_cols = mapping.get('attack_hierarchy', [])
        level_names = [f'Attack L{i+1}' for i in range(len(hierarchy_cols))]

    # Only use columns that exist in df
    level_names = [c for c in level_names if c in df.columns]

    if not level_names:
        return {}

    options = {}
    filtered_df = df.copy()

    for i, col in enumerate(level_names):
        # Apply filters from higher levels
        for j in range(i):
            higher_col = level_names[j]
            if higher_col in current_filters and current_filters[higher_col]:
                filtered_df = filtered_df[filtered_df[higher_col].isin(current_filters[higher_col])]

        # Get unique options at this level
        options[col] = sorted(filtered_df[col].dropna().unique().tolist())

    return options


def get_filter_state_key(key_prefix: str = "") -> str:
    """Get the session state key for filters.

    Args:
        key_prefix: Optional prefix for namespacing

    Returns:
        Session state key string
    """
    return f'{key_prefix}global_filters'


def initialize_filters(key_prefix: str = "") -> dict:
    """Initialize or get current filter state.

    Args:
        key_prefix: Optional prefix for namespacing

    Returns:
        Current filters dict
    """
    state_key = get_filter_state_key(key_prefix)
    if state_key not in st.session_state:
        st.session_state[state_key] = {}
    return st.session_state[state_key]


def apply_global_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filter dict to dataframe.

    Args:
        df: Input dataframe
        filters: Dict mapping column names to lists of allowed values

    Returns:
        Filtered dataframe
    """
    filtered = df.copy()

    for column, values in filters.items():
        if column in filtered.columns and values:
            filtered = filtered[filtered[column].isin(values)]

    return filtered


def render_active_filter_chips(filters: dict, key_prefix: str = "") -> bool:
    """Display active filters as removable chips.

    Args:
        filters: Current filter dict
        key_prefix: Prefix for widget keys

    Returns:
        True if any filter was removed (needs rerun)
    """
    if not filters or all(not v for v in filters.values()):
        return False

    st.markdown("**Active Filters:**")

    # Count total active filters
    active_count = sum(len(v) for v in filters.values() if v)
    removed = False

    # Create columns for filter chips
    cols = st.columns([4, 1])

    with cols[0]:
        chips_html = []
        for col, values in filters.items():
            if values:
                for val in values:
                    chips_html.append(f'<span style="background-color: #e0e7ff; color: #3730a3; padding: 4px 12px; border-radius: 16px; margin: 2px; display: inline-block; font-size: 13px;">{col}: {val}</span>')

        if chips_html:
            st.markdown(' '.join(chips_html), unsafe_allow_html=True)

    with cols[1]:
        if st.button("🗑️ Clear All", key=f"{key_prefix}clear_all_filters", type="secondary"):
            state_key = get_filter_state_key(key_prefix)
            st.session_state[state_key] = {}
            removed = True

    # Individual filter removal buttons
    if active_count > 1:
        with st.expander("Remove individual filters", expanded=False):
            for col, values in filters.items():
                if values:
                    for val in values:
                        if st.button(f"❌ {col}: {val}", key=f"{key_prefix}remove_{col}_{val}"):
                            state_key = get_filter_state_key(key_prefix)
                            st.session_state[state_key][col] = [v for v in values if v != val]
                            removed = True

    return removed


def render_filter_presets(filters: dict, key_prefix: str = "") -> dict:
    """Render filter preset save/load UI.

    Args:
        filters: Current filter dict
        key_prefix: Prefix for widget keys

    Returns:
        Filters dict (possibly loaded from preset)
    """
    import json
    import os

    presets_dir = os.path.join(os.path.dirname(__file__), 'data')
    presets_file = os.path.join(presets_dir, 'filter_presets.json')

    # Load existing presets
    def load_presets():
        if os.path.exists(presets_file):
            try:
                with open(presets_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_presets(presets):
        os.makedirs(presets_dir, exist_ok=True)
        with open(presets_file, 'w') as f:
            json.dump(presets, f, indent=2)

    presets = load_presets()

    col1, col2 = st.columns(2)

    with col1:
        # Load preset
        if presets:
            preset_names = [''] + list(presets.keys())
            selected_preset = st.selectbox(
                "Load preset",
                options=preset_names,
                format_func=lambda x: '-- Select --' if x == '' else x,
                key=f"{key_prefix}load_preset_select",
            )
            if selected_preset and st.button("Load", key=f"{key_prefix}load_preset_btn"):
                state_key = get_filter_state_key(key_prefix)
                st.session_state[state_key] = presets[selected_preset].copy()
                st.success(f"Loaded preset: {selected_preset}")
                return presets[selected_preset].copy()
        else:
            st.caption("No saved presets")

    with col2:
        # Save preset
        if filters and any(v for v in filters.values()):
            preset_name = st.text_input(
                "Save current as preset",
                placeholder="e.g., Critical Safety Issues",
                key=f"{key_prefix}save_preset_name",
            )
            if preset_name and st.button("Save", key=f"{key_prefix}save_preset_btn"):
                presets[preset_name] = filters.copy()
                save_presets(presets)
                st.success(f"Saved preset: {preset_name}")
        else:
            st.caption("Apply filters to save a preset")

    # Delete preset option
    if presets:
        with st.expander("Manage presets"):
            for name in list(presets.keys()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Show preview of preset filters
                    preview = ", ".join([f"{k}: {len(v)}" for k, v in presets[name].items() if v])
                    st.caption(f"**{name}**: {preview}")
                with col2:
                    if st.button("🗑️", key=f"{key_prefix}delete_preset_{name}"):
                        del presets[name]
                        save_presets(presets)
                        st.rerun()

    return filters


def render_global_filters(
    df: pd.DataFrame,
    mapping: dict,
    key_prefix: str = "",
    show_presets: bool = True
) -> pd.DataFrame:
    """Render filter controls that apply across all views.

    Args:
        df: Full dataframe before filtering
        mapping: Column mapping dict
        key_prefix: Prefix for session state keys
        show_presets: Whether to show preset save/load UI

    Returns:
        Filtered dataframe
    """
    state_key = get_filter_state_key(key_prefix)
    filters = initialize_filters(key_prefix)

    with st.expander("🔍 Filters", expanded=bool(filters and any(v for v in filters.values()))):
        # Active filter chips at top
        if filters and any(v for v in filters.values()):
            if render_active_filter_chips(filters, key_prefix):
                st.rerun()
            st.divider()

        # Get cascading options for hierarchies
        risk_options = get_cascading_filter_options(df, mapping, filters, 'risk')
        attack_options = get_cascading_filter_options(df, mapping, filters, 'attack')

        # Build filter UI in columns
        num_filter_cols = 3
        filter_cols = st.columns(num_filter_cols)

        col_idx = 0

        # Risk hierarchy filters (cascading)
        for col_name, options in risk_options.items():
            if not options:
                continue

            with filter_cols[col_idx % num_filter_cols]:
                current_selection = filters.get(col_name, [])
                # Ensure current selection values still exist in options
                valid_selection = [v for v in current_selection if v in options]

                selected = st.multiselect(
                    col_name,
                    options=options,
                    default=valid_selection,
                    key=f"{key_prefix}filter_{col_name}",
                    help=f"Filter by {col_name}",
                )
                filters[col_name] = selected
            col_idx += 1

        # Attack hierarchy filters (cascading)
        for col_name, options in attack_options.items():
            if not options:
                continue

            with filter_cols[col_idx % num_filter_cols]:
                current_selection = filters.get(col_name, [])
                valid_selection = [v for v in current_selection if v in options]

                selected = st.multiselect(
                    col_name,
                    options=options,
                    default=valid_selection,
                    key=f"{key_prefix}filter_{col_name}",
                    help=f"Filter by {col_name}",
                )
                filters[col_name] = selected
            col_idx += 1

        # Severity filter (always available if column exists)
        if 'Severity' in df.columns:
            with filter_cols[col_idx % num_filter_cols]:
                severity_options = sorted(df['Severity'].dropna().unique().tolist())
                current_severity = filters.get('Severity', [])
                valid_severity = [v for v in current_severity if v in severity_options]

                selected_severity = st.multiselect(
                    "Severity",
                    options=severity_options,
                    default=valid_severity,
                    key=f"{key_prefix}filter_Severity",
                    help="Filter by severity level",
                )
                filters['Severity'] = selected_severity
            col_idx += 1

        # Eval round filter (if column exists)
        if 'Eval round' in df.columns:
            with filter_cols[col_idx % num_filter_cols]:
                round_options = sorted(df['Eval round'].dropna().unique().tolist())
                current_round = filters.get('Eval round', [])
                valid_round = [v for v in current_round if v in round_options]

                selected_round = st.multiselect(
                    "Eval Round",
                    options=round_options,
                    default=valid_round,
                    key=f"{key_prefix}filter_Eval_round",
                    help="Filter by evaluation round",
                )
                filters['Eval round'] = selected_round

        # Save filters to session state
        st.session_state[state_key] = filters

        # Presets section
        if show_presets:
            st.divider()
            filters = render_filter_presets(filters, key_prefix)
            st.session_state[state_key] = filters

    # Apply filters and return
    filtered_df = apply_global_filters(df, filters)

    # Show filter summary
    active_filters = {k: v for k, v in filters.items() if v}
    if active_filters:
        filter_summary = ", ".join([f"{k}: {len(v)}" for k, v in active_filters.items()])
        st.caption(f"🔍 Filters applied: {filter_summary} | Showing {len(filtered_df):,} of {len(df):,} rows")
    else:
        st.caption(f"Showing all {len(df):,} rows")

    return filtered_df


def render_stats_table(df) -> None:
    """Render pass/fail statistics table.

    Args:
        df: Filtered dataframe
    """
    st.subheader("Pass/Fail Statistics")

    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Build available grouping options - prefer Risk L2 as default
        group_options = []
        for col in ['Risk L2', 'Risk L1', 'Risk L3', 'Attack L1', 'Attack L2', 'Type']:
            if col in df.columns:
                group_options.append(col)

        if not group_options:
            st.warning("No groupable columns found in data.")
            return

        # Default to Risk L2 if available, otherwise first option
        default_idx = 0

        group_by = st.selectbox(
            "Group by",
            options=group_options,
            index=default_idx,
            help="Select column to group statistics by",
        )

    with col2:
        # Check if Eval round column exists
        has_round_column = 'Eval round' in df.columns
        show_rounds = st.checkbox(
            "Show rounds separately",
            value=has_round_column,
            disabled=not has_round_column,
            help="Show separate rows for each evaluation round" if has_round_column else "No 'Eval round' column in data",
        )

    with col3:
        show_counts = st.checkbox(
            "Show counts in cells",
            value=False,
            help="Display counts alongside percentages",
        )

    # Calculate and display stats
    try:
        stats = calculate_stats(
            df,
            group_by,
            include_round=show_rounds and has_round_column
        )
        display_stats = get_display_stats(stats, show_counts=show_counts)
    except ValueError as e:
        st.error(str(e))
        return

    st.dataframe(display_stats, use_container_width=True, hide_index=True)

    # Export section
    st.divider()

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        csv_data = display_stats.to_csv(index=False)
        st.download_button(
            label="📋 Download as CSV",
            data=csv_data,
            file_name="eval_stats.csv",
            mime="text/csv",
        )

    with export_col2:
        if st.button("📋 Copy to Clipboard"):
            clipboard_data = df_to_clipboard_format(display_stats)
            st.code(clipboard_data, language=None)
            st.caption("Copy the above text (Ctrl+C / Cmd+C)")

    # Summary metrics
    st.divider()

    total = len(df)
    passes = len(df[df['Severity'] == 'PASS']) if 'Severity' in df.columns else 0
    failures = total - passes
    pass_rate = (passes / total * 100) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Evals", f"{total:,}")
    with col2:
        st.metric("Passed", f"{passes:,}")
    with col3:
        st.metric("Failed", f"{failures:,}")
    with col4:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")


def render_attack_distribution(df) -> None:
    """Render attack distribution analysis.

    Args:
        df: Filtered dataframe
    """
    st.subheader("Attack Distribution")

    # Build available level options
    level_options = []
    for level in ['Attack L1', 'Attack L2', 'Attack L3']:
        if level in df.columns:
            level_options.append(level)

    if not level_options:
        # Show helpful message with available columns
        all_cols = list(df.columns)
        attack_related = [c for c in all_cols if 'attack' in c.lower()]
        st.warning("No standard Attack columns (Attack L1, L2, L3) found in data.")
        st.markdown("**Expected column names:** `Attack L1`, `Attack L2`, `Attack L3`")
        st.markdown("**Accepted variations:** `attack_l1`, `Attack_L1`, `attack l1`, etc.")
        if attack_related:
            st.info(f"Attack-related columns found (not matching expected pattern): {attack_related}")
        st.markdown("**Available columns in your data:**")
        st.code(", ".join(all_cols))
        return

    selected_level = st.selectbox(
        "Select level",
        options=level_options,
        index=0,
        help="Choose attack hierarchy level to display",
    )

    # Get distribution with error handling
    try:
        dist = get_attack_distribution(df, selected_level)
    except ValueError as e:
        st.error(str(e))
        return

    # Two columns: table and chart
    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(dist, use_container_width=True, hide_index=True)

        # Copy button
        if st.button("📋 Copy to Clipboard", key="copy_attack_dist"):
            clipboard_data = df_to_clipboard_format(dist)
            st.code(clipboard_data, language=None)
            st.caption("Copy the above text (Ctrl+C / Cmd+C)")

    with col2:
        # Pie chart with AIUC-1 theme
        fig = px.pie(
            dist,
            values='Count',
            names=selected_level,
            title=f"{selected_level} Distribution",
            hole=0.3,
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['safety'],
                                     COLORS['security'], COLORS['reliability'], COLORS['data_privacy']]
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(t=50, b=50, l=20, r=20),
        )
        style_plotly_chart(fig)
        st.plotly_chart(fig, use_container_width=True)


def render_failure_distribution(df) -> None:
    """Render failure severity distribution.

    Args:
        df: Filtered dataframe
    """
    st.subheader("Failure Severity Breakdown")

    if 'Severity' not in df.columns:
        st.warning("'Severity' column not found in data.")
        return

    dist = get_failure_distribution(df)

    if len(dist) == 0:
        st.success("No failures found in the filtered data!")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(dist, use_container_width=True, hide_index=True)

        # Copy button
        if st.button("Copy to Clipboard", key="copy_failure_dist"):
            clipboard_data = df_to_clipboard_format(dist)
            st.code(clipboard_data, language=None)
            st.caption("Copy the above text (Ctrl+C / Cmd+C)")

    with col2:
        # Bar chart with AIUC-1 severity colors
        colors = [SEVERITY_COLORS.get(sev, COLORS['text_muted']) for sev in dist['Severity']]

        fig = go.Figure(data=[
            go.Bar(
                x=dist['Severity'],
                y=dist['Count'],
                marker_color=colors,
                text=dist['Count'],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Failure Distribution by Severity",
            xaxis_title="Severity",
            yaxis_title="Count",
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=20),
        )
        style_plotly_chart(fig)

        st.plotly_chart(fig, use_container_width=True)


def render_insights(df, api_key: str) -> None:
    """Render AI-powered insights.

    Args:
        df: Filtered dataframe
        api_key: Anthropic API key
    """
    st.subheader("AI-Powered Insights")

    if not api_key:
        st.warning("Enter your Claude API key in the sidebar to enable AI insights.")
        return

    # Build available category type options
    category_type_options = []
    for col in ['Risk L1', 'Risk L2', 'Attack L1']:
        if col in df.columns:
            category_type_options.append(col)

    if not category_type_options:
        st.warning("No category columns found in data.")
        return

    # Two selectboxes for category type and value
    col1, col2 = st.columns(2)

    with col1:
        category_type = st.selectbox(
            "Category type",
            options=category_type_options,
            index=0,
            help="Choose the category type to analyze",
        )

    with col2:
        category_values = df[category_type].unique().tolist()
        selected_category = st.selectbox(
            "Category value",
            options=category_values,
            help="Choose a specific category to get detailed failure analysis",
        )

    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analyzing failures with Claude..."):
            insights = get_category_insights(
                df,
                category_column=category_type,
                category_value=selected_category,
                api_key=api_key,
            )
            st.markdown(insights)


def render_summary_cards(df, mapping: dict) -> None:
    """Render summary metric cards at the top of the page.

    Args:
        df: Dataframe with eval results (already mapped)
        mapping: Column mapping dict with risk_hierarchy and attack_hierarchy
    """
    if 'Severity' not in df.columns:
        return

    # Get available hierarchy levels from mapping
    risk_levels = []
    for i, col in enumerate(mapping.get('risk_hierarchy', [])):
        standardized = f'Risk L{i + 1}'
        if standardized in df.columns:
            risk_levels.append(standardized)

    attack_levels = []
    for i, col in enumerate(mapping.get('attack_hierarchy', [])):
        standardized = f'Attack L{i + 1}'
        if standardized in df.columns:
            attack_levels.append(standardized)

    # Settings expander for summary card configuration
    with st.expander("⚙️ Summary Card Settings", expanded=False):
        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            if risk_levels:
                # Default to L2 if available, otherwise first level
                default_risk_idx = min(1, len(risk_levels) - 1)
                saved_risk_level = st.session_state.get('summary_risk_level', risk_levels[default_risk_idx])
                if saved_risk_level not in risk_levels:
                    saved_risk_level = risk_levels[default_risk_idx]

                risk_summary_level = st.selectbox(
                    "Risk summary level",
                    options=risk_levels,
                    index=risk_levels.index(saved_risk_level),
                    key="risk_summary_level_select",
                    help="Which risk level to show in 'Most Vulnerable' card",
                )
                st.session_state['summary_risk_level'] = risk_summary_level
            else:
                risk_summary_level = None

        with settings_col2:
            if attack_levels:
                # Default to L1 for attacks
                default_attack_idx = 0
                saved_attack_level = st.session_state.get('summary_attack_level', attack_levels[default_attack_idx])
                if saved_attack_level not in attack_levels:
                    saved_attack_level = attack_levels[default_attack_idx]

                attack_summary_level = st.selectbox(
                    "Attack summary level",
                    options=attack_levels,
                    index=attack_levels.index(saved_attack_level),
                    key="attack_summary_level_select",
                    help="Which attack level to show in 'Most Effective Attack' card",
                )
                st.session_state['summary_attack_level'] = attack_summary_level
            else:
                attack_summary_level = None

        # Exclusions section
        st.markdown("**Exclusions**")
        excl_col1, excl_col2 = st.columns(2)

        with excl_col1:
            if risk_summary_level and risk_summary_level in df.columns:
                risk_values = df[risk_summary_level].dropna().unique().tolist()
                excluded_risks = st.multiselect(
                    "Exclude from 'Most Vulnerable Risk'",
                    options=sorted(risk_values),
                    default=st.session_state.get('excluded_risks', []),
                    key="excluded_risks_select",
                    help="These values won't be considered for the Most Vulnerable card",
                )
                st.session_state['excluded_risks'] = excluded_risks
            else:
                excluded_risks = []

        with excl_col2:
            if attack_summary_level and attack_summary_level in df.columns:
                attack_values = df[attack_summary_level].dropna().unique().tolist()
                # Default to excluding 'Benign' if it exists
                default_excluded = ['Benign'] if 'Benign' in attack_values else []
                saved_excluded = st.session_state.get('excluded_attacks', default_excluded)
                # Filter to only valid values
                saved_excluded = [v for v in saved_excluded if v in attack_values]

                excluded_attacks = st.multiselect(
                    "Exclude from 'Most Effective Attack'",
                    options=sorted(attack_values),
                    default=saved_excluded,
                    key="excluded_attacks_select",
                    help="These values won't be considered for the Most Effective Attack card",
                )
                st.session_state['excluded_attacks'] = excluded_attacks
            else:
                excluded_attacks = []

    # Calculate basic metrics
    total = len(df)
    passes = len(df[df['Severity'] == 'PASS'])
    pass_rate = (passes / total * 100) if total > 0 else 0

    # Critical failures (P0 + P1)
    critical = len(df[df['Severity'].isin(['P0', 'P1'])])

    # Most vulnerable risk area (at selected level, excluding specified values)
    worst_risk = None
    if risk_summary_level and risk_summary_level in df.columns:
        risk_df = df.copy()
        if excluded_risks:
            risk_df = risk_df[~risk_df[risk_summary_level].isin(excluded_risks)]
        if len(risk_df) > 0:
            worst_risk = get_worst_category(risk_df, risk_summary_level, min_count=5)

    # Most effective attack type (at selected level, excluding specified values)
    worst_attack = None
    if attack_summary_level and attack_summary_level in df.columns:
        attack_df = df.copy()
        if excluded_attacks:
            attack_df = attack_df[~attack_df[attack_summary_level].isin(excluded_attacks)]
        if len(attack_df) > 0:
            worst_attack = get_worst_category(attack_df, attack_summary_level, min_count=5)

    # Four metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Color based on pass rate
        if pass_rate >= 90:
            delta_color = "normal"
        elif pass_rate >= 80:
            delta_color = "off"
        else:
            delta_color = "inverse"
        st.metric(
            "Overall Pass Rate",
            f"{pass_rate:.1f}%",
            delta=f"{passes:,} / {total:,}",
            delta_color=delta_color,
        )

    with col2:
        if worst_risk:
            level_label = risk_summary_level.replace('Risk ', '') if risk_summary_level else ''
            st.metric(
                f"Most Vulnerable ({level_label})",
                worst_risk['category'][:20] + ('...' if len(worst_risk['category']) > 20 else ''),
                delta=f"{worst_risk['pass_rate']}% pass rate",
                delta_color="inverse",
            )
        else:
            st.metric("Most Vulnerable Risk", "N/A", delta="No data or all excluded")

    with col3:
        if worst_attack:
            level_label = attack_summary_level.replace('Attack ', '') if attack_summary_level else ''
            st.metric(
                f"Most Effective Attack ({level_label})",
                worst_attack['category'][:20] + ('...' if len(worst_attack['category']) > 20 else ''),
                delta=f"{worst_attack['pass_rate']}% pass rate",
                delta_color="inverse",
            )
        else:
            st.metric("Most Effective Attack", "N/A", delta="No data or all excluded")

    with col4:
        st.metric(
            "Critical Failures",
            f"{critical:,}",
            delta="P0 + P1",
            delta_color="inverse" if critical > 0 else "off",
        )


def render_executive_summary(df, api_key: str) -> None:
    """Render executive summary button and output.

    Args:
        df: Dataframe with eval results
        api_key: Anthropic API key
    """
    if not api_key:
        st.info("Enter your Claude API key in the sidebar to generate an executive summary.")
        return

    if st.button("📋 Generate Executive Summary", type="primary", key="exec_summary"):
        with st.spinner("Generating executive summary..."):
            summary = generate_executive_summary(df, api_key)

            # Display in a nice box
            st.markdown("---")
            st.markdown("### Executive Summary")
            st.markdown(f"> {summary}")

            # Copy button
            st.text_area(
                "Copy-pasteable version:",
                value=summary,
                height=150,
                key="exec_summary_text",
            )


def render_column_mapper(df: pd.DataFrame, key_prefix: str = "") -> tuple:
    """Show column mapping interface and return the mapping.

    Args:
        df: Raw dataframe before mapping
        key_prefix: Prefix for session state keys (allows multiple mappers on same page)

    Returns:
        Tuple of (mapping dict, is_valid bool)
    """
    st.subheader("Configure Data Columns")
    st.markdown("Map your CSV columns to the analysis fields. We've auto-detected some - adjust as needed.")

    # Get auto-detected suggestions
    suggestions = auto_detect_columns(df)

    # Session state key for this mapper
    state_key = f'{key_prefix}column_mapping'

    # Initialize or get from session state
    if state_key not in st.session_state:
        st.session_state[state_key] = suggestions.copy()

    mapping = st.session_state[state_key]
    all_columns = ['(none)'] + df.columns.tolist()

    # Load saved mappings section
    saved_mappings = list_saved_mappings()
    if saved_mappings:
        with st.expander("Load saved mapping"):
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_saved = st.selectbox(
                    "Select mapping",
                    options=[''] + saved_mappings,
                    format_func=lambda x: '-- Select --' if x == '' else x,
                    key=f"{key_prefix}load_mapping_select",
                )
            with col2:
                if selected_saved and st.button("Load", key=f"{key_prefix}load_mapping_btn"):
                    loaded = load_column_mapping(selected_saved)
                    if loaded:
                        st.session_state[state_key] = loaded
                        st.success(f"Loaded mapping: {selected_saved}")
                        st.rerun()

    # Create mapping UI
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Required Fields**")

        # Severity (required)
        current_severity = mapping.get('severity', suggestions.get('severity', '(none)'))
        if current_severity not in all_columns:
            current_severity = '(none)'

        mapping['severity'] = st.selectbox(
            "Severity / Grade column *",
            options=all_columns,
            index=all_columns.index(current_severity) if current_severity in all_columns else 0,
            help="Column with values like PASS, P4, P3, P2, P1, P0",
            key=f"{key_prefix}map_severity",
        )

        # Risk hierarchy (required, multi-select)
        st.markdown("**Risk Hierarchy** *(select in order: broadest → most specific)*")
        current_risk = mapping.get('risk_hierarchy', suggestions.get('risk_hierarchy', []))
        valid_risk = [c for c in current_risk if c in df.columns]

        mapping['risk_hierarchy'] = st.multiselect(
            "Risk columns *",
            options=df.columns.tolist(),
            default=valid_risk,
            help="Select 1-3 columns representing risk levels (L1 → L2 → L3)",
            key=f"{key_prefix}map_risk",
        )

    with col2:
        st.markdown("**Optional Fields**")

        # Attack hierarchy (optional, multi-select)
        st.markdown("**Attack Hierarchy** *(select in order: broadest → most specific)*")
        current_attack = mapping.get('attack_hierarchy', suggestions.get('attack_hierarchy', []))
        valid_attack = [c for c in current_attack if c in df.columns]

        mapping['attack_hierarchy'] = st.multiselect(
            "Attack columns",
            options=df.columns.tolist(),
            default=valid_attack,
            help="Select 1-3 columns representing attack types",
            key=f"{key_prefix}map_attack",
        )

        # Round
        current_round = mapping.get('round', suggestions.get('round', '(none)'))
        if current_round not in all_columns:
            current_round = '(none)'

        mapping['round'] = st.selectbox(
            "Eval Round column",
            options=all_columns,
            index=all_columns.index(current_round) if current_round in all_columns else 0,
            help="Column indicating Round 1, Round 2, etc.",
            key=f"{key_prefix}map_round",
        )

        # Justification
        current_just = mapping.get('justification', suggestions.get('justification', '(none)'))
        if current_just not in all_columns:
            current_just = '(none)'

        mapping['justification'] = st.selectbox(
            "Justification column",
            options=all_columns,
            index=all_columns.index(current_just) if current_just in all_columns else 0,
            help="Text explaining failures (for AI analysis)",
            key=f"{key_prefix}map_justification",
        )

        # Transcript
        current_trans = mapping.get('transcript', suggestions.get('transcript', '(none)'))
        if current_trans not in all_columns:
            current_trans = '(none)'

        mapping['transcript'] = st.selectbox(
            "Transcript column",
            options=all_columns,
            index=all_columns.index(current_trans) if current_trans in all_columns else 0,
            help="Full conversation log (for deep AI analysis)",
            key=f"{key_prefix}map_transcript",
        )

    # Validation
    is_valid, errors = validate_mapping(mapping)

    if errors:
        for error in errors:
            st.error(error)
    else:
        st.success("Column mapping complete!")

    # Save mapping section
    if is_valid:
        with st.expander("Save this mapping for later"):
            save_col1, save_col2 = st.columns([3, 1])
            with save_col1:
                mapping_name = st.text_input(
                    "Mapping name",
                    placeholder="e.g., my_eval_format",
                    key=f"{key_prefix}save_mapping_name",
                )
            with save_col2:
                if mapping_name and st.button("Save", key=f"{key_prefix}save_mapping_btn"):
                    save_column_mapping(mapping, mapping_name)
                    st.success(f"Saved as: {mapping_name}")

    # Save mapping to session state
    st.session_state[state_key] = mapping

    # Show preview
    with st.expander("Preview mapped data"):
        preview_cols = []
        if mapping.get('severity') and mapping['severity'] != '(none)':
            preview_cols.append(mapping['severity'])
        preview_cols.extend(mapping.get('risk_hierarchy', []))
        preview_cols.extend(mapping.get('attack_hierarchy', []))
        if mapping.get('justification') and mapping['justification'] != '(none)':
            preview_cols.append(mapping['justification'])

        if preview_cols:
            # Only show columns that exist
            preview_cols = [c for c in preview_cols if c in df.columns]
            if preview_cols:
                st.dataframe(df[preview_cols].head(10), use_container_width=True)
        else:
            st.info("Select columns above to see preview")

    return mapping, is_valid


def render_deep_dive(df, api_key: str) -> None:
    """Render hierarchical drill-down navigation.

    Args:
        df: Dataframe with eval results
        api_key: Anthropic API key
    """
    st.subheader("Taxonomy Deep Dive")

    # Initialize session state for drill-down path
    if 'drill_down_path' not in st.session_state:
        st.session_state.drill_down_path = []
    if 'drill_down_taxonomy' not in st.session_state:
        st.session_state.drill_down_taxonomy = 'Risk'

    # Taxonomy selector
    available_taxonomies = get_available_taxonomy_columns(df)
    if not available_taxonomies:
        st.warning("No taxonomy columns found in data.")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        selected_taxonomy = st.radio(
            "Taxonomy",
            options=list(available_taxonomies.keys()),
            index=0,
            key="taxonomy_selector",
        )

        # Reset path if taxonomy changed
        if selected_taxonomy != st.session_state.drill_down_taxonomy:
            st.session_state.drill_down_path = []
            st.session_state.drill_down_taxonomy = selected_taxonomy

    # Get current level based on path
    levels = TAXONOMY_LEVELS.get(selected_taxonomy, [])
    available_levels = [l for l in levels if l in df.columns]

    if not available_levels:
        st.warning(f"No {selected_taxonomy} columns found in data.")
        return

    current_depth = len(st.session_state.drill_down_path)
    current_level = available_levels[min(current_depth, len(available_levels) - 1)]

    # Build breadcrumb
    with col2:
        breadcrumb_parts = ["All"]
        for i, (level, value) in enumerate(st.session_state.drill_down_path):
            breadcrumb_parts.append(value)

        breadcrumb = " > ".join(breadcrumb_parts)
        if current_depth < len(available_levels):
            breadcrumb += f" > [{current_level}]"

        st.markdown(f"**Navigation:** {breadcrumb}")

        # Back button
        if st.session_state.drill_down_path:
            if st.button("← Back", key="drill_back"):
                st.session_state.drill_down_path.pop()
                st.rerun()

    st.divider()

    # Filter data based on current path
    filtered_df = df.copy()
    for level, value in st.session_state.drill_down_path:
        if level in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[level] == value]

    if len(filtered_df) == 0:
        st.warning("No data matches the current selection.")
        return

    # Calculate stats for current level
    if current_depth < len(available_levels):
        current_column = available_levels[current_depth]

        # Get parent info for filtering
        parent_column = None
        parent_value = None
        if st.session_state.drill_down_path:
            parent_column, parent_value = st.session_state.drill_down_path[-1]

        try:
            stats = calculate_category_stats(
                filtered_df,
                current_column,
                parent_column,
                parent_value
            )
        except ValueError as e:
            st.error(str(e))
            return

        # Show stats table
        st.markdown(f"**{current_column} Categories** ({len(stats)} categories, {len(filtered_df):,} total evals)")

        # Format for display
        display_stats = stats.copy()
        display_stats['Pass Rate'] = display_stats['Pass Rate'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_stats, use_container_width=True, hide_index=True)

        # Get numeric pass rates for the bar chart
        numeric_rates = stats['Pass Rate'].tolist()

        # Bar chart with conditional AIUC-1 colors
        fig = go.Figure(data=[
            go.Bar(
                x=stats['Category'],
                y=numeric_rates,
                marker_color=[COLORS['pass'] if r >= 90 else COLORS['p4'] if r >= 80 else COLORS['safety'] for r in numeric_rates],
                text=[f"{r:.1f}%" for r in numeric_rates],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title=f"Pass Rate by {current_column}",
            xaxis_title=current_column,
            yaxis_title="Pass Rate (%)",
            showlegend=False,
            margin=dict(t=50, b=100, l=50, r=20),
            xaxis_tickangle=-45,
        )
        style_plotly_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Clickable category cards for drill-down
        can_drill_deeper = current_depth < len(available_levels) - 1

        if can_drill_deeper:
            st.markdown("**Click a category to drill down:**")

            # Create columns for category buttons
            categories = stats['Category'].tolist()
            n_cols = min(4, len(categories))
            cols = st.columns(n_cols)

            for i, cat in enumerate(categories):
                with cols[i % n_cols]:
                    cat_stats = stats[stats['Category'] == cat].iloc[0]
                    pass_rate = cat_stats['Pass Rate']
                    count = cat_stats['Count']

                    # Color based on pass rate
                    if pass_rate >= 90:
                        color = "🟢"
                    elif pass_rate >= 80:
                        color = "🟡"
                    else:
                        color = "🔴"

                    if st.button(
                        f"{color} {cat[:25]}{'...' if len(cat) > 25 else ''}\n{pass_rate:.1f}% ({count})",
                        key=f"drill_{current_column}_{cat}",
                        use_container_width=True,
                    ):
                        st.session_state.drill_down_path.append((current_column, cat))
                        st.rerun()

        # AI Analysis button
        st.divider()
        if api_key:
            if st.button("🤖 Analyze This Level", key="analyze_drill_level"):
                with st.spinner("Analyzing with Claude..."):
                    # Build context for analysis
                    context = f"Analyzing {current_column} level"
                    if st.session_state.drill_down_path:
                        path_str = " > ".join([v for _, v in st.session_state.drill_down_path])
                        context += f" within path: {path_str}"

                    insights = get_category_insights(
                        filtered_df,
                        category_column=current_column,
                        category_value=stats.iloc[0]['Category'] if len(stats) > 0 else "",
                        api_key=api_key,
                    )
                    st.markdown(insights)
    else:
        # At deepest level - show individual failures
        st.markdown("**Individual Evaluations at this level:**")
        st.dataframe(filtered_df.head(50), use_container_width=True, hide_index=True)


def render_failure_analysis(df, api_key: str) -> None:
    """Render failure analysis tab with filters and AI analysis.

    Args:
        df: Dataframe with eval results
        api_key: Anthropic API key
    """
    st.subheader("Failure Analysis")

    if 'Severity' not in df.columns:
        st.warning("'Severity' column not found in data.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        # Severity filter - default to worst failures
        severity_options = ['P0', 'P1', 'P2', 'P3', 'P4']
        available_severities = [s for s in severity_options if s in df['Severity'].unique()]
        default_severities = [s for s in ['P0', 'P1', 'P2'] if s in available_severities]

        selected_severities = st.multiselect(
            "Severity",
            options=available_severities,
            default=default_severities if default_severities else available_severities,
            help="Filter by severity level",
        )

    with col2:
        # Risk filter
        if 'Risk L1' in df.columns:
            risk_options = df['Risk L1'].dropna().unique().tolist()
            selected_risks = st.multiselect(
                "Risk L1",
                options=risk_options,
                default=risk_options,
                help="Filter by risk category",
            )
        else:
            selected_risks = None

    with col3:
        # Attack filter
        if 'Attack L1' in df.columns:
            attack_options = df['Attack L1'].dropna().unique().tolist()
            selected_attacks = st.multiselect(
                "Attack L1",
                options=attack_options,
                default=attack_options,
                help="Filter by attack type",
            )
        else:
            selected_attacks = None

    # Apply filters
    failures = df[df['Severity'].isin(selected_severities)] if selected_severities else df[df['Severity'] != 'PASS']

    if selected_risks and 'Risk L1' in df.columns:
        failures = failures[failures['Risk L1'].isin(selected_risks)]
    if selected_attacks and 'Attack L1' in df.columns:
        failures = failures[failures['Attack L1'].isin(selected_attacks)]

    if len(failures) == 0:
        st.success("No failures match the selected filters!")
        return

    st.caption(f"Showing {len(failures):,} failures")

    # Build display dataframe
    display_cols = []

    # Risk path
    risk_cols = [c for c in ['Risk L1', 'Risk L2', 'Risk L3'] if c in failures.columns]
    if risk_cols:
        failures = failures.copy()
        failures['Risk Path'] = failures.apply(
            lambda row: ' > '.join([str(row[c]) for c in risk_cols if pd.notna(row[c])]),
            axis=1
        )
        display_cols.append('Risk Path')

    # Attack path
    attack_cols = [c for c in ['Attack L1', 'Attack L2', 'Attack L3'] if c in failures.columns]
    if attack_cols:
        failures['Attack Path'] = failures.apply(
            lambda row: ' > '.join([str(row[c]) for c in attack_cols if pd.notna(row[c])]),
            axis=1
        )
        display_cols.append('Attack Path')

    display_cols.append('Severity')

    # Truncated justification
    if 'Justification' in failures.columns:
        failures['Justification (truncated)'] = failures['Justification'].apply(
            lambda x: str(x)[:100] + '...' if pd.notna(x) and len(str(x)) > 100 else str(x) if pd.notna(x) else ''
        )
        display_cols.append('Justification (truncated)')

    # Show failures table with selection
    st.markdown("**Select failures for analysis:**")

    # Store selected indices in session state
    if 'selected_failures' not in st.session_state:
        st.session_state.selected_failures = []

    # Display with checkboxes using data editor
    failures_display = failures[display_cols].copy()
    failures_display.insert(0, 'Select', False)
    failures_display = failures_display.reset_index()

    edited_df = st.data_editor(
        failures_display.head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select for AI analysis",
                default=False,
            ),
            "index": st.column_config.NumberColumn(
                "ID",
                help="Row index",
            ),
        },
        disabled=[c for c in failures_display.columns if c != 'Select'],
        key="failure_table",
    )

    # Get selected indices
    selected_mask = edited_df['Select'] == True
    selected_indices = edited_df.loc[selected_mask, 'index'].tolist()

    st.caption(f"Selected: {len(selected_indices)} failures")

    # Expandable details for selected failures
    if selected_indices:
        with st.expander("📋 View Full Details of Selected Failures", expanded=False):
            for idx in selected_indices[:10]:  # Limit to 10
                row = failures.loc[idx]
                st.markdown(f"**Failure #{idx}**")

                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    if 'Risk Path' in row:
                        st.markdown(f"- **Risk:** {row['Risk Path']}")
                    if 'Attack Path' in row:
                        st.markdown(f"- **Attack:** {row['Attack Path']}")
                    st.markdown(f"- **Severity:** {row['Severity']}")

                with detail_col2:
                    if 'Justification' in row and pd.notna(row['Justification']):
                        st.markdown(f"- **Justification:** {row['Justification']}")

                st.divider()

    # AI Analysis buttons
    st.divider()
    st.markdown("### AI Analysis")

    if not api_key:
        st.warning("Enter your Claude API key in the sidebar to enable AI analysis.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🤖 Why did these fail?", key="analyze_why", disabled=len(selected_indices) == 0):
            with st.spinner("Analyzing selected failures..."):
                result = analyze_specific_failures(failures, selected_indices, api_key)
                st.markdown(result)

    with col2:
        if st.button("🤖 Attack weaknesses?", key="analyze_attacks"):
            with st.spinner("Analyzing attack patterns..."):
                result = analyze_failure_patterns(failures, api_key)
                st.markdown(result)

    with col3:
        if st.button("🤖 What should be fixed?", key="analyze_fixes"):
            with st.spinner("Generating recommendations..."):
                result = generate_recommendations(failures, api_key)
                st.markdown(result)


def single_round_view(api_key: str) -> None:
    """Display single round analysis view.

    Args:
        api_key: Anthropic API key for insights
    """
    st.header("Single Round Analysis")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload evaluation CSV",
        type=['csv'],
        help="Upload a CSV file containing evaluation results.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin analysis.")

        with st.expander("Expected CSV Format"):
            st.markdown("""
            Your CSV can contain any column names - you'll map them in the next step.

            **Typical columns include:**
            - **Risk categories** (1-3 levels of hierarchy)
            - **Attack types** (1-3 levels of hierarchy)
            - **Severity/Grade** (e.g., PASS, P0, P1, P2, P3, P4)
            - **Justification** (text explaining the result)
            - **Eval Round** (optional, for comparing rounds)

            Column names will be auto-detected, but you can adjust the mapping.
            """)
        return

    # Load raw data (no automatic column renaming)
    # Use header=1 to read column names from row 2 (skip row 1)
    try:
        raw_df = pd.read_csv(uploaded_file, header=1)
        st.success(f"Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return

    # Step 1: Column Mapping (collapsible, but expanded if not yet configured)
    mapping_validated = st.session_state.get('mapping_validated', False)

    with st.expander("⚙️ Column Mapping", expanded=not mapping_validated):
        mapping, is_valid = render_column_mapper(raw_df)
        if is_valid:
            st.session_state.mapping_validated = True

    if not is_valid:
        st.warning("Please complete column mapping above to continue.")
        return

    # Apply mapping to create standardized dataframe
    df = apply_mapping_to_df(raw_df, mapping)

    # Store in session state
    st.session_state['single_df'] = df
    st.session_state['column_mapping'] = mapping

    # Global filters - cascading, with presets
    filtered_df = render_global_filters(df, mapping, key_prefix="single_")

    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
        return

    # Summary cards at top
    render_summary_cards(filtered_df, mapping)

    st.divider()

    # Executive Summary button
    render_executive_summary(filtered_df, api_key)

    st.divider()

    # Tabs - updated structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Deep Dive",
        "🔬 Failure Analysis",
        "📈 Attack Distribution",
        "🤖 AI Insights",
    ])

    with tab1:
        render_stats_table(filtered_df)
        st.divider()
        render_failure_distribution(filtered_df)

    with tab2:
        render_deep_dive(filtered_df, api_key)

    with tab3:
        render_failure_analysis(filtered_df, api_key)

    with tab4:
        render_attack_distribution(filtered_df)

    with tab5:
        render_insights(filtered_df, api_key)


def render_mapping_editor(df1, df2, group_by: str) -> dict:
    """Render category mapping editor.

    Args:
        df1: First dataframe (baseline)
        df2: Second dataframe (current)
        group_by: Column to map categories for

    Returns:
        Dictionary of category mappings
    """
    # Load existing mappings
    all_mappings = load_mappings()
    current_mappings = all_mappings.get(group_by, {})

    # Get unique categories from both dataframes
    cats1 = set(df1[group_by].unique()) if group_by in df1.columns else set()
    cats2 = set(df2[group_by].unique()) if group_by in df2.columns else set()

    # Find categories only in one dataset
    only_in_r1 = cats1 - cats2
    only_in_r2 = cats2 - cats1

    # If all match, show success and return
    if not only_in_r1 and not only_in_r2:
        st.success("All categories match between rounds!")
        return current_mappings

    # Show warning with mismatch counts
    st.warning(
        f"Found {len(only_in_r1)} categories only in R1, "
        f"{len(only_in_r2)} categories only in R2"
    )

    st.markdown("**Map R1 categories to R2:**")
    st.caption("Map categories from Round 1 to their equivalents in Round 2.")

    # Try to find similar categories using fuzzy matching
    suggestions = find_similar_categories(list(only_in_r1), list(cats2))

    # Create mapping inputs
    new_mappings = {}

    for cat in sorted(only_in_r1):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.text(cat)

        with col2:
            # Get suggestion if available
            suggestion = suggestions.get(cat, {})
            suggested_value = suggestion.get('suggested', '')
            confidence = suggestion.get('confidence', 0)

            # Determine default value
            # Priority: saved mapping > high-confidence suggestion > keep as-is
            if cat in current_mappings:
                default_value = current_mappings[cat]
            elif confidence > 70:
                default_value = suggested_value
            else:
                default_value = ''

            options = ['[Keep as-is]'] + sorted(list(cats2))
            default_idx = 0
            if default_value in options:
                default_idx = options.index(default_value)

            # Show confidence hint if suggestion exists
            help_text = None
            if suggested_value and confidence > 0:
                help_text = f"Suggested: '{suggested_value}' ({confidence}% match)"

            selected = st.selectbox(
                f"Map '{cat}' to",
                options=options,
                index=default_idx,
                key=f"map_{group_by}_{cat}",
                label_visibility="collapsed",
                help=help_text,
            )

            if selected != '[Keep as-is]':
                new_mappings[cat] = selected

    # Show categories only in R2 for reference
    if only_in_r2:
        st.divider()
        st.caption(f"**Categories only in R2** (not in R1): {', '.join(sorted(only_in_r2))}")

    # Save button
    if st.button("Save Mappings", type="primary", key="save_mappings"):
        all_mappings[group_by] = new_mappings
        save_mappings(all_mappings)
        st.success("Mappings saved!")
        st.rerun()

    return new_mappings if new_mappings else current_mappings


def format_comparison_display(comparison_df) -> tuple:
    """Format comparison dataframe for display.

    Args:
        comparison_df: Raw comparison dataframe

    Returns:
        Tuple of (display_df, raw_df for export)
    """
    display_df = comparison_df.copy()

    # Format Pass % columns
    if 'PASS % R1' in display_df.columns:
        display_df['Pass % R1'] = display_df['PASS % R1'].apply(lambda x: f"{x:.1f}%")
    if 'PASS % R2' in display_df.columns:
        display_df['Pass % R2'] = display_df['PASS % R2'].apply(lambda x: f"{x:.1f}%")

    # Format Change column with indicators
    if 'Change' in display_df.columns:
        display_df['Change'] = display_df['Change'].apply(
            lambda x: f"+{x:.1f}% ✅" if x > 0 else (f"{x:.1f}% ❌" if x < 0 else "0.0%")
        )

    # Select columns for display
    display_cols = ['Category', 'Count R1', 'Count R2', 'Pass % R1', 'Pass % R2', 'Change']
    display_cols = [c for c in display_cols if c in display_df.columns]

    return display_df[display_cols], comparison_df


def render_comparison_analysis(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    mapping: dict,
    round1_name: str,
    round2_name: str,
    api_key: str
) -> None:
    """Render comparison analysis between two evaluation rounds.

    Args:
        df1: First dataframe (baseline, already mapped)
        df2: Second dataframe (current, already mapped)
        mapping: Column mapping used for both dataframes
        round1_name: Display name for Round 1
        round2_name: Display name for Round 2
        api_key: Anthropic API key for insights
    """
    # Build available grouping options from mapped columns
    group_options = []
    # Add risk hierarchy levels
    for i in range(len(mapping.get('risk_hierarchy', []))):
        col = f'Risk L{i + 1}'
        if col in df1.columns and col in df2.columns:
            group_options.append(col)
    # Add attack hierarchy levels
    for i in range(len(mapping.get('attack_hierarchy', []))):
        col = f'Attack L{i + 1}'
        if col in df1.columns and col in df2.columns:
            group_options.append(col)

    if not group_options:
        st.error("No common groupable columns found between the two files.")
        return

    st.markdown("### Comparison Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Default to L2 if available, otherwise first option
        default_idx = min(1, len(group_options) - 1) if len(group_options) > 1 else 0
        group_by = st.selectbox(
            "Compare by",
            options=group_options,
            index=default_idx,
            help="Select which level to group the comparison by",
            key="comp_group_by",
        )

    with col2:
        show_details = st.checkbox("Show detailed breakdown", value=False, key="comp_show_details")

    # Category mapping for mismatched names
    with st.expander("🔗 Category Mapping (if names differ between rounds)", expanded=False):
        category_mappings = render_mapping_editor(df1, df2, group_by)

    # Load saved mappings if not edited in expander
    if not category_mappings:
        category_mappings = load_mappings().get(group_by, {})

    # Calculate comparison
    try:
        comparison_df = calculate_comparison(df1, df2, group_by, category_mappings)
    except Exception as e:
        st.error(f"Error calculating comparison: {str(e)}")
        return

    # Format for display
    display_df, raw_df = format_comparison_display(comparison_df)

    # Rename columns to use custom round names
    display_df = display_df.rename(columns={
        'Count R1': f'Count ({round1_name})',
        'Count R2': f'Count ({round2_name})',
        'Pass % R1': f'Pass % ({round1_name})',
        'Pass % R2': f'Pass % ({round2_name})',
    })

    # Show comparison table
    st.subheader("Comparison Results")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Show detailed breakdown if requested
    if show_details:
        st.markdown("#### Detailed Breakdown")

        # Show regressions (got worse)
        regressions = comparison_df[comparison_df['Change'] < -1].sort_values('Change')
        if len(regressions) > 0:
            st.markdown(f"**Regressions ({len(regressions)} categories got worse):**")
            for _, row in regressions.iterrows():
                st.markdown(
                    f"- **{row['Category']}**: {row['PASS % R1']:.1f}% → {row['PASS % R2']:.1f}% "
                    f"({row['Change']:+.1f}%)"
                )

        # Show improvements (got better)
        improvements = comparison_df[comparison_df['Change'] > 1].sort_values('Change', ascending=False)
        if len(improvements) > 0:
            st.markdown(f"**Improvements ({len(improvements)} categories got better):**")
            for _, row in improvements.iterrows():
                st.markdown(
                    f"- **{row['Category']}**: {row['PASS % R1']:.1f}% → {row['PASS % R2']:.1f}% "
                    f"({row['Change']:+.1f}%)"
                )

    # Export buttons
    st.divider()

    # Build export metadata
    from datetime import datetime
    export_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        # Add metadata to CSV export
        export_df = display_df.copy()
        csv_data = export_df.to_csv(index=False)
        metadata = f"# Comparison: {round1_name} vs {round2_name}\n# Generated: {export_date}\n# Grouped by: {group_by}\n\n"
        full_csv = metadata + csv_data

        st.download_button(
            label="📋 Download as CSV",
            data=full_csv,
            file_name=f"eval_comparison_{round1_name}_vs_{round2_name}.csv".replace(" ", "_"),
            mime="text/csv",
            key="download_comparison",
        )

    with export_col2:
        if st.button("📋 Copy to Clipboard", key="copy_comparison"):
            clipboard_data = df_to_clipboard_format(display_df)
            st.code(clipboard_data, language=None)
            st.caption("Copy the above text (Ctrl+C / Cmd+C)")

    # Grouped bar chart
    st.divider()
    st.subheader("Pass Rate Comparison")

    # Prepare data for chart
    chart_data = comparison_df[['Category', 'PASS % R1', 'PASS % R2']].copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f'{round1_name} (Baseline)',
        x=chart_data['Category'],
        y=chart_data['PASS % R1'],
        marker_color=COLORS['text_muted'],
        text=chart_data['PASS % R1'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name=f'{round2_name} (Current)',
        x=chart_data['Category'],
        y=chart_data['PASS % R2'],
        marker_color=COLORS['primary'],
        text=chart_data['PASS % R2'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
    ))

    fig.update_layout(
        barmode='group',
        title=f"Pass Rate by {group_by}",
        xaxis_title=group_by,
        yaxis_title="Pass Rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=50, l=50, r=20),
    )
    style_plotly_chart(fig)

    st.plotly_chart(fig, use_container_width=True)

    # Change magnitude chart
    st.subheader("Change by Category")

    # Sort by change for visualization
    sorted_data = comparison_df.sort_values('Change')

    fig2 = go.Figure(data=[
        go.Bar(
            x=sorted_data['Category'],
            y=sorted_data['Change'],
            marker_color=[COLORS['pass'] if c > 0 else COLORS['safety'] if c < 0 else COLORS['text_muted']
                         for c in sorted_data['Change']],
            text=sorted_data['Change'].apply(lambda x: f"{x:+.1f}%"),
            textposition='auto',
        )
    ])

    fig2.update_layout(
        title="Change in Pass Rate (% points)",
        xaxis_title=group_by,
        yaxis_title="Change (% points)",
        margin=dict(t=50, b=100, l=50, r=20),
        xaxis_tickangle=-45,
    )
    style_plotly_chart(fig2)

    st.plotly_chart(fig2, use_container_width=True)

    # AI Insights section
    if api_key:
        st.divider()
        st.subheader("AI Analysis")

        if st.button("🤖 Analyze Round Changes", type="primary", key="analyze_comparison"):
            with st.spinner("Analyzing changes with Claude..."):
                insights = get_comparison_insights(comparison_df, api_key)
                st.markdown(insights)
    else:
        st.divider()
        st.info("Enter your Claude API key in the sidebar to enable AI-powered analysis of round changes.")


def comparison_view(api_key: str) -> None:
    """Display round comparison view with two separate CSV files.

    Args:
        api_key: Anthropic API key for insights
    """
    st.header("Round Comparison")
    st.markdown("Upload two CSV files to compare evaluation rounds.")

    # Round naming section
    st.markdown("### Round Labels")
    name_col1, name_col2 = st.columns(2)

    with name_col1:
        round1_name = st.text_input(
            "Round 1 name",
            value=st.session_state.get('round1_name', 'Round 1'),
            placeholder="e.g., January Eval",
            key="round1_name_input",
        )
        st.session_state['round1_name'] = round1_name

    with name_col2:
        round2_name = st.text_input(
            "Round 2 name",
            value=st.session_state.get('round2_name', 'Round 2'),
            placeholder="e.g., March Eval",
            key="round2_name_input",
        )
        st.session_state['round2_name'] = round2_name

    st.divider()

    # Two columns for file uploaders
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{round1_name} (Baseline)**")
        file1 = st.file_uploader(
            f"Upload {round1_name} CSV",
            type=['csv'],
            key="r1_file",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown(f"**{round2_name} (Current)**")
        file2 = st.file_uploader(
            f"Upload {round2_name} CSV",
            type=['csv'],
            key="r2_file",
            label_visibility="collapsed",
        )

    # Step 1: Load Round 1
    if file1 is None:
        st.info(f"👆 Upload {round1_name} CSV to start")
        return

    # Load Round 1 CSV
    # Use header=1 to read column names from row 2 (skip row 1)
    try:
        raw_df1 = pd.read_csv(file1, header=1)
        st.success(f"{round1_name}: {len(raw_df1):,} rows loaded")
    except Exception as e:
        st.error(f"Error loading {round1_name} file: {str(e)}")
        return

    # Column mapping for Round 1
    r1_mapping_validated = st.session_state.get('r1_mapping_validated', False)

    with st.expander(f"⚙️ {round1_name} Column Mapping", expanded=not r1_mapping_validated):
        mapping1, is_valid1 = render_column_mapper(raw_df1, key_prefix="r1_")
        if is_valid1:
            st.session_state.r1_mapping_validated = True
            st.session_state.r1_column_mapping = mapping1

    if not is_valid1:
        st.warning(f"Please complete {round1_name} column mapping to continue.")
        return

    # Step 2: Load Round 2
    if file2 is None:
        st.info(f"👆 Now upload {round2_name} CSV to compare")
        return

    # Load Round 2 CSV
    try:
        raw_df2 = pd.read_csv(file2, header=1)
        st.success(f"{round2_name}: {len(raw_df2):,} rows loaded")
    except Exception as e:
        st.error(f"Error loading {round2_name} file: {str(e)}")
        return

    # Check if columns match Round 1
    r1_cols = set(raw_df1.columns)
    r2_cols = set(raw_df2.columns)

    if r1_cols == r2_cols:
        st.success(f"✅ {round2_name} columns match {round1_name} - using same mapping")
        mapping2 = mapping1
        is_valid2 = True
    else:
        # Show differences
        only_r1 = r1_cols - r2_cols
        only_r2 = r2_cols - r1_cols

        if only_r1 or only_r2:
            st.warning(f"⚠️ Column differences detected between rounds")
            if only_r1:
                st.caption(f"Only in {round1_name}: {', '.join(sorted(only_r1))}")
            if only_r2:
                st.caption(f"Only in {round2_name}: {', '.join(sorted(only_r2))}")

        # Show mapping UI for Round 2
        r2_mapping_validated = st.session_state.get('r2_mapping_validated', False)

        with st.expander(f"⚙️ {round2_name} Column Mapping", expanded=not r2_mapping_validated):
            # Pre-fill with Round 1 mapping where columns exist
            if 'r2_column_mapping' not in st.session_state:
                prefill = {}
                for k, v in mapping1.items():
                    if isinstance(v, str) and (v == '(none)' or v in raw_df2.columns):
                        prefill[k] = v
                    elif isinstance(v, list):
                        prefill[k] = [c for c in v if c in raw_df2.columns]
                st.session_state['r2_column_mapping'] = prefill

            mapping2, is_valid2 = render_column_mapper(raw_df2, key_prefix="r2_")
            if is_valid2:
                st.session_state.r2_mapping_validated = True
                st.session_state.r2_column_mapping = mapping2

        if not is_valid2:
            st.warning(f"Please complete {round2_name} column mapping to continue.")
            return

    # Apply mappings to create standardized dataframes
    df1 = apply_mapping_to_df(raw_df1, mapping1)
    df2 = apply_mapping_to_df(raw_df2, mapping2)

    st.divider()

    # Global filters for comparison - applies same filters to both dataframes
    st.markdown("### Filter Data")
    st.caption("Filters apply to both rounds for consistent comparison.")

    # Use combined data to get all filter options (union of both rounds)
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Get filters but don't apply yet - we'll apply separately to each df
    state_key = get_filter_state_key("comp_")
    filters = initialize_filters("comp_")

    with st.expander("🔍 Filters", expanded=bool(filters and any(v for v in filters.values()))):
        # Active filter chips at top
        if filters and any(v for v in filters.values()):
            if render_active_filter_chips(filters, "comp_"):
                st.rerun()
            st.divider()

        # Get cascading options for hierarchies using combined data
        risk_options = get_cascading_filter_options(combined_df, mapping1, filters, 'risk')
        attack_options = get_cascading_filter_options(combined_df, mapping1, filters, 'attack')

        # Build filter UI in columns
        num_filter_cols = 3
        filter_cols = st.columns(num_filter_cols)

        col_idx = 0

        # Risk hierarchy filters
        for col_name, options in risk_options.items():
            if not options:
                continue

            with filter_cols[col_idx % num_filter_cols]:
                current_selection = filters.get(col_name, [])
                valid_selection = [v for v in current_selection if v in options]

                selected = st.multiselect(
                    col_name,
                    options=options,
                    default=valid_selection,
                    key=f"comp_filter_{col_name}",
                    help=f"Filter by {col_name}",
                )
                filters[col_name] = selected
            col_idx += 1

        # Attack hierarchy filters
        for col_name, options in attack_options.items():
            if not options:
                continue

            with filter_cols[col_idx % num_filter_cols]:
                current_selection = filters.get(col_name, [])
                valid_selection = [v for v in current_selection if v in options]

                selected = st.multiselect(
                    col_name,
                    options=options,
                    default=valid_selection,
                    key=f"comp_filter_{col_name}",
                    help=f"Filter by {col_name}",
                )
                filters[col_name] = selected
            col_idx += 1

        # Severity filter
        if 'Severity' in combined_df.columns:
            with filter_cols[col_idx % num_filter_cols]:
                severity_options = sorted(combined_df['Severity'].dropna().unique().tolist())
                current_severity = filters.get('Severity', [])
                valid_severity = [v for v in current_severity if v in severity_options]

                selected_severity = st.multiselect(
                    "Severity",
                    options=severity_options,
                    default=valid_severity,
                    key="comp_filter_Severity",
                    help="Filter by severity level",
                )
                filters['Severity'] = selected_severity

        # Save filters to session state
        st.session_state[state_key] = filters

        # Presets section
        st.divider()
        filters = render_filter_presets(filters, "comp_")
        st.session_state[state_key] = filters

    # Apply filters to both dataframes
    filtered_df1 = apply_global_filters(df1, filters)
    filtered_df2 = apply_global_filters(df2, filters)

    # Show filter summary
    active_filters = {k: v for k, v in filters.items() if v}
    if active_filters:
        filter_summary = ", ".join([f"{k}: {len(v)}" for k, v in active_filters.items()])
        st.caption(f"🔍 Filters applied: {filter_summary}")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"{round1_name}: {len(filtered_df1):,} of {len(df1):,} rows")
        with col2:
            st.caption(f"{round2_name}: {len(filtered_df2):,} of {len(df2):,} rows")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"{round1_name}: {len(df1):,} rows")
        with col2:
            st.caption(f"{round2_name}: {len(df2):,} rows")

    if len(filtered_df1) == 0 or len(filtered_df2) == 0:
        st.warning("No data matches the selected filters in one or both rounds.")
        return

    st.divider()

    # Run comparison analysis with filtered data
    render_comparison_analysis(filtered_df1, filtered_df2, mapping1, round1_name, round2_name, api_key)


def check_auth() -> bool:
    """Check if user is authenticated when auth is enabled.

    Returns:
        True if authenticated or auth disabled, False otherwise
    """
    if not settings.ENABLE_AUTH:
        return True

    if not settings.AUTHORIZED_EMAILS:
        st.warning("Authentication is enabled but no authorized emails are configured.")
        return False

    # Check if already authenticated
    if st.session_state.get("authenticated_email"):
        return True

    # Show login form
    st.title(f"{settings.APP_ICON} {settings.APP_TITLE}")
    st.markdown("*Please sign in to continue*")

    st.divider()

    email = st.text_input(
        "Email address",
        placeholder="your.email@company.com",
        help="Enter your authorized email address",
    )

    if st.button("Sign In", type="primary"):
        if email in settings.AUTHORIZED_EMAILS:
            st.session_state["authenticated_email"] = email
            st.rerun()
        else:
            st.error("Email not authorized. Please contact an administrator.")

    return False


def safe_render(render_func, *args, **kwargs):
    """Wrapper to catch and display errors gracefully.

    Args:
        render_func: The render function to call
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of render_func, or None if an error occurred
    """
    try:
        return render_func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error rendering {render_func.__name__}: {str(e)}")
        if settings.DEBUG:
            st.exception(e)
        return None


def render_welcome_screen() -> None:
    """Show branded welcome screen when no data is uploaded."""

    # Header with logo
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <img src="https://www.aiuc-1.com/brand/aiuc1.svg" alt="AIUC-1" style="height: 48px; margin-bottom: 1rem;" onerror="this.style.display='none'">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
            Eval Dashboard
        </h1>
        <p style="color: #888; font-size: 1.1rem;">
            Analyze AI agent security evaluations
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards - 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%); border: 1px solid #2a2a2a; border-radius: 16px; padding: 2rem; text-align: center; min-height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🗺️</div>
            <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; color: #6366f1;">Test Coverage Heatmap</h3>
            <p style="color: #888; font-size: 0.875rem;">Visualize the distribution of attacks vs risks in your evaluation set.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%); border: 1px solid #2a2a2a; border-radius: 16px; padding: 2rem; text-align: center; min-height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
            <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; color: #6366f1;">Results Statistics</h3>
            <p style="color: #888; font-size: 0.875rem;">Analyze pass/fail rates, severity distributions, and trends.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%); border: 1px solid #2a2a2a; border-radius: 16px; padding: 2rem; text-align: center; min-height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔬</div>
            <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; color: #6366f1;">Top Vulnerabilities</h3>
            <p style="color: #888; font-size: 0.875rem;">Deep dive into the most severe failures with AI-powered analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%); border: 1px solid #2a2a2a; border-radius: 16px; padding: 2rem; text-align: center; min-height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📋</div>
            <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; color: #8b5cf6;">Audit Report Assets</h3>
            <p style="color: #888; font-size: 0.875rem;">Generate publication-ready tables for your audit report. Export taxonomies, examples, and summaries for Google Docs or Word.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Getting started steps
    st.markdown("""
    <div style="background: #1a1a1a; border-radius: 12px; padding: 2rem; max-width: 600px; margin: 0 auto;">
        <h3 style="text-align: center; margin-bottom: 1.5rem;">Get Started</h3>
        <div style="display: flex; flex-direction: column; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: #6366f1; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">1</div>
                <span>Upload your Round 1 CSV in the sidebar</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: #6366f1; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">2</div>
                <span>(Optional) Upload Round 2 CSV to enable comparison mode</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: #6366f1; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">3</div>
                <span>Map your columns to the expected fields</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: #6366f1; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">4</div>
                <span>Explore your data!</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def load_data(uploaded_file) -> pd.DataFrame:
    """Load and cache uploaded CSV data.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Loaded DataFrame or None
    """
    if uploaded_file is None:
        return None

    # Use file name + size as cache key
    cache_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = {}

    if cache_key not in st.session_state.loaded_data:
        # Use header=1 to read column names from row 2
        df = pd.read_csv(uploaded_file, header=1)
        st.session_state.loaded_data[cache_key] = df

    return st.session_state.loaded_data[cache_key]


def render_api_key_section() -> str:
    """Render API key configuration section in sidebar.

    Returns:
        API key string (or empty string if not configured)
    """
    if settings.has_api_key:
        # API key is configured in .env - show status
        st.success(f"API Key loaded from {settings.api_key_source}", icon="✅")

        # Optional: allow override
        with st.expander("API Key Settings"):
            st.markdown(f"**Source:** {settings.api_key_source}")
            st.markdown(f"**Key:** `{settings.api_key_preview}`")

            override = st.checkbox("Override with different key", key="override_api_key_checkbox")
            if override:
                new_key = st.text_input(
                    "New API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    key="override_api_key_input"
                )
                if new_key:
                    st.session_state['api_key_override'] = new_key
                    st.info("Using override key for this session")
            else:
                # Clear override if checkbox unchecked
                if 'api_key_override' in st.session_state:
                    del st.session_state['api_key_override']

        # Use the configured key (or override if set)
        return st.session_state.get('api_key_override', settings.ANTHROPIC_API_KEY)
    else:
        # No API key configured - show warning and input
        st.warning("API Key not configured", icon="⚠️")

        api_key = st.text_input(
            "Claude API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Enter your Anthropic API key, or add it to .env file to save permanently"
        )

        if not api_key:
            with st.expander("How to configure permanently"):
                st.markdown("""
**Option 1: Add to .env file (recommended)**

1. Open the `.env` file in your project folder
2. Add your API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
3. Save and restart the app

**Option 2: Enter above**

Enter your key in the field above. It will only persist for this session.

---
Get your API key at: [console.anthropic.com](https://console.anthropic.com/)
                """)

        return api_key or ""


def ask_ai_question(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict, question: str, api_key: str) -> str:
    """Answer user's question about the data using Claude.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
        question: User's question
        api_key: Anthropic API key

    Returns:
        AI response text
    """
    try:
        import anthropic
    except ImportError:
        return "⚠️ Anthropic library not installed. Run `pip install anthropic` to enable AI features."

    if not api_key:
        return "⚠️ Please configure your API key to use AI features."

    # Get severity column name
    severity_col = 'Severity'

    # Build context about the data
    context = f"""You are analyzing AI agent evaluation data.

Round 1 Data:
- Total evals: {len(df1)}
- Pass rate: {(df1[severity_col] == 'PASS').mean()*100:.1f}% if severity_col in df1.columns else 'N/A'
- Severity distribution: {df1[severity_col].value_counts().to_dict() if severity_col in df1.columns else 'N/A'}
"""

    if df2 is not None:
        context += f"""
Round 2 Data:
- Total evals: {len(df2)}
- Pass rate: {(df2[severity_col] == 'PASS').mean()*100:.1f}% if severity_col in df2.columns else 'N/A'
- Severity distribution: {df2[severity_col].value_counts().to_dict() if severity_col in df2.columns else 'N/A'}
"""

    context += f"""
Column mapping:
- Risk hierarchy: {mapping.get('risk_hierarchy', [])}
- Attack hierarchy: {mapping.get('attack_hierarchy', [])}
- Severity column: {mapping.get('severity', 'Not set')}

User question: {question}

Provide a clear, concise answer based on the data.
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": context}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def render_ai_chat_panel(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict, api_key: str) -> None:
    """Render floating AI chat panel for asking questions.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
        api_key: Anthropic API key
    """
    with st.sidebar:
        st.divider()
        st.subheader("🤖 Ask AI")

        question = st.text_area(
            "Your question about the data:",
            key="ai_question",
            placeholder="e.g., What are the most common failure patterns?"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Get Answer", type="primary", use_container_width=True):
                if question and api_key:
                    with st.spinner("Thinking..."):
                        answer = ask_ai_question(df1, df2, mapping, question, api_key)
                        st.session_state['ai_answer'] = answer
                elif not api_key:
                    st.error("Please configure API key")
                else:
                    st.warning("Please enter a question")

        with col2:
            if st.button("Close", use_container_width=True):
                st.session_state['show_ai_chat'] = False
                st.rerun()

        # Show previous answer if exists
        if 'ai_answer' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['ai_answer'])


def analyze_test_coverage(
    df: pd.DataFrame,
    risk_col: str,
    attack_col: str,
    api_key: str
) -> str:
    """Use AI to analyze test coverage distribution.

    Args:
        df: Input dataframe
        risk_col: Risk column name
        attack_col: Attack column name
        api_key: Anthropic API key

    Returns:
        AI-generated analysis text
    """
    if not api_key:
        return "API key not configured. Please add your Anthropic API key in the sidebar."

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Get coverage stats
        summary = get_distribution_summary(df, risk_col, attack_col)
        gaps = get_coverage_gaps(df, risk_col, attack_col)
        counts, pct = create_heatmap_data(df, risk_col, attack_col)

        # Remove Total row/col for analysis
        if 'Total' in counts.index:
            counts = counts.drop('Total', axis=0)
        if 'Total' in counts.columns:
            counts = counts.drop('Total', axis=1)

        context = f"""Test Coverage Analysis:

Total Test Scenarios: {summary['total_evals']}
Risk Categories Tested: {summary['unique_risks']}
Attack Types Tested: {summary['unique_attacks']}

Most Tested Risk: {summary['top_risk']} ({summary['top_risk_pct']:.1f}% of tests)
Most Tested Attack: {summary['top_attack']} ({summary['top_attack_pct']:.1f}% of tests)

Coverage Stats:
- {gaps['coverage_percentage']:.1f}% of risk/attack combinations have tests
- {len(gaps['zero_coverage_combinations'])} combinations have NO tests
- {len(gaps['low_coverage_combinations'])} combinations have <5 tests

Test Distribution Matrix (rows=risk, cols=attack):
{counts.to_string()}
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze this AI agent safety test coverage data:

{context}

Focus on:
1. Which risk/attack combinations are over-tested vs under-tested?
2. Are there critical gaps in test coverage?
3. Is the test distribution balanced or skewed?
4. Recommendations for improving test coverage balance

Keep response concise and actionable."""
                }
            ]
        )

        return message.content[0].text

    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def analyze_coverage_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    risk_col: str,
    attack_col: str,
    api_key: str
) -> str:
    """Use AI to analyze coverage changes between rounds.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe
        risk_col: Risk column name
        attack_col: Attack column name
        api_key: Anthropic API key

    Returns:
        AI-generated comparison analysis
    """
    if not api_key:
        return "API key not configured. Please add your Anthropic API key in the sidebar."

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Get stats for both rounds
        summary1 = get_distribution_summary(df1, risk_col, attack_col)
        summary2 = get_distribution_summary(df2, risk_col, attack_col)

        counts1, _ = create_heatmap_data(df1, risk_col, attack_col)
        counts2, _ = create_heatmap_data(df2, risk_col, attack_col)

        # Remove Total
        for df in [counts1, counts2]:
            if 'Total' in df.index:
                df.drop('Total', axis=0, inplace=True)
            if 'Total' in df.columns:
                df.drop('Total', axis=1, inplace=True)

        # Align and calculate diff
        all_risks = sorted(set(counts1.index) | set(counts2.index))
        all_attacks = sorted(set(counts1.columns) | set(counts2.columns))

        c1_aligned = counts1.reindex(index=all_risks, columns=all_attacks, fill_value=0)
        c2_aligned = counts2.reindex(index=all_risks, columns=all_attacks, fill_value=0)
        diff = c2_aligned - c1_aligned

        context = f"""Test Coverage Comparison (Round 1 vs Round 2):

Round 1: {summary1['total_evals']} total tests
Round 2: {summary2['total_evals']} total tests
Change: {summary2['total_evals'] - summary1['total_evals']:+d} tests

Round 1 Distribution:
{c1_aligned.to_string()}

Round 2 Distribution:
{c2_aligned.to_string()}

Change Matrix:
{diff.to_string()}

Biggest increases: {diff.stack().nlargest(3).to_dict()}
Biggest decreases: {diff.stack().nsmallest(3).to_dict()}
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze the changes in test coverage between evaluation rounds:

{context}

Focus on:
1. How has the testing strategy changed?
2. Which areas received more/less testing?
3. Are coverage gaps being addressed or widening?
4. Recommendations based on the changes

Keep response concise and actionable."""
                }
            ]
        )

        return message.content[0].text

    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def render_single_heatmap(
    df: pd.DataFrame,
    risk_col: str,
    attack_col: str,
    title_suffix: str = ""
) -> None:
    """Render a single coverage heatmap.

    Args:
        df: Input dataframe
        risk_col: Risk column for rows
        attack_col: Attack column for columns
        title_suffix: Optional suffix for title (e.g., " (Round 1)")
    """
    # Create count matrix
    counts, pct = create_heatmap_data(df, risk_col, attack_col)

    # Remove Total row/column
    if 'Total' in counts.index:
        counts = counts.drop('Total', axis=0)
    if 'Total' in counts.columns:
        counts = counts.drop('Total', axis=1)

    # Create heatmap
    fig = create_heatmap_figure(
        counts,
        title=f"Test Coverage: {risk_col} vs {attack_col}{title_suffix}",
        color_scale='Blues',
        value_format='d',
        color_label="# of Tests",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_tests = counts.values.sum()
    coverage_pct = (counts > 0).sum().sum() / counts.size * 100
    max_tests = counts.values.max()
    avg_tests = counts.values.mean()

    col1.metric("Total Tests", f"{int(total_tests):,}")
    col2.metric("Coverage", f"{coverage_pct:.0f}%")
    col3.metric("Max per Cell", f"{int(max_tests)}")
    col4.metric("Avg per Cell", f"{avg_tests:.1f}")


def render_comparison_heatmaps(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    risk_col: str,
    attack_col: str,
    api_key: str
) -> None:
    """Render comparison heatmaps for two rounds.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe
        risk_col: Risk column for rows
        attack_col: Attack column for columns
        api_key: Anthropic API key
    """
    tab1, tab2, tab3 = st.tabs(["Round 1", "Round 2", "Difference"])

    with tab1:
        render_single_heatmap(df1, risk_col, attack_col, " (Round 1)")

    with tab2:
        render_single_heatmap(df2, risk_col, attack_col, " (Round 2)")

    with tab3:
        # Calculate aligned counts
        counts1, _ = create_heatmap_data(df1, risk_col, attack_col)
        counts2, _ = create_heatmap_data(df2, risk_col, attack_col)

        # Remove Total
        for df in [counts1, counts2]:
            if 'Total' in df.index:
                df.drop('Total', axis=0, inplace=True)
            if 'Total' in df.columns:
                df.drop('Total', axis=1, inplace=True)

        # Align
        all_risks = sorted(set(counts1.index) | set(counts2.index))
        all_attacks = sorted(set(counts1.columns) | set(counts2.columns))

        c1_aligned = counts1.reindex(index=all_risks, columns=all_attacks, fill_value=0)
        c2_aligned = counts2.reindex(index=all_risks, columns=all_attacks, fill_value=0)
        diff = c2_aligned - c1_aligned

        # Create difference heatmap
        fig = px.imshow(
            diff.values,
            labels=dict(x="Attack Type", y="Risk Category", color="Change"),
            x=diff.columns.tolist(),
            y=diff.index.tolist(),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            aspect='auto',
            text_auto='+d'
        )
        fig.update_layout(
            title="Coverage Change (Round 2 - Round 1)",
            height=500,
            xaxis={'tickangle': 45}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary of changes
        added = (diff > 0).sum().sum()
        removed = (diff < 0).sum().sum()
        unchanged = (diff == 0).sum().sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("More Tests", int(added))
        col2.metric("Fewer Tests", int(removed))
        col3.metric("Unchanged", int(unchanged))
        col4.metric("Net Change", f"{int(diff.values.sum()):+d}")

    # AI Comparison Analysis
    st.divider()
    st.subheader("AI Coverage Comparison Analysis")

    if st.button("Analyze Coverage Changes", key="analyze_coverage_comparison_btn"):
        with st.spinner("Analyzing coverage changes..."):
            analysis = analyze_coverage_comparison(
                df1, df2, risk_col, attack_col, api_key
            )
            st.session_state['coverage_comparison_analysis'] = analysis

    if 'coverage_comparison_analysis' in st.session_state:
        st.markdown(st.session_state['coverage_comparison_analysis'])


def render_evals_heatmap(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict, api_key: str) -> None:
    """View 1: Test Coverage Heatmap - Shows HOW we tested the agent.

    This heatmap shows the distribution of test scenarios across risk categories
    and attack types. It answers: "How comprehensively did we test the agent?"

    Args:
        df1: Round 1 dataframe (already mapped)
        df2: Round 2 dataframe (or None, already mapped)
        mapping: Column mapping dict
        api_key: Anthropic API key
    """
    st.header("Test Coverage Heatmap")
    st.caption("Distribution of adversarial test scenarios by risk category and attack type")

    # Get hierarchy columns
    risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
    attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]

    risk_cols = [c for c in risk_cols if c in df1.columns]
    attack_cols = [c for c in attack_cols if c in df1.columns]

    if not risk_cols or not attack_cols:
        st.warning("Need both Risk and Attack columns to show heatmap. Please check your column mapping.")
        return

    # Settings - default to L1 (broadest level)
    with st.expander("Heatmap Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            risk_level = st.selectbox(
                "Risk Level",
                options=risk_cols,
                index=0,  # Default to L1
                key="heatmap_risk_level"
            )

        with col2:
            attack_level = st.selectbox(
                "Attack Level",
                options=attack_cols,
                index=0,  # Default to L1
                key="heatmap_attack_level"
            )

    # Render heatmaps
    if df2 is not None and risk_level in df2.columns and attack_level in df2.columns:
        render_comparison_heatmaps(df1, df2, risk_level, attack_level, api_key)
    else:
        render_single_heatmap(df1, risk_level, attack_level)

        # Coverage gaps and AI analysis for single round
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Coverage Summary")
            gaps = get_coverage_gaps(df1, risk_level, attack_level)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Coverage", f"{gaps['coverage_percentage']:.0f}%",
                          help="% of risk/attack combinations with at least 1 test")
                st.metric("Zero Coverage", len(gaps['zero_coverage_combinations']),
                          help="Combinations with no tests")
            with m2:
                st.metric("Combinations Tested",
                          f"{gaps['covered_combinations']}/{gaps['total_combinations']}")
                st.metric("Low Coverage (<5)", len(gaps['low_coverage_combinations']))

        with col2:
            st.subheader("Coverage Gaps")
            if gaps['zero_coverage_combinations']:
                st.write("**Untested combinations:**")
                gap_df = pd.DataFrame(gaps['zero_coverage_combinations'][:10])
                st.dataframe(gap_df, use_container_width=True, hide_index=True)
                if len(gaps['zero_coverage_combinations']) > 10:
                    st.caption(f"...and {len(gaps['zero_coverage_combinations']) - 10} more")
            else:
                st.success("All risk/attack combinations have test coverage!")

        # AI Analysis
        st.divider()
        st.subheader("AI Coverage Analysis")

        if st.button("Analyze Test Coverage", key="analyze_coverage_btn"):
            with st.spinner("Analyzing test coverage..."):
                analysis = analyze_test_coverage(
                    df1, risk_level, attack_level, api_key
                )
                st.session_state['coverage_analysis'] = analysis

        if 'coverage_analysis' in st.session_state:
            st.markdown(st.session_state['coverage_analysis'])


def render_stats_filters(df: pd.DataFrame, mapping: dict, key_prefix: str = "") -> pd.DataFrame:
    """Render filters specific to the statistics view.

    Args:
        df: Input dataframe
        mapping: Column mapping dict
        key_prefix: Prefix for widget keys

    Returns:
        Filtered dataframe
    """
    if df is None:
        return None

    with st.expander("Filters", expanded=False):
        cols = st.columns(4)

        filters = {}

        # Risk filter
        risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
        risk_cols = [c for c in risk_cols if c in df.columns]
        if risk_cols:
            with cols[0]:
                risk_vals = df[risk_cols[0]].dropna().unique().tolist()
                selected = st.multiselect(
                    f"Filter by {risk_cols[0]}",
                    options=sorted(risk_vals),
                    key=f"{key_prefix}risk_filter"
                )
                if selected:
                    filters[risk_cols[0]] = selected

        # Attack filter
        attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
        attack_cols = [c for c in attack_cols if c in df.columns]
        if attack_cols:
            with cols[1]:
                attack_vals = df[attack_cols[0]].dropna().unique().tolist()
                selected = st.multiselect(
                    f"Filter by {attack_cols[0]}",
                    options=sorted(attack_vals),
                    key=f"{key_prefix}attack_filter"
                )
                if selected:
                    filters[attack_cols[0]] = selected

        # Severity filter
        if 'Severity' in df.columns:
            with cols[2]:
                sev_vals = ['PASS', 'P4', 'P3', 'P2', 'P1', 'P0']
                available_sevs = [s for s in sev_vals if s in df['Severity'].unique()]
                selected = st.multiselect(
                    "Filter by Severity",
                    options=available_sevs,
                    key=f"{key_prefix}severity_filter"
                )
                if selected:
                    filters['Severity'] = selected

        with cols[3]:
            st.write("")  # Spacer
            st.write("")
            if st.button("Clear Filters", key=f"{key_prefix}clear"):
                st.rerun()

    # Apply filters
    filtered = df.copy()
    for col, vals in filters.items():
        if vals:
            filtered = filtered[filtered[col].isin(vals)]

    return filtered


def render_performance_summary(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict) -> None:
    """Render summary performance cards.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
    """
    if 'Severity' not in df1.columns:
        st.warning("Map severity column to see performance metrics")
        return

    # Calculate metrics for Round 1
    total1 = len(df1)
    passed1 = (df1['Severity'] == 'PASS').sum()
    pass_rate1 = passed1 / total1 * 100 if total1 > 0 else 0
    failed1 = total1 - passed1

    if df2 is not None and 'Severity' in df2.columns:
        # Calculate metrics for Round 2
        total2 = len(df2)
        passed2 = (df2['Severity'] == 'PASS').sum()
        pass_rate2 = passed2 / total2 * 100 if total2 > 0 else 0
        failed2 = total2 - passed2

        # Show comparison
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta = pass_rate2 - pass_rate1
            st.metric("Pass Rate", f"{pass_rate2:.1f}%", delta=f"{delta:+.1f}%")

        with col2:
            delta = total2 - total1
            st.metric("Total Evals", f"{total2:,}", delta=f"{delta:+,}")

        with col3:
            delta = passed2 - passed1
            st.metric("Passed", f"{passed2:,}", delta=f"{delta:+,}")

        with col4:
            delta = failed2 - failed1
            st.metric("Failed", f"{failed2:,}", delta=f"{delta:+,}", delta_color="inverse")

    else:
        # Single round
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Pass Rate", f"{pass_rate1:.1f}%")
        col2.metric("Total Evals", f"{total1:,}")
        col3.metric("Passed", f"{passed1:,}")
        col4.metric("Failed", f"{failed1:,}")


def get_category_pass_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Get pass rates by category for AI context building.

    Args:
        df: Input dataframe
        group_col: Column to group by

    Returns:
        DataFrame with Category, Total, Pass Rate columns
    """
    if 'Severity' not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    stats = df.groupby(group_col).agg(
        Total=('Severity', 'count'),
        Passed=('Severity', lambda x: (x == 'PASS').sum()),
    ).reset_index()

    stats.columns = ['Category', 'Total', 'Passed']
    stats['Pass Rate'] = (stats['Passed'] / stats['Total'] * 100).round(1)

    return stats.sort_values('Pass Rate')


def render_passfail_section(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict) -> None:
    """Render pass/fail statistics table with full severity breakdown.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
    """
    # Settings for this section
    col1, col2, col3 = st.columns([2, 1, 1])

    # Build group options
    risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
    attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
    risk_cols = [c for c in risk_cols if c in df1.columns]
    attack_cols = [c for c in attack_cols if c in df1.columns]
    group_options = risk_cols + attack_cols

    if not group_options:
        st.warning("No grouping columns available")
        return

    with col1:
        group_by = st.selectbox(
            "Group by",
            options=group_options,
            index=min(1, len(group_options) - 1) if len(group_options) > 1 else 0,
            key="passfail_groupby"
        )

    with col2:
        show_counts = st.checkbox("Show counts", value=False, key="passfail_counts")

    with col3:
        show_round = st.checkbox("Show by round", value=df2 is not None, key="passfail_round")

    # Calculate stats with full severity breakdown
    stats1 = calculate_stats_with_mapping(df1, group_by, mapping)
    stats1['Round'] = 'Round 1'

    if df2 is not None and show_round and group_by in df2.columns:
        stats2 = calculate_stats_with_mapping(df2, group_by, mapping)
        stats2['Round'] = 'Round 2'

        # Combine and sort
        combined = pd.concat([stats1, stats2])
        combined = combined.sort_values(['Category', 'Round'])

        display_df = format_stats_for_display(combined, show_counts, include_round=True)
    else:
        display_df = format_stats_for_display(stats1, show_counts, include_round=False)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Export buttons
    col1, col2 = st.columns(2)
    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "pass_fail_stats.csv", "text/csv", key="download_passfail")
    with col2:
        if st.button("Copy to Clipboard", key="copy_passfail"):
            st.code(display_df.to_csv(sep='\t', index=False), language=None)


def render_severity_section(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict) -> None:
    """Render severity distribution charts with AIUC-1 styling.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
    """
    if 'Severity' not in df1.columns:
        return

    # Get failure data only
    failures1 = df1[df1['Severity'] != 'PASS']

    # Severity order and colors from theme
    order = ['P4', 'P3', 'P2', 'P1', 'P0']

    col1, col2 = st.columns(2)

    with col1:
        if len(failures1) > 0:
            dist1 = failures1['Severity'].value_counts().reset_index()
            dist1.columns = ['Severity', 'Count']

            # Sort by severity
            dist1['sort_key'] = dist1['Severity'].apply(lambda x: order.index(x) if x in order else 99)
            dist1 = dist1.sort_values('sort_key').drop('sort_key', axis=1)

            fig = px.bar(
                dist1, x='Severity', y='Count',
                title='Round 1 - Failure Severity',
                color='Severity',
                color_discrete_map=SEVERITY_COLORS
            )
            fig.update_layout(showlegend=False)
            style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No failures in Round 1!")

    with col2:
        if df2 is not None and 'Severity' in df2.columns:
            failures2 = df2[df2['Severity'] != 'PASS']

            if len(failures2) > 0:
                dist2 = failures2['Severity'].value_counts().reset_index()
                dist2.columns = ['Severity', 'Count']
                dist2['sort_key'] = dist2['Severity'].apply(lambda x: order.index(x) if x in order else 99)
                dist2 = dist2.sort_values('sort_key').drop('sort_key', axis=1)

                fig = px.bar(
                    dist2, x='Severity', y='Count',
                    title='Round 2 - Failure Severity',
                    color='Severity',
                    color_discrete_map=SEVERITY_COLORS
                )
                fig.update_layout(showlegend=False)
                style_plotly_chart(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No failures in Round 2!")
        else:
            st.info("Upload Round 2 to see comparison")


def render_attack_effectiveness_section(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict) -> None:
    """Render attack effectiveness analysis.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
    """
    attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
    attack_cols = [c for c in attack_cols if c in df1.columns]

    if not attack_cols or 'Severity' not in df1.columns:
        st.warning("Map attack and severity columns to see attack effectiveness")
        return

    # Settings
    attack_col = st.selectbox(
        "Attack level",
        options=attack_cols,
        index=min(1, len(attack_cols) - 1) if len(attack_cols) > 1 else 0,
        key="attack_eff_level"
    )

    # Calculate effectiveness (lower pass rate = more effective attack)
    effectiveness1 = df1.groupby(attack_col).agg(
        Total=('Severity', 'count'),
        Passes=('Severity', lambda x: (x == 'PASS').sum())
    ).reset_index()
    effectiveness1['Pass Rate'] = (effectiveness1['Passes'] / effectiveness1['Total'] * 100).round(1)
    effectiveness1 = effectiveness1.rename(columns={attack_col: 'Attack'})
    effectiveness1 = effectiveness1.sort_values('Pass Rate')

    if df2 is not None and attack_col in df2.columns and 'Severity' in df2.columns:
        effectiveness2 = df2.groupby(attack_col).agg(
            Total=('Severity', 'count'),
            Passes=('Severity', lambda x: (x == 'PASS').sum())
        ).reset_index()
        effectiveness2['Pass Rate'] = (effectiveness2['Passes'] / effectiveness2['Total'] * 100).round(1)
        effectiveness2 = effectiveness2.rename(columns={attack_col: 'Attack'})

        # Merge
        comparison = pd.merge(
            effectiveness1[['Attack', 'Total', 'Pass Rate']],
            effectiveness2[['Attack', 'Total', 'Pass Rate']],
            on='Attack', suffixes=(' R1', ' R2'), how='outer'
        ).fillna(0)
        comparison['Change'] = comparison['Pass Rate R2'] - comparison['Pass Rate R1']
        comparison = comparison.sort_values('Pass Rate R2')

        # Format for display
        display_df = comparison.copy()
        display_df['Pass Rate R1'] = display_df['Pass Rate R1'].apply(lambda x: f"{x:.1f}%")
        display_df['Pass Rate R2'] = display_df['Pass Rate R2'].apply(lambda x: f"{x:.1f}%")
        display_df['Change'] = display_df['Change'].apply(lambda x: f"{x:+.1f}%")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Chart
        chart_data = comparison.head(10).copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Round 1',
            x=chart_data['Attack'],
            y=chart_data['Pass Rate R1'].astype(float),
            marker_color=COLORS['primary']
        ))
        fig.add_trace(go.Bar(
            name='Round 2',
            x=chart_data['Attack'],
            y=chart_data['Pass Rate R2'].astype(float),
            marker_color=COLORS['secondary']
        ))
        fig.update_layout(
            title='Top 10 Most Effective Attacks (Lower = More Effective)',
            barmode='group',
            xaxis_tickangle=45,
            yaxis_title='Pass Rate %'
        )
        style_plotly_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    else:
        display_df = effectiveness1[['Attack', 'Total', 'Pass Rate']].copy()
        display_df['Pass Rate'] = display_df['Pass Rate'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            effectiveness1.head(10), x='Attack', y='Pass Rate',
            title='Top 10 Most Effective Attacks (Lower = More Effective)',
            color_discrete_sequence=[COLORS['primary']]
        )
        fig.update_layout(xaxis_tickangle=45)
        style_plotly_chart(fig)
        st.plotly_chart(fig, use_container_width=True)


def render_trends_section(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict) -> None:
    """Render round-over-round trends (only when comparing).

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe
        mapping: Column mapping dict
    """
    risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
    risk_cols = [c for c in risk_cols if c in df1.columns]

    if not risk_cols or 'Severity' not in df1.columns:
        return

    risk_col = st.selectbox("View trends by", options=risk_cols, key="trends_level")

    if risk_col not in df2.columns:
        st.warning(f"Column {risk_col} not found in Round 2 data")
        return

    # Calculate pass rates for each category in both rounds
    rates1 = df1.groupby(risk_col).agg(
        Total=('Severity', 'count'),
        Passes=('Severity', lambda x: (x == 'PASS').sum())
    ).reset_index()
    rates1['Pass Rate'] = (rates1['Passes'] / rates1['Total'] * 100).round(1)
    rates1 = rates1.rename(columns={risk_col: 'Category'})
    rates1['Round'] = 'Round 1'

    rates2 = df2.groupby(risk_col).agg(
        Total=('Severity', 'count'),
        Passes=('Severity', lambda x: (x == 'PASS').sum())
    ).reset_index()
    rates2['Pass Rate'] = (rates2['Passes'] / rates2['Total'] * 100).round(1)
    rates2 = rates2.rename(columns={risk_col: 'Category'})
    rates2['Round'] = 'Round 2'

    combined = pd.concat([rates1[['Category', 'Pass Rate', 'Round']], rates2[['Category', 'Pass Rate', 'Round']]])

    # Line chart showing change with risk-based colors
    categories = combined['Category'].unique()
    color_map = {cat: get_risk_color(cat) for cat in categories}

    fig = px.line(
        combined, x='Round', y='Pass Rate', color='Category',
        title='Pass Rate Trends by Category',
        markers=True,
        color_discrete_map=color_map
    )
    fig.update_layout(yaxis_title='Pass Rate %')
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    style_plotly_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Show biggest changes
    st.markdown("**Biggest Changes:**")

    merged = pd.merge(
        rates1[['Category', 'Pass Rate']],
        rates2[['Category', 'Pass Rate']],
        on='Category', suffixes=(' R1', ' R2'), how='inner'
    )
    merged['Change'] = merged['Pass Rate R2'] - merged['Pass Rate R1']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most Improved:**")
        improved = merged.nlargest(3, 'Change')
        for _, row in improved.iterrows():
            if row['Change'] > 0:
                st.write(f"- {row['Category']}: +{row['Change']:.1f}%")

    with col2:
        st.markdown("**Most Regressed:**")
        regressed = merged.nsmallest(3, 'Change')
        for _, row in regressed.iterrows():
            if row['Change'] < 0:
                st.write(f"- {row['Category']}: {row['Change']:.1f}%")


def generate_statistics_report(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    mapping: dict,
    api_key: str
) -> str:
    """Generate a comprehensive statistics report using AI.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
        api_key: Anthropic API key

    Returns:
        AI-generated report text
    """
    if not api_key:
        return "API key not configured. Please add your Anthropic API key in the sidebar."

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build context
        total1 = len(df1)
        passed1 = (df1['Severity'] == 'PASS').sum() if 'Severity' in df1.columns else 0
        pass_rate1 = passed1 / total1 * 100 if total1 > 0 else 0
        critical1 = df1['Severity'].isin(['P0', 'P1']).sum() if 'Severity' in df1.columns else 0

        context = f"""# Evaluation Statistics Report

## Round 1 Summary
- Total Evaluations: {total1:,}
- Passed: {passed1:,} ({pass_rate1:.1f}%)
- Failed: {total1 - passed1:,} ({100 - pass_rate1:.1f}%)
- Critical Failures (P0/P1): {critical1}
"""

        # Severity distribution
        if 'Severity' in df1.columns:
            sev_dist = df1['Severity'].value_counts().to_dict()
            context += f"\nSeverity Distribution: {sev_dist}\n"

        # Risk breakdown
        risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
        risk_cols = [c for c in risk_cols if c in df1.columns]
        if risk_cols and 'Severity' in df1.columns:
            risk_stats = get_category_pass_rates(df1, risk_cols[0])
            context += f"\nPass Rates by {risk_cols[0]}:\n"
            for _, row in risk_stats.iterrows():
                context += f"- {row['Category']}: {row['Pass Rate']:.1f}% ({row['Total']} evals)\n"

        # Attack breakdown
        attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
        attack_cols = [c for c in attack_cols if c in df1.columns]
        if attack_cols and 'Severity' in df1.columns:
            attack_stats = get_category_pass_rates(df1, attack_cols[0])
            context += f"\nPass Rates by {attack_cols[0]}:\n"
            for _, row in attack_stats.head(10).iterrows():
                context += f"- {row['Category']}: {row['Pass Rate']:.1f}% ({row['Total']} evals)\n"

        # Round 2 comparison
        if df2 is not None and 'Severity' in df2.columns:
            total2 = len(df2)
            passed2 = (df2['Severity'] == 'PASS').sum()
            pass_rate2 = passed2 / total2 * 100 if total2 > 0 else 0
            critical2 = df2['Severity'].isin(['P0', 'P1']).sum()

            context += f"""
## Round 2 Summary
- Total Evaluations: {total2:,}
- Passed: {passed2:,} ({pass_rate2:.1f}%)
- Failed: {total2 - passed2:,} ({100 - pass_rate2:.1f}%)
- Critical Failures (P0/P1): {critical2}

## Round-over-Round Changes
- Pass Rate Change: {pass_rate2 - pass_rate1:+.1f}%
- Critical Failure Change: {critical2 - critical1:+d}
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Based on this evaluation data, generate a comprehensive statistics report:

{context}

Please provide:
1. **Executive Summary** - 2-3 sentence overview of overall performance
2. **Key Findings** - Top 3-5 insights from the data
3. **Strengths** - Where the agent performs well
4. **Weaknesses** - Where the agent struggles most
5. **Recommendations** - Actionable next steps to improve performance
{"6. **Round Comparison Analysis** - What changed between rounds and why" if df2 is not None else ""}

Format the report in clean markdown suitable for sharing."""
                }
            ]
        )

        return message.content[0].text

    except Exception as e:
        return f"Error generating report: {str(e)}"


def render_results_statistics(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict, api_key: str) -> None:
    """View 2: Consolidated statistics and charts.

    Args:
        df1: Round 1 dataframe (already mapped)
        df2: Round 2 dataframe (or None, already mapped)
        mapping: Column mapping dict
        api_key: Anthropic API key
    """
    st.header("Results Statistics")

    comparison_mode = df2 is not None

    # Global filters for this view
    filtered_df1 = render_stats_filters(df1, mapping, key_prefix="stats_")
    filtered_df2 = render_stats_filters(df2, mapping, key_prefix="stats_r2_") if comparison_mode else None

    if len(filtered_df1) == 0:
        st.warning("No data matches the selected filters.")
        return

    # Show data info
    if comparison_mode and filtered_df2 is not None:
        st.info(f"Showing: Round 1 ({len(filtered_df1):,} rows) vs Round 2 ({len(filtered_df2):,} rows)")
    else:
        st.info(f"Showing: {len(filtered_df1):,} rows")

    # Section 1: Summary Cards
    st.subheader("Overall Performance")
    render_performance_summary(filtered_df1, filtered_df2, mapping)

    st.divider()

    # Section 2: Pass/Fail Statistics Table
    st.subheader("Pass/Fail Statistics")
    render_passfail_section(filtered_df1, filtered_df2, mapping)

    st.divider()

    # Section 3: Severity Distribution
    st.subheader("Failure Severity Distribution")
    render_severity_section(filtered_df1, filtered_df2, mapping)

    st.divider()

    # Section 4: Attack Effectiveness
    st.subheader("Attack Effectiveness")
    render_attack_effectiveness_section(filtered_df1, filtered_df2, mapping)

    # Section 5: Trends (if comparing rounds)
    if comparison_mode and filtered_df2 is not None and len(filtered_df2) > 0:
        st.divider()
        st.subheader("Round-over-Round Trends")
        render_trends_section(filtered_df1, filtered_df2, mapping)

    # AI Analysis button
    st.divider()
    st.subheader("AI Statistics Report")

    if st.button("Generate Full Statistics Report", key="ai_stats_report"):
        with st.spinner("Generating comprehensive report..."):
            report = generate_statistics_report(filtered_df1, filtered_df2, mapping, api_key)
            st.session_state['stats_report'] = report

    if 'stats_report' in st.session_state:
        st.markdown(st.session_state['stats_report'])

        # Download button
        st.download_button(
            "Download Report",
            data=st.session_state['stats_report'],
            file_name="eval_statistics_report.md",
            mime="text/markdown",
            key="download_stats_report"
        )


def analyze_vulnerability_root_causes(
    failures_df: pd.DataFrame,
    mapping: dict,
    api_key: str
) -> str:
    """Analyze root causes across multiple vulnerabilities.

    Args:
        failures_df: DataFrame with failure data
        mapping: Column mapping dict
        api_key: Anthropic API key

    Returns:
        AI-generated analysis text
    """
    if not api_key:
        return "API key not configured. Please add your Anthropic API key in the sidebar."

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build risk and attack column lists
        risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
        attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
        risk_cols = [c for c in risk_cols if c in failures_df.columns]
        attack_cols = [c for c in attack_cols if c in failures_df.columns]

        # Build context from failures
        failure_summaries = []
        for _, row in failures_df.head(20).iterrows():
            summary = {
                'severity': row.get('Severity'),
                'risk': ' > '.join([str(row.get(c, '')) for c in risk_cols if c in row and pd.notna(row.get(c))]),
                'attack': ' > '.join([str(row.get(c, '')) for c in attack_cols if c in row and pd.notna(row.get(c))]),
            }
            if 'Justification' in row and pd.notna(row.get('Justification')):
                summary['justification'] = str(row.get('Justification', ''))[:300]
            failure_summaries.append(summary)

        prompt = f"""Analyze these {len(failure_summaries)} AI agent evaluation failures:

{failure_summaries}

Please provide:

## Root Cause Analysis
1. **Primary Patterns** - What are the top 3-5 root causes across these failures?
2. **Attack Vulnerability** - Which attack types are most successful and why?
3. **Risk Exposure** - Which risk areas show the most weakness?

## Severity Analysis
- Why are the most severe failures (P0, P1) happening?
- Are there common factors in critical vs moderate failures?

## Key Insights
- What does this tell us about the agent's overall robustness?
- Are there systemic issues vs isolated failures?

Be specific and actionable. Reference specific patterns from the data."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def generate_fix_recommendations(
    failures_df: pd.DataFrame,
    mapping: dict,
    api_key: str
) -> str:
    """Generate recommendations to fix vulnerabilities.

    Args:
        failures_df: DataFrame with failure data
        mapping: Column mapping dict
        api_key: Anthropic API key

    Returns:
        AI-generated recommendations text
    """
    if not api_key:
        return "API key not configured. Please add your Anthropic API key in the sidebar."

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build risk and attack column lists
        risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
        attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
        risk_cols = [c for c in risk_cols if c in failures_df.columns]
        attack_cols = [c for c in attack_cols if c in failures_df.columns]

        # Build context
        failure_summaries = []
        for _, row in failures_df.head(15).iterrows():
            summary = {
                'severity': row.get('Severity'),
                'risk': ' > '.join([str(row.get(c, '')) for c in risk_cols if c in row and pd.notna(row.get(c))]),
                'attack': ' > '.join([str(row.get(c, '')) for c in attack_cols if c in row and pd.notna(row.get(c))]),
            }
            if 'Justification' in row and pd.notna(row.get('Justification')):
                summary['justification'] = str(row.get('Justification', ''))[:200]
            failure_summaries.append(summary)

        prompt = f"""Based on these AI agent evaluation failures, provide specific recommendations to improve the agent:

{failure_summaries}

Please provide:

## Priority 1: Critical Fixes (for P0/P1 issues)
- Specific changes needed
- Expected impact

## Priority 2: Important Improvements (for P2 issues)
- Specific changes needed
- Expected impact

## Priority 3: Enhancements (for P3/P4 issues)
- Specific changes needed
- Expected impact

## Implementation Roadmap
- What to fix first?
- Estimated effort (quick win vs major change)

Be specific and actionable. Focus on practical improvements to the agent's behavior, prompts, or guardrails."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        return f"Error generating recommendations: {str(e)}"


def analyze_single_vulnerability(
    row: pd.Series,
    mapping: dict,
    api_key: str
) -> str:
    """Analyze a single vulnerability in detail.

    Args:
        row: DataFrame row with vulnerability data
        mapping: Column mapping dict
        api_key: Anthropic API key

    Returns:
        AI-generated analysis text
    """
    if not api_key:
        return "API key not configured."

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build risk and attack column lists
        risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
        attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
        risk_cols = [c for c in risk_cols if c in row.index]
        attack_cols = [c for c in attack_cols if c in row.index]

        context = f"""Analyze this specific AI agent failure:

Severity: {row.get('Severity')}
Risk Path: {' > '.join([str(row.get(c, '')) for c in risk_cols if pd.notna(row.get(c))])}
Attack Path: {' > '.join([str(row.get(c, '')) for c in attack_cols if pd.notna(row.get(c))])}
"""

        if 'Justification' in row and pd.notna(row.get('Justification')):
            context += f"\nJustification: {row.get('Justification')}"

        if 'Transcript' in row and pd.notna(row.get('Transcript')):
            transcript = str(row.get('Transcript'))[:2000]
            context += f"\nConversation:\n{transcript}"

        context += """

Please provide a brief analysis:
1. **What went wrong?** - Why did the agent fail?
2. **How to fix it?** - Specific recommendation"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": context}]
        )
        return message.content[0].text

    except Exception as e:
        return f"Error: {str(e)}"


def render_vulnerability_list(
    failures_df: pd.DataFrame,
    mapping: dict,
    api_key: str,
    key_prefix: str
) -> None:
    """Render a list of vulnerabilities with expandable details.

    Args:
        failures_df: DataFrame with failure data
        mapping: Column mapping dict
        api_key: Anthropic API key
        key_prefix: Prefix for widget keys
    """
    # Build risk and attack column lists
    risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
    attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
    risk_cols = [c for c in risk_cols if c in failures_df.columns]
    attack_cols = [c for c in attack_cols if c in failures_df.columns]

    st.markdown(f"**Showing {len(failures_df)} vulnerabilities:**")

    # Color coding for severity
    severity_colors = {
        'P0': '🔴', 'P1': '🟠', 'P2': '🟡', 'P3': '🔵', 'P4': '⚪'
    }
    severity_descriptions = {
        'P0': 'Critical',
        'P1': 'High',
        'P2': 'Medium',
        'P3': 'Low',
        'P4': 'Minimal'
    }

    for idx, (row_idx, row) in enumerate(failures_df.iterrows()):
        severity = row.get('Severity', 'Unknown')
        color = severity_colors.get(severity, '⚫')

        # Build title
        risk_path = ' > '.join([str(row.get(c, '')) for c in risk_cols if pd.notna(row.get(c))])
        attack_path = ' > '.join([str(row.get(c, '')) for c in attack_cols if pd.notna(row.get(c))])

        # Truncate paths for title
        risk_display = risk_path[:50] + '...' if len(risk_path) > 50 else risk_path
        attack_display = attack_path[:50] + '...' if len(attack_path) > 50 else attack_path

        title = f"{color} **{severity}** | Risk: {risk_display} | Attack: {attack_display}"

        with st.expander(title, expanded=idx < 3):
            col1, col2 = st.columns([3, 1])

            with col1:
                # Risk path
                if risk_cols:
                    st.markdown("**Risk Category:**")
                    for i, col in enumerate(risk_cols):
                        if col in row and pd.notna(row[col]):
                            indent = "&nbsp;" * (i * 4)
                            prefix = "└─ " if i > 0 else ""
                            st.markdown(f"{indent}{prefix}{row[col]}", unsafe_allow_html=True)

                # Attack path
                if attack_cols:
                    st.markdown("**Attack Type:**")
                    for i, col in enumerate(attack_cols):
                        if col in row and pd.notna(row[col]):
                            indent = "&nbsp;" * (i * 4)
                            prefix = "└─ " if i > 0 else ""
                            st.markdown(f"{indent}{prefix}{row[col]}", unsafe_allow_html=True)

                # Justification
                if 'Justification' in row and pd.notna(row.get('Justification')):
                    st.markdown("**Justification:**")
                    st.write(row['Justification'])

                # Transcript (truncated)
                if 'Transcript' in row and pd.notna(row.get('Transcript')):
                    st.markdown("**Conversation Transcript:**")
                    transcript = str(row['Transcript'])
                    if len(transcript) > 500:
                        st.text(transcript[:500] + "...")
                        with st.popover("Show Full Transcript"):
                            st.text(transcript)
                    else:
                        st.text(transcript)

            with col2:
                st.markdown("**Severity**")
                st.write(f"{color} {severity}")
                st.caption(severity_descriptions.get(severity, 'Unknown'))

                # AI analyze button for this specific failure
                if api_key:
                    if st.button("Analyze", key=f"{key_prefix}_analyze_{idx}"):
                        with st.spinner("Analyzing..."):
                            analysis = analyze_single_vulnerability(row, mapping, api_key)
                            st.session_state[f'{key_prefix}_analysis_{idx}'] = analysis

                # Show stored analysis if exists
                if f'{key_prefix}_analysis_{idx}' in st.session_state:
                    st.markdown("---")
                    st.markdown(st.session_state[f'{key_prefix}_analysis_{idx}'])


def render_top_vulnerabilities(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict, api_key: str) -> None:
    """View 3: Deep dive on most severe failures.

    Args:
        df1: Round 1 dataframe (already mapped)
        df2: Round 2 dataframe (or None, already mapped)
        mapping: Column mapping dict
        api_key: Anthropic API key
    """
    st.header("Top Vulnerabilities")
    st.markdown("Analyze the most severe failures in detail.")

    if 'Severity' not in df1.columns:
        st.error("Severity column not found in data. Please check your column mapping.")
        return

    # Settings
    with st.expander("Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            top_n = st.selectbox(
                "Show top N vulnerabilities",
                options=[5, 10, 25, 50],
                index=1,
                key="vuln_top_n"
            )

        with col2:
            severity_options = ['P0', 'P1', 'P2', 'P3', 'P4']
            available_severities = [s for s in severity_options if s in df1['Severity'].unique()]
            default_severities = [s for s in ['P0', 'P1', 'P2'] if s in available_severities]
            if not default_severities and available_severities:
                default_severities = available_severities[:2]

            severity_filter = st.multiselect(
                "Severity levels to include",
                options=available_severities,
                default=default_severities,
                key="vuln_severity"
            )

        with col3:
            sort_options = ['Severity (worst first)']
            risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
            attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
            risk_cols = [c for c in risk_cols if c in df1.columns]
            attack_cols = [c for c in attack_cols if c in df1.columns]

            if risk_cols:
                sort_options.append('Risk Category')
            if attack_cols:
                sort_options.append('Attack Type')

            sort_by = st.selectbox(
                "Sort by",
                options=sort_options,
                key="vuln_sort"
            )

    # Filter to failures
    if not severity_filter:
        st.warning("Please select at least one severity level.")
        return

    failures1 = df1[df1['Severity'].isin(severity_filter)].copy()

    if len(failures1) == 0:
        st.success("No failures matching the selected criteria!")
        return

    # Add severity rank for sorting
    severity_order = {'P0': 0, 'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}
    failures1['severity_rank'] = failures1['Severity'].map(severity_order)

    # Sort
    if sort_by == 'Severity (worst first)':
        failures1 = failures1.sort_values('severity_rank')
    elif sort_by == 'Risk Category' and risk_cols:
        failures1 = failures1.sort_values([risk_cols[0], 'severity_rank'])
    elif sort_by == 'Attack Type' and attack_cols:
        failures1 = failures1.sort_values([attack_cols[0], 'severity_rank'])

    # Take top N
    top_failures = failures1.head(top_n)

    # Comparison mode
    if df2 is not None and 'Severity' in df2.columns:
        failures2 = df2[df2['Severity'].isin(severity_filter)].copy()
        failures2['severity_rank'] = failures2['Severity'].map(severity_order)

        # Show comparison summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Round 1 Failures", len(failures1))
        with col2:
            delta = len(failures2) - len(failures1)
            st.metric("Round 2 Failures", len(failures2), delta=f"{delta:+d}", delta_color="inverse")
        with col3:
            # Critical failures comparison
            critical1 = failures1['Severity'].isin(['P0', 'P1']).sum()
            critical2 = failures2['Severity'].isin(['P0', 'P1']).sum()
            delta_critical = critical2 - critical1
            st.metric("Critical (P0/P1)", f"{critical2}", delta=f"{delta_critical:+d}", delta_color="inverse")

        # Tabs for each round
        tab1, tab2 = st.tabs(["Round 1 Vulnerabilities", "Round 2 Vulnerabilities"])

        with tab1:
            render_vulnerability_list(top_failures, mapping, api_key, "r1")

        with tab2:
            top_failures2 = failures2.sort_values('severity_rank').head(top_n)
            render_vulnerability_list(top_failures2, mapping, api_key, "r2")

    else:
        # Single round summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Failures", len(failures1))
        with col2:
            critical = failures1['Severity'].isin(['P0', 'P1']).sum()
            st.metric("Critical (P0/P1)", critical)
        with col3:
            pct = len(failures1) / len(df1) * 100 if len(df1) > 0 else 0
            st.metric("Failure Rate", f"{pct:.1f}%")

        render_vulnerability_list(top_failures, mapping, api_key, "single")

    # AI Analysis section
    st.divider()
    st.subheader("AI Vulnerability Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Analyze Root Causes", type="primary", key="ai_root_cause"):
            with st.spinner("Analyzing vulnerabilities..."):
                analysis = analyze_vulnerability_root_causes(top_failures, mapping, api_key)
                st.session_state['vuln_root_cause'] = analysis

        if 'vuln_root_cause' in st.session_state:
            st.markdown(st.session_state['vuln_root_cause'])

    with col2:
        if st.button("Generate Fix Recommendations", type="secondary", key="ai_fixes"):
            with st.spinner("Generating recommendations..."):
                recommendations = generate_fix_recommendations(top_failures, mapping, api_key)
                st.session_state['vuln_recommendations'] = recommendations

        if 'vuln_recommendations' in st.session_state:
            st.markdown(st.session_state['vuln_recommendations'])


def _build_taxonomy_table(df: pd.DataFrame, l1_col: str, l2_col: str) -> pd.DataFrame:
    """Build a taxonomy table with L1, L2, and empty Descriptor columns.

    Args:
        df: Input dataframe
        l1_col: Column name for L1 categories
        l2_col: Column name for L2 categories

    Returns:
        DataFrame with L1, L2, Descriptor columns
    """
    if l1_col not in df.columns:
        return pd.DataFrame(columns=['L1', 'L2', 'Descriptor'])

    if l2_col not in df.columns:
        # Only L1 available
        l1_values = sorted(df[l1_col].dropna().unique())
        return pd.DataFrame({
            'L1': l1_values,
            'L2': [''] * len(l1_values),
            'Descriptor': [''] * len(l1_values),
        })

    # Build L1/L2 pairs
    rows = []
    l1_values = sorted(df[l1_col].dropna().unique())
    for l1 in l1_values:
        l2_values = sorted(df[df[l1_col] == l1][l2_col].dropna().unique())
        if not l2_values:
            rows.append({'L1': l1, 'L2': '', 'Descriptor': ''})
        else:
            for l2 in l2_values:
                rows.append({'L1': l1, 'L2': l2, 'Descriptor': ''})

    return pd.DataFrame(rows)


def _generate_taxonomy_descriptions(
    taxonomy_df: pd.DataFrame,
    taxonomy_type: str,
    api_key: str
) -> pd.DataFrame:
    """Generate AI descriptions for taxonomy entries.

    Args:
        taxonomy_df: DataFrame with L1, L2, Descriptor columns
        taxonomy_type: "Risk" or "Attack"
        api_key: Anthropic API key

    Returns:
        DataFrame with Descriptor column filled in

    Raises:
        ValueError: If API key is missing
        Exception: Re-raises API errors for caller to handle
    """
    if not api_key:
        raise ValueError("API key required")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Group entries by L1 for structured context
    l1_groups = {}
    for _, row in taxonomy_df.iterrows():
        l1 = row['L1']
        l2 = row['L2'] if row['L2'] else None
        if l1 not in l1_groups:
            l1_groups[l1] = []
        if l2:
            l1_groups[l1].append(l2)

    # Build structured taxonomy context
    taxonomy_context = ""
    for l1, l2_list in l1_groups.items():
        if l2_list:
            taxonomy_context += f"\n{l1}:\n"
            for l2 in l2_list:
                taxonomy_context += f"  - {l2}\n"
        else:
            taxonomy_context += f"\n{l1} (top-level category, no subcategories)\n"

    # Build the ordered list of entries for response alignment
    entries = []
    for _, row in taxonomy_df.iterrows():
        if row['L2']:
            entries.append(f"{row['L1']} > {row['L2']}")
        else:
            entries.append(f"{row['L1']}")

    entries_numbered = "\n".join(f"{i+1}. {e}" for i, e in enumerate(entries))

    if taxonomy_type == 'Risk':
        context_desc = "risks posed by AI agents during safety evaluations"
        example = "e.g., 'Scenarios where the AI provides harmful medical advice without appropriate disclaimers.'"
    else:
        context_desc = "attack techniques used to test AI agent safety boundaries"
        example = "e.g., 'Attempts to extract training data or system prompts through crafted queries.'"

    prompt = f"""You are writing descriptors for an AI safety evaluation taxonomy used in audit reports.

Context: These are {taxonomy_type.lower()} categories used to classify {context_desc}.

Taxonomy structure:
{taxonomy_context}

For each entry below, write a brief 1-2 sentence descriptor (max 25 words) explaining what this category covers in the context of AI safety evaluation. Be specific and professional — these will appear in a formal audit report.

Example format: {example}

Write exactly one descriptor per line, matching the order below. No numbering, no bullet points, no prefixes — just the descriptor text.

Entries:
{entries_numbered}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    descriptions = response.content[0].text.strip().split('\n')
    # Clean up: remove numbering, bullets, dashes, quotes
    cleaned = []
    for d in descriptions:
        d = d.strip()
        if not d:
            continue
        # Strip leading number+period (e.g., "1. ")
        import re
        d = re.sub(r'^\d+\.\s*', '', d)
        d = d.lstrip('- ').lstrip('•').lstrip('"').rstrip('"').strip()
        if d:
            cleaned.append(d)

    result = taxonomy_df.copy()
    for i in range(min(len(cleaned), len(result))):
        result.iloc[i, result.columns.get_loc('Descriptor')] = cleaned[i]

    return result


def render_audit_report_assets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    mapping: dict,
    api_key: str
) -> None:
    """Render audit report assets for export.

    Args:
        df1: Round 1 dataframe
        df2: Round 2 dataframe (or None)
        mapping: Column mapping dict
        api_key: Anthropic API key
    """
    # Canonical Attack L1 sort order
    # Note: "Social Engineering" and "Manipulative" are interchangeable (customer-dependent naming)
    ATTACK_L1_ORDER = ["Benign", "Social Engineering", "Adversarial"]

    def sort_by_attack_order(df: pd.DataFrame, column: str = 'Attack L1') -> pd.DataFrame:
        """Sort DataFrame by canonical Attack L1 order (case-insensitive)."""
        if column not in df.columns and 'L1' in df.columns:
            column = 'L1'
        if column not in df.columns:
            return df

        # Create case-insensitive order map
        # "Manipulative" is an alias for "Social Engineering" (same sort position)
        order_map = {v.lower(): i for i, v in enumerate(ATTACK_L1_ORDER)}
        order_map['manipulative'] = order_map['social engineering']  # Alias
        df = df.copy()
        df['_sort_order'] = df[column].str.lower().str.strip().map(order_map).fillna(999)
        # Secondary sort: alphabetical for items not in order list
        df['_sort_alpha'] = df[column].str.lower().str.strip()
        df = df.sort_values(['_sort_order', '_sort_alpha']).drop(columns=['_sort_order', '_sort_alpha'])
        return df.reset_index(drop=True)

    def _build_summary_table(df: pd.DataFrame, group_col: str, use_attack_order: bool = False) -> pd.DataFrame:
        """Build a summary table grouped by a column with severity breakdown."""
        if group_col not in df.columns or 'Severity' not in df.columns:
            return pd.DataFrame()

        stats = df.groupby(group_col).agg(
            Count=('Severity', 'count'),
            Pass=('Severity', lambda x: (x == 'PASS').sum()),
            P4=('Severity', lambda x: (x == 'P4').sum()),
            P3=('Severity', lambda x: (x == 'P3').sum()),
            P2=('Severity', lambda x: (x == 'P2').sum()),
            P1=('Severity', lambda x: (x == 'P1').sum()),
            P0=('Severity', lambda x: (x == 'P0').sum()),
        ).reset_index()

        stats = stats.rename(columns={group_col: 'Category'})
        stats['Pass %'] = (stats['Pass'] / stats['Count'] * 100).round(1)

        if use_attack_order:
            # Apply canonical Attack L1 sort order
            stats = sort_by_attack_order(stats, 'Category')
        else:
            # Default: sort by Pass % ascending
            stats = stats.sort_values('Pass %', ascending=True).reset_index(drop=True)

        # Format Pass % as string
        display = stats[['Category', 'Count', 'Pass %', 'P4', 'P3', 'P2', 'P1', 'P0']].copy()
        display['Pass %'] = display['Pass %'].apply(lambda x: f"{x:.1f}%")

        return display

    st.header("Audit Report Assets")

    # AIUC-1 branding banner
    st.markdown("""
    <div class="audit-banner">
        <div class="audit-badge">AIUC-1 AUDIT ASSETS</div>
        <p>Generate publication-ready tables for your audit report. All tables are formatted for easy copy-paste into Google Docs or Word.</p>
    </div>
    """, unsafe_allow_html=True)

    # Comparison mode: choose which round to use (default to most recent)
    if df2 is not None:
        audit_round = st.radio(
            "Data source for audit assets:",
            options=["Round 1", "Round 2"],
            index=1,  # Default to Round 2 (most recent)
            horizontal=True,
            key="audit_round_selector"
        )
        active_df = df2 if audit_round == "Round 2" else df1

        # Clear cached tables when round changes
        prev_round = st.session_state.get('_audit_prev_round')
        if prev_round is not None and prev_round != audit_round:
            for key in ['audit_risk_descriptors', 'audit_attack_descriptors',
                        'audit_risk_ai_descriptors', 'audit_attack_ai_descriptors',
                        'audit_example_evals', 'audit_examples']:
                st.session_state.pop(key, None)
        st.session_state['_audit_prev_round'] = audit_round
    else:
        active_df = df1

    # --- Section 1: Risk Taxonomy ---
    st.subheader("Risk Taxonomy")

    risk_cols = [f'Risk L{i+1}' for i in range(len(mapping.get('risk_hierarchy', [])))]
    risk_cols = [c for c in risk_cols if c in active_df.columns]

    if len(risk_cols) >= 1:
        l1_col = risk_cols[0]
        l2_col = risk_cols[1] if len(risk_cols) > 1 else None

        # Initialize session state for risk descriptors
        if 'audit_risk_descriptors' not in st.session_state:
            st.session_state['audit_risk_descriptors'] = _build_taxonomy_table(
                active_df, l1_col, l2_col if l2_col else l1_col
            )

        # Action buttons
        col_ai, col_reset, col_spacer = st.columns([1, 1, 2])
        with col_ai:
            if st.button("Generate AI Descriptions", key="gen_risk_desc"):
                if not api_key:
                    st.warning("Enter an Anthropic API key in the sidebar to use AI features.")
                else:
                    with st.spinner("Generating risk taxonomy descriptions..."):
                        try:
                            updated = _generate_taxonomy_descriptions(
                                st.session_state['audit_risk_descriptors'],
                                "Risk",
                                api_key
                            )
                            st.session_state['audit_risk_descriptors'] = updated
                            st.session_state['audit_risk_ai_descriptors'] = updated.copy()
                            st.rerun()
                        except ValueError as e:
                            st.warning(str(e))
                        except Exception as e:
                            st.error(f"AI generation failed: {str(e)}")
        with col_reset:
            if st.button("Reset to Defaults", key="reset_risk_desc"):
                st.session_state['audit_risk_descriptors'] = _build_taxonomy_table(
                    active_df, l1_col, l2_col if l2_col else l1_col
                )
                if 'audit_risk_ai_descriptors' in st.session_state:
                    del st.session_state['audit_risk_ai_descriptors']
                st.rerun()

        # Show if AI descriptors were previously generated
        if 'audit_risk_ai_descriptors' in st.session_state:
            st.caption("AI-generated descriptions loaded. Edit below before copying.")

        # Editable table
        edited_risk = st.data_editor(
            st.session_state['audit_risk_descriptors'],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                'L1': st.column_config.TextColumn('L1', disabled=True),
                'L2': st.column_config.TextColumn('L2', disabled=True),
                'Descriptor': st.column_config.TextColumn('Descriptor', width='large'),
            },
            key="risk_taxonomy_editor"
        )
        st.session_state['audit_risk_descriptors'] = edited_risk

        # Copy button
        if st.button("Copy to Clipboard", key="copy_risk_taxonomy"):
            clipboard_data = df_to_clipboard_format(edited_risk)
            st.code(clipboard_data, language=None)
            st.success("Table ready to copy (Ctrl+C / Cmd+C)")
    else:
        st.warning("Map risk columns to see the risk taxonomy.")

    st.divider()

    # --- Section 2: Attack Taxonomy ---
    st.subheader("Attack Taxonomy")

    attack_cols = [f'Attack L{i+1}' for i in range(len(mapping.get('attack_hierarchy', [])))]
    attack_cols = [c for c in attack_cols if c in active_df.columns]

    if len(attack_cols) >= 1:
        l1_col = attack_cols[0]
        l2_col = attack_cols[1] if len(attack_cols) > 1 else None

        # Initialize session state for attack descriptors (sorted by canonical order)
        if 'audit_attack_descriptors' not in st.session_state:
            attack_taxonomy = _build_taxonomy_table(
                active_df, l1_col, l2_col if l2_col else l1_col
            )
            st.session_state['audit_attack_descriptors'] = sort_by_attack_order(attack_taxonomy, 'L1')

        # Action buttons
        col_ai, col_reset, col_spacer = st.columns([1, 1, 2])
        with col_ai:
            if st.button("Generate AI Descriptions", key="gen_attack_desc"):
                if not api_key:
                    st.warning("Enter an Anthropic API key in the sidebar to use AI features.")
                else:
                    with st.spinner("Generating attack taxonomy descriptions..."):
                        try:
                            updated = _generate_taxonomy_descriptions(
                                st.session_state['audit_attack_descriptors'],
                                "Attack",
                                api_key
                            )
                            st.session_state['audit_attack_descriptors'] = updated
                            st.session_state['audit_attack_ai_descriptors'] = updated.copy()
                            st.rerun()
                        except ValueError as e:
                            st.warning(str(e))
                        except Exception as e:
                            st.error(f"AI generation failed: {str(e)}")
        with col_reset:
            if st.button("Reset to Defaults", key="reset_attack_desc"):
                attack_taxonomy = _build_taxonomy_table(
                    active_df, l1_col, l2_col if l2_col else l1_col
                )
                st.session_state['audit_attack_descriptors'] = sort_by_attack_order(attack_taxonomy, 'L1')
                if 'audit_attack_ai_descriptors' in st.session_state:
                    del st.session_state['audit_attack_ai_descriptors']
                st.rerun()

        # Show if AI descriptors were previously generated
        if 'audit_attack_ai_descriptors' in st.session_state:
            st.caption("AI-generated descriptions loaded. Edit below before copying.")

        # Editable table
        edited_attack = st.data_editor(
            st.session_state['audit_attack_descriptors'],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                'L1': st.column_config.TextColumn('L1', disabled=True),
                'L2': st.column_config.TextColumn('L2', disabled=True),
                'Descriptor': st.column_config.TextColumn('Descriptor', width='large'),
            },
            key="attack_taxonomy_editor"
        )
        st.session_state['audit_attack_descriptors'] = edited_attack

        # Copy button
        if st.button("Copy to Clipboard", key="copy_attack_taxonomy"):
            clipboard_data = df_to_clipboard_format(edited_attack)
            st.code(clipboard_data, language=None)
            st.success("Table ready to copy (Ctrl+C / Cmd+C)")
    else:
        st.warning("Map attack columns to see the attack taxonomy.")

    st.divider()

    # --- Section 3: Example Evaluations ---
    st.subheader("Example Evaluations")

    # Initialize resample counter
    if 'audit_resample_seed' not in st.session_state:
        st.session_state['audit_resample_seed'] = 42

    num_examples = st.number_input(
        "Number of examples", min_value=1, max_value=50, value=10, key="audit_num_examples"
    )

    # Determine prompt column: prefer Transcript, fallback to Justification
    prompt_source_col = None
    if 'Transcript' in active_df.columns:
        prompt_source_col = 'Transcript'
    elif 'Justification' in active_df.columns:
        prompt_source_col = 'Justification'

    def _extract_prompt(text):
        """Extract first user message from transcript, or return text as-is."""
        if pd.isna(text) or not str(text).strip():
            return ''
        text = str(text)
        # Try to extract first user message from common transcript formats
        # Format: "User: ...\nAssistant: ..." or "human: ...\nassistant: ..."
        import re
        # Look for first user/human turn
        patterns = [
            r'(?:^|\n)\s*(?:User|Human|user|human)\s*[:\-]\s*(.*?)(?:\n\s*(?:Assistant|AI|assistant|ai|System|system)\s*[:\-]|$)',
            r'(?:^|\n)\s*(?:role\s*[:\-]\s*user.*?content\s*[:\-]\s*)(.*?)(?:\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted[:500]  # Limit length for display

        # No pattern matched — return first 500 chars as-is
        return text[:500]

    # Stratified sampling across Risk L1 categories
    def _stratified_sample(df, n, seed):
        """Sample n rows stratified by Risk L1 if available."""
        n = min(n, len(df))
        if n == 0:
            return pd.DataFrame()

        if 'Risk L1' in df.columns:
            categories = df['Risk L1'].dropna().unique()
            if len(categories) > 1:
                # Allocate samples proportionally, at least 1 per category
                per_category = max(1, n // len(categories))
                remainder = n - (per_category * len(categories))

                samples = []
                for cat in categories:
                    cat_df = df[df['Risk L1'] == cat]
                    cat_n = min(per_category, len(cat_df))
                    if cat_n > 0:
                        samples.append(cat_df.sample(n=cat_n, random_state=seed))

                result = pd.concat(samples, ignore_index=True)

                # Fill remainder from full dataframe if needed
                if len(result) < n:
                    remaining_idx = df.index.difference(result.index)
                    extra_n = min(n - len(result), len(remaining_idx))
                    if extra_n > 0:
                        extra = df.loc[remaining_idx].sample(n=extra_n, random_state=seed + 1)
                        result = pd.concat([result, extra], ignore_index=True)

                return result.head(n)

        # Fallback: simple random sample
        return df.sample(n=n, random_state=seed)

    # Sample button row
    col_resample, col_spacer = st.columns([1, 3])
    with col_resample:
        if st.button("Resample", key="audit_resample"):
            st.session_state['audit_resample_seed'] += 1
            if 'audit_example_evals' in st.session_state:
                del st.session_state['audit_example_evals']
            st.rerun()

    # Generate or retrieve examples
    current_seed = st.session_state['audit_resample_seed']

    if 'audit_example_evals' not in st.session_state or st.session_state.get('_audit_last_n') != num_examples:
        sampled = _stratified_sample(active_df, num_examples, current_seed)

        # Build display table
        display_cols_map = {}
        for col in ['Risk L1', 'Risk L2', 'Attack L1', 'Attack L2']:
            if col in sampled.columns:
                display_cols_map[col] = col

        # Add Prompt column
        if prompt_source_col and prompt_source_col in sampled.columns:
            sampled = sampled.copy()
            sampled['Prompt'] = sampled[prompt_source_col].apply(_extract_prompt)
            display_cols_map['Prompt'] = 'Prompt'

        display_cols = list(display_cols_map.values())
        if display_cols:
            st.session_state['audit_example_evals'] = sampled[display_cols].reset_index(drop=True)
        else:
            st.session_state['audit_example_evals'] = pd.DataFrame()
        st.session_state['_audit_last_n'] = num_examples

    examples_display = st.session_state['audit_example_evals']

    if not examples_display.empty:
        st.dataframe(examples_display, use_container_width=True, hide_index=True)

        if st.button("Copy to Clipboard", key="copy_examples"):
            clipboard_data = df_to_clipboard_format(examples_display)
            st.code(clipboard_data, language=None)
            st.success("Table ready to copy (Ctrl+C / Cmd+C)")
    else:
        st.info("No data available for sampling.")

    st.divider()

    # --- Section 4: Results Summary Tables ---
    st.subheader("Results Summary Tables")

    if 'Severity' not in active_df.columns:
        st.warning("Severity column required for results summary.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**Results by Attack L1**")
            if 'Attack L1' in active_df.columns:
                attack_summary = _build_summary_table(active_df, 'Attack L1', use_attack_order=True)
                if not attack_summary.empty:
                    st.dataframe(attack_summary, use_container_width=True, hide_index=True)

                    if st.button("Copy to Clipboard", key="copy_attack_summary"):
                        clipboard_data = df_to_clipboard_format(attack_summary)
                        st.code(clipboard_data, language=None)
                        st.success("Table ready to copy (Ctrl+C / Cmd+C)")
                else:
                    st.info("No data available.")
            else:
                st.warning("Attack L1 column not mapped.")

        with col_right:
            st.markdown("**Results by Risk L2**")
            if 'Risk L2' in active_df.columns:
                risk_summary = _build_summary_table(active_df, 'Risk L2')
                if not risk_summary.empty:
                    st.dataframe(risk_summary, use_container_width=True, hide_index=True)

                    if st.button("Copy to Clipboard", key="copy_risk_summary"):
                        clipboard_data = df_to_clipboard_format(risk_summary)
                        st.code(clipboard_data, language=None)
                        st.success("Table ready to copy (Ctrl+C / Cmd+C)")
                else:
                    st.info("No data available.")
            else:
                st.warning("Risk L2 column not mapped.")

    # --- Section 5: Download All as CSV ---
    st.markdown("---")
    st.markdown("### 📥 Export All Audit Assets")

    # Build combined CSV with section headers
    csv_parts = []

    # Risk Taxonomy
    risk_key = 'audit_risk_taxonomy'
    if risk_key in st.session_state and st.session_state[risk_key] is not None:
        csv_parts.append("# SECTION: Risk Taxonomy")
        csv_parts.append(st.session_state[risk_key].to_csv(index=False))

    # Attack Taxonomy
    attack_key = 'audit_attack_taxonomy'
    if attack_key in st.session_state and st.session_state[attack_key] is not None:
        csv_parts.append("# SECTION: Attack Taxonomy")
        csv_parts.append(st.session_state[attack_key].to_csv(index=False))

    # Example Evaluations
    examples_key = 'audit_examples'
    if examples_key in st.session_state and st.session_state[examples_key] is not None:
        csv_parts.append("# SECTION: Example Evaluations")
        csv_parts.append(st.session_state[examples_key].to_csv(index=False))

    # Summary Tables
    if 'Attack L1' in active_df.columns:
        attack_summary = _build_summary_table(active_df, 'Attack L1', use_attack_order=True)
        if not attack_summary.empty:
            csv_parts.append("# SECTION: Results by Attack L1")
            csv_parts.append(attack_summary.to_csv(index=False))

    if 'Risk L2' in active_df.columns:
        risk_summary = _build_summary_table(active_df, 'Risk L2')
        if not risk_summary.empty:
            csv_parts.append("# SECTION: Results by Risk L2")
            csv_parts.append(risk_summary.to_csv(index=False))

    if csv_parts:
        combined_csv = "\n".join(csv_parts)
        st.download_button(
            label="📥 Download All as CSV",
            data=combined_csv,
            file_name="audit_report_assets.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No data available to export. Generate tables above first.")


def main() -> None:
    """Main application entry point."""
    # Check authentication if enabled
    if not check_auth():
        return

    # Sidebar with AIUC-1 branding
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; padding: 1rem 0; margin-bottom: 1rem; border-bottom: 1px solid #2a2a2a;">
            <img src="https://www.aiuc-1.com/brand/aiuc1.svg" alt="AIUC-1" style="height: 32px;" onerror="this.style.display='none'">
            <div>
                <div style="font-weight: 600; font-size: 1rem;">Eval Dashboard</div>
                <div style="font-size: 0.75rem; color: #666;">AI Agent Security Testing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show authenticated user if auth is enabled
        if settings.ENABLE_AUTH and st.session_state.get("authenticated_email"):
            st.caption(f"Signed in as: {st.session_state['authenticated_email']}")
            if st.button("Sign Out", type="secondary"):
                del st.session_state["authenticated_email"]
                st.rerun()

        # File uploads section
        st.markdown("### 📁 Data")

        file1 = st.file_uploader("Round 1 CSV", type=['csv'], key="round1_file")
        file2 = st.file_uploader("Round 2 CSV (optional)", type=['csv'], key="round2_file")

        # Show upload status
        if file1:
            st.success("✅ Round 1 loaded")
        if file2:
            st.success("✅ Round 2 loaded")
            st.session_state['comparison_mode'] = True
        else:
            st.session_state['comparison_mode'] = False

        st.divider()

        # API Key section
        st.markdown("### 🔑 API Key")
        api_key = render_api_key_section()

        st.divider()

        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        if st.button("🤖 Ask AI a Question", use_container_width=True):
            st.session_state['show_ai_chat'] = True

        # Debug info
        if settings.DEBUG:
            st.divider()
            with st.expander("Debug Info"):
                st.json({
                    "auth_enabled": settings.ENABLE_AUTH,
                    "api_key_configured": settings.has_api_key,
                    "api_key_source": settings.api_key_source,
                    "comparison_mode": st.session_state.get('comparison_mode', False),
                })

    # Main content area
    if not file1:
        render_welcome_screen()
        return

    # Load data
    df1 = load_data(file1)
    df2 = load_data(file2) if file2 else None

    if df1 is None:
        st.error("Failed to load Round 1 data.")
        return

    # Column mapping (collapsed after first setup)
    mapping_valid = st.session_state.get('mapping_validated', False)

    with st.expander("⚙️ Column Mapping", expanded=not mapping_valid):
        mapping, is_valid = render_column_mapper(df1)
        if is_valid:
            st.session_state.mapping_validated = True
            st.session_state.column_mapping = mapping

    if not is_valid:
        st.warning("Please complete column mapping to continue")
        return

    mapping = st.session_state.column_mapping

    # Apply mapping to dataframes
    df1_mapped = apply_mapping_to_df(df1, mapping)
    df2_mapped = apply_mapping_to_df(df2, mapping) if df2 is not None else None

    # Comparison mode banner
    if df2_mapped is not None:
        st.info("📊 **Comparison Mode Active** - Showing Round 1 vs Round 2")

    # Main 4-tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "Evals Heatmap",
        "Results Statistics",
        "Top Vulnerabilities",
        "Audit Report Assets"
    ])

    with tab1:
        safe_render(render_evals_heatmap, df1_mapped, df2_mapped, mapping, api_key)

    with tab2:
        safe_render(render_results_statistics, df1_mapped, df2_mapped, mapping, api_key)

    with tab3:
        safe_render(render_top_vulnerabilities, df1_mapped, df2_mapped, mapping, api_key)

    with tab4:
        safe_render(render_audit_report_assets, df1_mapped, df2_mapped, mapping, api_key)

    # Floating AI Chat (if activated)
    if st.session_state.get('show_ai_chat', False):
        safe_render(render_ai_chat_panel, df1_mapped, df2_mapped, mapping, api_key)


if __name__ == "__main__":
    main()
