"""Heatmap generation utilities for the eval dashboard."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Optional


def create_heatmap_data(
    df: pd.DataFrame,
    risk_col: str,
    attack_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a cross-tabulation of risk vs attack for heatmap.

    Args:
        df: Input dataframe
        risk_col: Column name for risk categories (rows)
        attack_col: Column name for attack types (columns)

    Returns:
        Tuple of (counts crosstab, percentages crosstab)
    """
    # Create cross-tabulation with margins
    crosstab = pd.crosstab(
        df[risk_col],
        df[attack_col],
        margins=True,
        margins_name='Total'
    )

    # Calculate percentages
    total = len(df)
    crosstab_pct = (crosstab / total * 100).round(2)

    return crosstab, crosstab_pct


def create_pass_rate_heatmap_data(
    df: pd.DataFrame,
    risk_col: str,
    attack_col: str,
    severity_col: str = 'Severity'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create pass rate cross-tabulation for heatmap.

    Args:
        df: Input dataframe
        risk_col: Column name for risk categories (rows)
        attack_col: Column name for attack types (columns)
        severity_col: Column name for severity/pass status

    Returns:
        Tuple of (pass rate matrix, count matrix)
    """
    # Calculate pass rate for each combination
    pass_rate = pd.crosstab(
        df[risk_col],
        df[attack_col],
        values=df[severity_col].apply(lambda x: 1 if x == 'PASS' else 0),
        aggfunc='mean'
    ) * 100

    # Calculate counts for each combination
    counts = pd.crosstab(df[risk_col], df[attack_col])

    return pass_rate.round(1), counts


def create_heatmap_figure(
    data: pd.DataFrame,
    title: str = "Eval Distribution Heatmap",
    color_scale: str = 'RdYlGn_r',
    value_format: str = '.1f',
    color_label: str = "% of Evals",
    show_text: bool = True,
    height: int = 500
) -> go.Figure:
    """Create a Plotly heatmap figure.

    Args:
        data: DataFrame with values for heatmap
        title: Chart title
        color_scale: Plotly color scale name
        value_format: Format string for annotations
        color_label: Label for color bar
        show_text: Whether to show text annotations
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    # Remove Total row/column if present
    plot_data = data.copy()
    if 'Total' in plot_data.index:
        plot_data = plot_data.drop('Total', axis=0)
    if 'Total' in plot_data.columns:
        plot_data = plot_data.drop('Total', axis=1)

    fig = px.imshow(
        plot_data.values,
        labels=dict(x="Attack Type", y="Risk Category", color=color_label),
        x=plot_data.columns.tolist(),
        y=plot_data.index.tolist(),
        color_continuous_scale=color_scale,
        aspect='auto',
        text_auto=value_format if show_text else False
    )

    fig.update_layout(
        title=title,
        xaxis_title="Attack Type",
        yaxis_title="Risk Category",
        height=height,
        xaxis={'tickangle': 45}
    )

    return fig


def create_pass_rate_heatmap_figure(
    pass_rate: pd.DataFrame,
    counts: pd.DataFrame,
    title: str = "Pass Rate Heatmap",
    height: int = 500
) -> go.Figure:
    """Create a pass rate heatmap with count annotations.

    Args:
        pass_rate: DataFrame with pass rates (0-100)
        counts: DataFrame with counts
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    # Create custom text showing both pass rate and count
    text_matrix = []
    for i, row in enumerate(pass_rate.index):
        text_row = []
        for j, col in enumerate(pass_rate.columns):
            rate = pass_rate.loc[row, col]
            count = counts.loc[row, col] if row in counts.index and col in counts.columns else 0
            if pd.notna(rate) and count > 0:
                text_row.append(f"{rate:.1f}%<br>({count})")
            else:
                text_row.append("")
        text_matrix.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=pass_rate.values,
        x=pass_rate.columns.tolist(),
        y=pass_rate.index.tolist(),
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale='RdYlGn',
        colorbar=dict(title="Pass Rate %"),
        hovertemplate="Risk: %{y}<br>Attack: %{x}<br>Pass Rate: %{z:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Attack Type",
        yaxis_title="Risk Category",
        height=height,
        xaxis={'tickangle': 45}
    )

    return fig


def create_comparison_heatmap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    risk_col: str,
    attack_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create aligned heatmap data for comparison between two rounds.

    Args:
        df1: First dataframe (Round 1)
        df2: Second dataframe (Round 2)
        risk_col: Column name for risk categories
        attack_col: Column name for attack types

    Returns:
        Tuple of (pct1_aligned, pct2_aligned, difference)
    """
    _, pct1 = create_heatmap_data(df1, risk_col, attack_col)
    _, pct2 = create_heatmap_data(df2, risk_col, attack_col)

    # Get all unique values
    all_risks = sorted(set(pct1.index) | set(pct2.index) - {'Total'})
    all_attacks = sorted(set(pct1.columns) | set(pct2.columns) - {'Total'})

    # Align and fill missing with 0
    pct1_aligned = pct1.reindex(index=all_risks, columns=all_attacks, fill_value=0)
    pct2_aligned = pct2.reindex(index=all_risks, columns=all_attacks, fill_value=0)

    # Calculate difference
    diff = pct2_aligned - pct1_aligned

    return pct1_aligned, pct2_aligned, diff


def create_pass_rate_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    risk_col: str,
    attack_col: str,
    severity_col: str = 'Severity'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create pass rate comparison between two rounds.

    Args:
        df1: First dataframe (Round 1)
        df2: Second dataframe (Round 2)
        risk_col: Column name for risk categories
        attack_col: Column name for attack types
        severity_col: Column name for severity

    Returns:
        Tuple of (rate1_aligned, rate2_aligned, difference)
    """
    rate1, _ = create_pass_rate_heatmap_data(df1, risk_col, attack_col, severity_col)
    rate2, _ = create_pass_rate_heatmap_data(df2, risk_col, attack_col, severity_col)

    # Get all unique values
    all_risks = sorted(set(rate1.index) | set(rate2.index))
    all_attacks = sorted(set(rate1.columns) | set(rate2.columns))

    # Align and fill missing with NaN (no data)
    rate1_aligned = rate1.reindex(index=all_risks, columns=all_attacks)
    rate2_aligned = rate2.reindex(index=all_risks, columns=all_attacks)

    # Calculate difference (only where both have data)
    diff = rate2_aligned - rate1_aligned

    return rate1_aligned, rate2_aligned, diff


def get_distribution_summary(
    df: pd.DataFrame,
    risk_col: str,
    attack_col: str
) -> dict:
    """Get summary statistics for the distribution.

    Args:
        df: Input dataframe
        risk_col: Column name for risk categories
        attack_col: Column name for attack types

    Returns:
        Dict with summary statistics
    """
    risk_counts = df[risk_col].value_counts()
    attack_counts = df[attack_col].value_counts()

    summary = {
        'total_evals': len(df),
        'unique_risks': df[risk_col].nunique(),
        'unique_attacks': df[attack_col].nunique(),
        'top_risk': risk_counts.index[0] if len(risk_counts) > 0 else 'N/A',
        'top_risk_count': int(risk_counts.iloc[0]) if len(risk_counts) > 0 else 0,
        'top_risk_pct': float(risk_counts.iloc[0] / len(df) * 100) if len(risk_counts) > 0 else 0,
        'top_attack': attack_counts.index[0] if len(attack_counts) > 0 else 'N/A',
        'top_attack_count': int(attack_counts.iloc[0]) if len(attack_counts) > 0 else 0,
        'top_attack_pct': float(attack_counts.iloc[0] / len(df) * 100) if len(attack_counts) > 0 else 0,
    }

    return summary


def get_coverage_gaps(
    df: pd.DataFrame,
    risk_col: str,
    attack_col: str,
    min_count: int = 5
) -> dict:
    """Identify coverage gaps in the evaluation set.

    Args:
        df: Input dataframe
        risk_col: Column name for risk categories
        attack_col: Column name for attack types
        min_count: Minimum count threshold for "adequately covered"

    Returns:
        Dict with coverage gap information
    """
    counts = pd.crosstab(df[risk_col], df[attack_col])

    # Find cells with low coverage
    low_coverage = []
    zero_coverage = []

    for risk in counts.index:
        for attack in counts.columns:
            count = counts.loc[risk, attack]
            if count == 0:
                zero_coverage.append({'risk': risk, 'attack': attack})
            elif count < min_count:
                low_coverage.append({'risk': risk, 'attack': attack, 'count': int(count)})

    # Find categories with overall low coverage
    risk_totals = df[risk_col].value_counts()
    attack_totals = df[attack_col].value_counts()

    low_risk_coverage = risk_totals[risk_totals < min_count].to_dict()
    low_attack_coverage = attack_totals[attack_totals < min_count].to_dict()

    return {
        'zero_coverage_combinations': zero_coverage[:20],  # Limit to 20
        'low_coverage_combinations': low_coverage[:20],
        'low_risk_coverage': low_risk_coverage,
        'low_attack_coverage': low_attack_coverage,
        'total_combinations': len(counts.index) * len(counts.columns),
        'covered_combinations': int((counts > 0).sum().sum()),
        'coverage_percentage': float((counts > 0).sum().sum() / (len(counts.index) * len(counts.columns)) * 100)
    }
