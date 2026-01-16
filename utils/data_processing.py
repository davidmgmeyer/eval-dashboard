"""Data processing utilities for the eval dashboard."""

import pandas as pd
from typing import Optional


# Column name variations mapping to standardized names
COLUMN_MAPPINGS = {
    'risk_l1': 'Risk L1',
    'risk l1': 'Risk L1',
    'Risk_L1': 'Risk L1',
    'risk_l2': 'Risk L2',
    'risk l2': 'Risk L2',
    'Risk_L2': 'Risk L2',
    'risk_l3': 'Risk L3',
    'risk l3': 'Risk L3',
    'Risk_L3': 'Risk L3',
    'attack_l1': 'Attack L1',
    'attack l1': 'Attack L1',
    'Attack_L1': 'Attack L1',
    'attack_l2': 'Attack L2',
    'attack l2': 'Attack L2',
    'Attack_L2': 'Attack L2',
    'attack_l3': 'Attack L3',
    'attack l3': 'Attack L3',
    'Attack_L3': 'Attack L3',
    'severity': 'Severity',
    'grade': 'Severity',
    'Grade': 'Severity',
    'justification': 'Justification',
    'reason': 'Justification',
    'Reason': 'Justification',
    'eval round': 'Eval round',
    'eval_round': 'Eval round',
    'Eval_round': 'Eval round',
    'round': 'Eval round',
    'Round': 'Eval round',
}

SEVERITY_ORDER = ['PASS', 'P4', 'P3', 'P2', 'P1', 'P0']


def load_and_clean_csv(uploaded_file, debug: bool = False) -> tuple:
    """Load CSV file and standardize column names.

    Args:
        uploaded_file: Streamlit uploaded file object or file path
        debug: If True, return debug info along with dataframe

    Returns:
        If debug=False: DataFrame with standardized column names
        If debug=True: Tuple of (DataFrame, debug_info dict)
    """
    # Use header=1 to read column names from row 2 (skip row 1)
    df = pd.read_csv(uploaded_file, header=1)

    # Store original columns for debugging
    original_columns = list(df.columns)

    # Strip whitespace from column names first
    df.columns = df.columns.str.strip()

    # Build a case-insensitive lookup for column mappings
    column_mappings_lower = {k.lower(): v for k, v in COLUMN_MAPPINGS.items()}

    # Standardize column names using multiple strategies
    new_columns = {}
    for col in df.columns:
        col_stripped = col.strip()
        col_lower = col_stripped.lower()

        # Strategy 1: Exact case-insensitive match in mappings
        if col_lower in column_mappings_lower:
            new_columns[col] = column_mappings_lower[col_lower]
            continue

        # Strategy 2: Pattern matching for Risk/Attack columns
        # Handle variations like "Risk L1", "risk_l1", "Risk_L1", "risk l1"
        col_normalized = col_lower.replace('_', ' ').replace('-', ' ')

        # Check for Risk L1, L2, L3
        if 'risk' in col_normalized:
            if 'l1' in col_normalized or 'level 1' in col_normalized or 'level1' in col_normalized:
                new_columns[col] = 'Risk L1'
            elif 'l2' in col_normalized or 'level 2' in col_normalized or 'level2' in col_normalized:
                new_columns[col] = 'Risk L2'
            elif 'l3' in col_normalized or 'level 3' in col_normalized or 'level3' in col_normalized:
                new_columns[col] = 'Risk L3'

        # Check for Attack L1, L2, L3
        elif 'attack' in col_normalized:
            if 'l1' in col_normalized or 'level 1' in col_normalized or 'level1' in col_normalized:
                new_columns[col] = 'Attack L1'
            elif 'l2' in col_normalized or 'level 2' in col_normalized or 'level2' in col_normalized:
                new_columns[col] = 'Attack L2'
            elif 'l3' in col_normalized or 'level 3' in col_normalized or 'level3' in col_normalized:
                new_columns[col] = 'Attack L3'

        # Check for Severity/Grade
        elif col_normalized in ['severity', 'grade', 'result', 'outcome']:
            new_columns[col] = 'Severity'

        # Check for Eval round
        elif 'round' in col_normalized or 'eval' in col_normalized:
            if 'round' in col_normalized:
                new_columns[col] = 'Eval round'

    if new_columns:
        df = df.rename(columns=new_columns)

    # Build debug info
    debug_info = {
        'original_columns': original_columns,
        'standardized_columns': list(df.columns),
        'column_mappings_applied': new_columns,
        'risk_columns': [c for c in df.columns if 'Risk' in c],
        'attack_columns': [c for c in df.columns if 'Attack' in c],
        'has_severity': 'Severity' in df.columns,
        'has_eval_round': 'Eval round' in df.columns,
    }

    if debug:
        return df, debug_info
    return df


def get_available_columns(df: pd.DataFrame) -> dict:
    """Get information about available columns in the dataframe.

    Args:
        df: Input dataframe

    Returns:
        Dict with column information
    """
    return {
        'all_columns': list(df.columns),
        'risk_columns': [c for c in df.columns if 'risk' in c.lower()],
        'attack_columns': [c for c in df.columns if 'attack' in c.lower()],
        'has_severity': 'Severity' in df.columns,
        'has_justification': 'Justification' in df.columns,
    }


def calculate_stats(
    df: pd.DataFrame,
    group_by: str,
    include_round: bool = False,
    round_column: str = 'Eval round'
) -> pd.DataFrame:
    """Calculate statistics grouped by a column.

    Args:
        df: Input dataframe
        group_by: Column name to group by
        include_round: If True, also group by round column
        round_column: Name of the round column (default: 'Eval round')

    Returns:
        DataFrame with counts and percentages for each severity level
    """
    if group_by not in df.columns:
        raise ValueError(f"Column '{group_by}' not found in dataframe")

    if 'Severity' not in df.columns:
        raise ValueError("Column 'Severity' not found in dataframe")

    # Determine grouping columns
    if include_round and round_column in df.columns:
        group_cols = [group_by, round_column]
    else:
        group_cols = [group_by]

    # Get counts per group and severity
    grouped = df.groupby(group_cols + ['Severity']).size().unstack(fill_value=0)

    # Ensure all severity columns exist
    for sev in SEVERITY_ORDER:
        if sev not in grouped.columns:
            grouped[sev] = 0

    # Reorder columns
    grouped = grouped[SEVERITY_ORDER]

    # Calculate total count per group
    grouped['Count'] = grouped.sum(axis=1)

    # Calculate percentages
    for sev in SEVERITY_ORDER:
        grouped[f'{sev} %'] = (grouped[sev] / grouped['Count'] * 100).round(1)

    # Reset index
    grouped = grouped.reset_index()

    # Rename columns based on whether we included round
    if include_round and round_column in df.columns:
        grouped = grouped.rename(columns={
            group_by: 'Evaluation category',
            round_column: 'Round'
        })
        # Sort by Category first, then Round (so rounds appear next to each other)
        grouped = grouped.sort_values(['Evaluation category', 'Round'], ascending=[True, False])
        grouped = grouped.reset_index(drop=True)
        # Reorder columns: Round, Category, Count, severities, percentages
        col_order = ['Round', 'Evaluation category', 'Count'] + SEVERITY_ORDER + [f'{s} %' for s in SEVERITY_ORDER]
    else:
        grouped = grouped.rename(columns={group_by: 'Category'})
        # Reorder columns: Category, Count, severities, percentages
        col_order = ['Category', 'Count'] + SEVERITY_ORDER + [f'{s} %' for s in SEVERITY_ORDER]

    grouped = grouped[col_order]

    return grouped


def get_display_stats(stats_df: pd.DataFrame, show_counts: bool = False) -> pd.DataFrame:
    """Format statistics for display.

    Args:
        stats_df: Statistics dataframe from calculate_stats
        show_counts: If True, show "91.1% (72)" format; otherwise "91.1%"

    Returns:
        Formatted dataframe for display
    """
    display_df = stats_df.copy()

    # Rename PASS to Pass for display
    display_severity_names = {'PASS': 'Pass', 'P4': 'P4', 'P3': 'P3', 'P2': 'P2', 'P1': 'P1', 'P0': 'P0'}

    for sev in SEVERITY_ORDER:
        pct_col = f'{sev} %'
        count_col = sev
        display_name = display_severity_names.get(sev, sev)

        if pct_col in display_df.columns and count_col in display_df.columns:
            if show_counts:
                display_df[display_name] = display_df.apply(
                    lambda row, s=sev, p=pct_col, c=count_col: f"{row[p]:.1f}% ({int(row[c])})"
                    if row[c] > 0 else "",
                    axis=1
                )
            else:
                display_df[display_name] = display_df.apply(
                    lambda row, s=sev, p=pct_col, c=count_col: f"{row[p]:.1f}%"
                    if row[c] > 0 else "",
                    axis=1
                )

    # Determine which columns to keep based on what's in the dataframe
    # Support both "Category" (aggregated) and "Round"/"Evaluation category" (with rounds)
    base_cols = []
    if 'Round' in display_df.columns:
        base_cols.append('Round')
    if 'Evaluation category' in display_df.columns:
        base_cols.append('Evaluation category')
    elif 'Category' in display_df.columns:
        base_cols.append('Category')
    base_cols.append('Count')

    # Use display names for severity columns
    display_severity_cols = [display_severity_names.get(s, s) for s in SEVERITY_ORDER]

    display_cols = base_cols + display_severity_cols
    display_df = display_df[[c for c in display_cols if c in display_df.columns]]

    return display_df


def calculate_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    group_by: str,
    mappings: Optional[dict] = None
) -> pd.DataFrame:
    """Calculate comparison statistics between two dataframes.

    Args:
        df1: First dataframe (R1/baseline)
        df2: Second dataframe (R2/comparison)
        group_by: Column name to group by
        mappings: Optional dict to map category names for alignment

    Returns:
        DataFrame with stats from both and delta calculations
    """
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Apply mappings if provided
    if mappings and group_by in df1_copy.columns:
        df1_copy[group_by] = df1_copy[group_by].replace(mappings)
    if mappings and group_by in df2_copy.columns:
        df2_copy[group_by] = df2_copy[group_by].replace(mappings)

    # Calculate stats for both
    stats1 = calculate_stats(df1_copy, group_by)
    stats2 = calculate_stats(df2_copy, group_by)

    # Rename columns for merging
    stats1_renamed = stats1.rename(columns={
        'Count': 'Count R1',
        **{sev: f'{sev} R1' for sev in SEVERITY_ORDER},
        **{f'{sev} %': f'{sev} % R1' for sev in SEVERITY_ORDER}
    })

    stats2_renamed = stats2.rename(columns={
        'Count': 'Count R2',
        **{sev: f'{sev} R2' for sev in SEVERITY_ORDER},
        **{f'{sev} %': f'{sev} % R2' for sev in SEVERITY_ORDER}
    })

    # Merge on Category
    comparison = pd.merge(
        stats1_renamed,
        stats2_renamed,
        on='Category',
        how='outer'
    ).fillna(0)

    # Calculate change (delta) for Pass %
    comparison['Change'] = (
        comparison['PASS % R2'] - comparison['PASS % R1']
    ).round(1)

    return comparison


def get_attack_distribution(df: pd.DataFrame, level: str = 'Attack L1') -> pd.DataFrame:
    """Get distribution of attacks at specified level.

    Args:
        df: Input dataframe
        level: Attack level column name

    Returns:
        DataFrame with Count and Percentage columns

    Raises:
        ValueError: If column not found, includes available columns in message
    """
    if level not in df.columns:
        available = list(df.columns)
        attack_cols = [c for c in df.columns if 'attack' in c.lower()]
        msg = f"Column '{level}' not found in dataframe.\n"
        msg += f"Available columns: {available}\n"
        if attack_cols:
            msg += f"Attack-related columns found: {attack_cols}"
        else:
            msg += "No attack-related columns found."
        raise ValueError(msg)

    counts = df[level].value_counts()
    total = counts.sum()

    distribution = pd.DataFrame({
        'Count': counts,
        'Percentage': (counts / total * 100).round(1)
    })

    distribution.index.name = level
    distribution = distribution.reset_index()

    return distribution


def get_failure_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Get distribution of failure severities.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with failure severity distribution
    """
    if 'Severity' not in df.columns:
        raise ValueError("Column 'Severity' not found in dataframe")

    # Filter to failures only
    failures = df[df['Severity'] != 'PASS']

    if len(failures) == 0:
        return pd.DataFrame(columns=['Severity', 'Count', 'Percentage'])

    counts = failures['Severity'].value_counts()
    total = counts.sum()

    distribution = pd.DataFrame({
        'Severity': counts.index,
        'Count': counts.values,
        'Percentage': (counts.values / total * 100).round(1)
    })

    # Sort by severity order (P4, P3, P2, P1, P0)
    severity_order = ['P4', 'P3', 'P2', 'P1', 'P0']
    distribution['sort_key'] = distribution['Severity'].apply(
        lambda x: severity_order.index(x) if x in severity_order else len(severity_order)
    )
    distribution = distribution.sort_values('sort_key').drop('sort_key', axis=1)
    distribution = distribution.reset_index(drop=True)

    return distribution


def filter_dataframe(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply multiple filters to a dataframe.

    Args:
        df: Input dataframe
        filters: Dict mapping column names to lists of allowed values
                 e.g., {'Risk L1': ['Safety', 'Security'], 'Severity': ['P0', 'P1']}

    Returns:
        Filtered dataframe
    """
    filtered = df.copy()

    for column, values in filters.items():
        if column in filtered.columns and values:
            filtered = filtered[filtered[column].isin(values)]

    return filtered


def calculate_stats_with_mapping(df: pd.DataFrame, group_by: str, mapping: dict) -> pd.DataFrame:
    """Calculate statistics using the column mapping.

    Args:
        df: Input dataframe
        group_by: Column name to group by
        mapping: Column mapping dict (used to get severity column if needed)

    Returns:
        DataFrame with statistics per category
    """
    if 'Severity' not in df.columns or group_by not in df.columns:
        return pd.DataFrame()

    stats = df.groupby(group_by).agg(
        Count=('Severity', 'count'),
        Pass=('Severity', lambda x: (x == 'PASS').sum()),
        P4=('Severity', lambda x: (x == 'P4').sum()),
        P3=('Severity', lambda x: (x == 'P3').sum()),
        P2=('Severity', lambda x: (x == 'P2').sum()),
        P1=('Severity', lambda x: (x == 'P1').sum()),
        P0=('Severity', lambda x: (x == 'P0').sum()),
    ).reset_index()

    stats = stats.rename(columns={group_by: 'Category'})

    # Calculate percentages
    for col in ['Pass', 'P4', 'P3', 'P2', 'P1', 'P0']:
        stats[f'{col} %'] = (stats[col] / stats['Count'] * 100).round(1)

    return stats


def format_stats_for_display(
    stats_df: pd.DataFrame,
    show_counts: bool = False,
    include_round: bool = False
) -> pd.DataFrame:
    """Format statistics dataframe for display.

    Args:
        stats_df: Statistics dataframe from calculate_stats_with_mapping
        show_counts: If True, show "91.1% (72)" format
        include_round: If True, include Round column in output

    Returns:
        Formatted dataframe for display
    """
    if stats_df.empty:
        return stats_df

    if include_round and 'Round' in stats_df.columns:
        display_cols = ['Round', 'Category', 'Count']
    else:
        display_cols = ['Category', 'Count']

    display_df = stats_df[display_cols].copy()

    for col in ['Pass', 'P4', 'P3', 'P2', 'P1', 'P0']:
        if f'{col} %' in stats_df.columns and col in stats_df.columns:
            if show_counts:
                display_df[col] = stats_df.apply(
                    lambda r, c=col: f"{r[f'{c} %']:.1f}% ({int(r[c])})" if r[c] > 0 else "",
                    axis=1
                )
            else:
                display_df[col] = stats_df.apply(
                    lambda r, c=col: f"{r[f'{c} %']:.1f}%" if r[c] > 0 else "",
                    axis=1
                )

    return display_df
