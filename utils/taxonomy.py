"""Taxonomy utilities for hierarchical navigation of eval data."""

import pandas as pd
from typing import Optional


# Taxonomy level definitions
TAXONOMY_LEVELS = {
    'Risk': ['Risk L1', 'Risk L2', 'Risk L3'],
    'Attack': ['Attack L1', 'Attack L2', 'Attack L3'],
}


def get_hierarchy_path(row: pd.Series, taxonomy_type: str = 'Risk') -> str:
    """Get the full hierarchy path for a row.

    Args:
        row: A pandas Series (single row from dataframe)
        taxonomy_type: Either 'Risk' or 'Attack'

    Returns:
        String like "Safety > Emotional harm > Gaslighting"
    """
    if taxonomy_type not in TAXONOMY_LEVELS:
        raise ValueError(f"taxonomy_type must be one of {list(TAXONOMY_LEVELS.keys())}")

    levels = TAXONOMY_LEVELS[taxonomy_type]
    parts = []

    for level in levels:
        if level in row.index and pd.notna(row[level]):
            parts.append(str(row[level]))

    return " > ".join(parts) if parts else ""


def get_children(
    df: pd.DataFrame,
    parent_column: str,
    parent_value: str,
    child_column: str
) -> list:
    """Get all unique child categories for a given parent.

    Args:
        df: Input dataframe
        parent_column: Column name of the parent level (e.g., 'Risk L1')
        parent_value: Value of the parent category (e.g., 'Safety')
        child_column: Column name of the child level (e.g., 'Risk L2')

    Returns:
        List of unique child category values
    """
    if parent_column not in df.columns:
        raise ValueError(f"Column '{parent_column}' not found in dataframe")
    if child_column not in df.columns:
        raise ValueError(f"Column '{child_column}' not found in dataframe")

    filtered = df[df[parent_column] == parent_value]
    children = filtered[child_column].dropna().unique().tolist()

    return sorted(children)


def get_parent_column(column: str) -> Optional[str]:
    """Get the parent column for a given taxonomy column.

    Args:
        column: Current column name (e.g., 'Risk L2')

    Returns:
        Parent column name or None if at top level
    """
    for taxonomy_type, levels in TAXONOMY_LEVELS.items():
        if column in levels:
            idx = levels.index(column)
            if idx > 0:
                return levels[idx - 1]
            return None
    return None


def get_child_column(column: str) -> Optional[str]:
    """Get the child column for a given taxonomy column.

    Args:
        column: Current column name (e.g., 'Risk L1')

    Returns:
        Child column name or None if at bottom level
    """
    for taxonomy_type, levels in TAXONOMY_LEVELS.items():
        if column in levels:
            idx = levels.index(column)
            if idx < len(levels) - 1:
                return levels[idx + 1]
            return None
    return None


def get_taxonomy_type(column: str) -> Optional[str]:
    """Get the taxonomy type for a given column.

    Args:
        column: Column name (e.g., 'Risk L2')

    Returns:
        'Risk', 'Attack', or None
    """
    for taxonomy_type, levels in TAXONOMY_LEVELS.items():
        if column in levels:
            return taxonomy_type
    return None


def get_level_number(column: str) -> Optional[int]:
    """Get the level number (1, 2, or 3) for a column.

    Args:
        column: Column name (e.g., 'Risk L2')

    Returns:
        Level number (1, 2, or 3) or None
    """
    for taxonomy_type, levels in TAXONOMY_LEVELS.items():
        if column in levels:
            return levels.index(column) + 1
    return None


def get_available_taxonomy_columns(df: pd.DataFrame) -> dict:
    """Get which taxonomy columns are available in the dataframe.

    Args:
        df: Input dataframe

    Returns:
        Dict with 'Risk' and 'Attack' keys, each containing list of available columns
    """
    result = {}
    for taxonomy_type, levels in TAXONOMY_LEVELS.items():
        available = [col for col in levels if col in df.columns]
        if available:
            result[taxonomy_type] = available
    return result


def calculate_category_stats(
    df: pd.DataFrame,
    column: str,
    parent_column: Optional[str] = None,
    parent_value: Optional[str] = None
) -> pd.DataFrame:
    """Calculate pass/fail statistics for categories at a given level.

    Args:
        df: Input dataframe
        column: Column to group by
        parent_column: Optional parent column to filter by
        parent_value: Optional parent value to filter by

    Returns:
        DataFrame with Category, Count, Pass Count, Fail Count, Pass Rate columns
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if 'Severity' not in df.columns:
        raise ValueError("Column 'Severity' not found in dataframe")

    # Filter by parent if specified
    filtered = df.copy()
    if parent_column and parent_value and parent_column in df.columns:
        filtered = filtered[filtered[parent_column] == parent_value]

    # Group and calculate stats
    grouped = filtered.groupby(column).agg(
        Count=('Severity', 'count'),
        Pass_Count=('Severity', lambda x: (x == 'PASS').sum()),
    ).reset_index()

    grouped['Fail Count'] = grouped['Count'] - grouped['Pass_Count']
    grouped['Pass Rate'] = (grouped['Pass_Count'] / grouped['Count'] * 100).round(1)

    # Rename columns
    grouped = grouped.rename(columns={
        column: 'Category',
        'Pass_Count': 'Pass Count',
    })

    # Sort by pass rate (lowest first to highlight problems)
    grouped = grouped.sort_values('Pass Rate', ascending=True)

    return grouped


def get_worst_category(
    df: pd.DataFrame,
    column: str,
    min_count: int = 5
) -> Optional[dict]:
    """Get the category with the lowest pass rate.

    Args:
        df: Input dataframe
        column: Column to analyze
        min_count: Minimum sample size to consider

    Returns:
        Dict with 'category', 'pass_rate', 'count' or None
    """
    if column not in df.columns:
        return None

    stats = calculate_category_stats(df, column)
    stats = stats[stats['Count'] >= min_count]

    if len(stats) == 0:
        return None

    worst = stats.iloc[0]  # Already sorted ascending by pass rate
    return {
        'category': worst['Category'],
        'pass_rate': worst['Pass Rate'],
        'count': int(worst['Count']),
    }


def get_best_category(
    df: pd.DataFrame,
    column: str,
    min_count: int = 5
) -> Optional[dict]:
    """Get the category with the highest pass rate.

    Args:
        df: Input dataframe
        column: Column to analyze
        min_count: Minimum sample size to consider

    Returns:
        Dict with 'category', 'pass_rate', 'count' or None
    """
    if column not in df.columns:
        return None

    stats = calculate_category_stats(df, column)
    stats = stats[stats['Count'] >= min_count]

    if len(stats) == 0:
        return None

    best = stats.iloc[-1]  # Sorted ascending, so last is highest
    return {
        'category': best['Category'],
        'pass_rate': best['Pass Rate'],
        'count': int(best['Count']),
    }
