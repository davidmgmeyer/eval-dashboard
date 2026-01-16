"""Utilities for handling category name variations between eval rounds."""

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Try to import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# Path to mappings file (relative to project root)
MAPPINGS_FILE = Path(__file__).parent.parent / "data" / "category_mappings.json"


def load_mappings() -> dict:
    """Load saved category mappings from JSON file.

    Returns:
        Dictionary of mappings, or empty dict if file doesn't exist
    """
    if not MAPPINGS_FILE.exists():
        return {}

    try:
        with open(MAPPINGS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_mappings(mappings: dict) -> None:
    """Save category mappings to JSON file.

    Args:
        mappings: Dictionary of category mappings to save
    """
    # Create parent directory if needed
    MAPPINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MAPPINGS_FILE, 'w') as f:
        json.dump(mappings, f, indent=2)


def find_similar_categories(
    source_categories: list,
    target_categories: list,
    threshold: int = 70
) -> dict:
    """Find similar category names using fuzzy matching.

    Args:
        source_categories: List of category names to match from
        target_categories: List of category names to match against
        threshold: Minimum similarity score (0-100) to include in results

    Returns:
        Dictionary mapping source categories to suggested matches:
        {'disclosure': {'suggested': 'Vulnerability disclosure', 'confidence': 85}}
    """
    if not RAPIDFUZZ_AVAILABLE:
        return {}

    suggestions = {}

    for source in source_categories:
        # Skip if exact match exists
        if source in target_categories:
            continue

        # Find best match using rapidfuzz
        result = process.extractOne(
            source,
            target_categories,
            scorer=fuzz.token_sort_ratio
        )

        if result:
            match, score, _ = result
            if score >= threshold and match != source:
                suggestions[source] = {
                    'suggested': match,
                    'confidence': int(score)
                }

    return suggestions


def apply_mappings(
    df: pd.DataFrame,
    column: str,
    mappings: dict
) -> pd.DataFrame:
    """Apply category mappings to a dataframe column.

    Args:
        df: Input dataframe
        column: Column name to apply mappings to
        mappings: Dictionary mapping old values to new values

    Returns:
        DataFrame with mappings applied
    """
    if column not in df.columns:
        return df

    df_copy = df.copy()
    df_copy[column] = df_copy[column].replace(mappings)

    return df_copy
