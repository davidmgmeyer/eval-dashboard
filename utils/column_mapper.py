"""Column mapping utilities - let users define which columns to use."""

import json
import os
import re
from typing import Optional

import pandas as pd


# Define the roles columns can have
COLUMN_ROLES = {
    'risk_hierarchy': {
        'name': 'Risk Hierarchy',
        'description': 'Risk categories from broad (L1) to specific (L3)',
        'multi': True,
        'max_columns': 3,
        'required': True,
    },
    'attack_hierarchy': {
        'name': 'Attack Hierarchy',
        'description': 'Attack types from broad (L1) to specific (L3)',
        'multi': True,
        'max_columns': 3,
        'required': False,
    },
    'severity': {
        'name': 'Severity / Grade',
        'description': 'Column containing PASS, P4, P3, P2, P1, P0 values',
        'multi': False,
        'required': True,
    },
    'round': {
        'name': 'Eval Round',
        'description': 'Column indicating which evaluation round (e.g., Round 1, Round 2)',
        'multi': False,
        'required': False,
    },
    'justification': {
        'name': 'Justification / Reason',
        'description': 'Text explaining why the eval failed (used for AI analysis)',
        'multi': False,
        'required': False,
    },
    'transcript': {
        'name': 'Conversation Transcript',
        'description': 'Full conversation log (used for deep AI analysis)',
        'multi': False,
        'required': False,
    },
    'eval_id': {
        'name': 'Eval ID',
        'description': 'Unique identifier for each evaluation',
        'multi': False,
        'required': False,
    },
}

# Directory for saved mappings
MAPPINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def auto_detect_columns(df: pd.DataFrame) -> dict:
    """Try to automatically detect column mappings based on common names.

    Args:
        df: Input dataframe

    Returns:
        Dict with suggested mappings for each role
    """
    suggestions = {}
    columns = df.columns.tolist()

    # Risk hierarchy detection
    risk_cols = [c for c in columns if 'risk' in c.lower()]
    if risk_cols:
        # Sort by level number if present (L1, L2, L3 or level 1, level 2, etc.)
        def get_level(col):
            col_lower = col.lower()
            # Check for L1, L2, L3
            match = re.search(r'l(\d)', col_lower)
            if match:
                return int(match.group(1))
            # Check for "level 1", "level 2", etc.
            match = re.search(r'level\s*(\d)', col_lower)
            if match:
                return int(match.group(1))
            return 99  # Unknown level goes last

        risk_cols_sorted = sorted(risk_cols, key=get_level)
        suggestions['risk_hierarchy'] = risk_cols_sorted[:3]

    # Attack hierarchy detection
    attack_cols = [c for c in columns if 'attack' in c.lower()]
    if attack_cols:
        def get_level(col):
            col_lower = col.lower()
            match = re.search(r'l(\d)', col_lower)
            if match:
                return int(match.group(1))
            match = re.search(r'level\s*(\d)', col_lower)
            if match:
                return int(match.group(1))
            return 99

        attack_cols_sorted = sorted(attack_cols, key=get_level)
        suggestions['attack_hierarchy'] = attack_cols_sorted[:3]

    # Severity detection
    severity_keywords = ['severity', 'grade', 'result', 'score', 'outcome']
    for c in columns:
        if c.lower() in severity_keywords:
            suggestions['severity'] = c
            break

    # If no keyword match, check for columns containing expected values
    if 'severity' not in suggestions:
        for c in columns:
            try:
                col_values = df[c].astype(str).str.upper()
                if col_values.str.contains(r'^(PASS|P[0-4])$', regex=True).any():
                    suggestions['severity'] = c
                    break
            except Exception:
                continue

    # Round detection
    round_keywords = ['round', 'eval round', 'evaluation round', 'iteration']
    for c in columns:
        col_lower = c.lower().replace('_', ' ')
        if any(kw in col_lower for kw in round_keywords):
            suggestions['round'] = c
            break

    # Justification detection
    just_keywords = ['justification', 'reason', 'explanation', 'why', 'rationale', 'notes']
    for c in columns:
        col_lower = c.lower()
        if any(kw in col_lower for kw in just_keywords):
            suggestions['justification'] = c
            break

    # Transcript detection
    trans_keywords = ['transcript', 'conversation', 'log', 'chat', 'dialogue', 'messages']
    for c in columns:
        col_lower = c.lower()
        if any(kw in col_lower for kw in trans_keywords):
            suggestions['transcript'] = c
            break

    # ID detection
    id_keywords = ['id', 'eval_id', 'run_id', 'uuid', 'identifier']
    for c in columns:
        col_lower = c.lower().replace('_', '')
        if any(kw.replace('_', '') in col_lower for kw in id_keywords):
            suggestions['eval_id'] = c
            break

    return suggestions


def validate_mapping(mapping: dict) -> tuple:
    """Validate a column mapping.

    Args:
        mapping: Column mapping dict

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check required fields
    if not mapping.get('severity') or mapping.get('severity') == '(none)':
        errors.append("Severity column is required")

    if not mapping.get('risk_hierarchy'):
        errors.append("At least one Risk column is required")

    # Check max columns for hierarchies
    if len(mapping.get('risk_hierarchy', [])) > 3:
        errors.append("Maximum 3 Risk hierarchy columns allowed")

    if len(mapping.get('attack_hierarchy', [])) > 3:
        errors.append("Maximum 3 Attack hierarchy columns allowed")

    return len(errors) == 0, errors


def apply_mapping_to_df(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Create a standardized dataframe using the column mapping.

    Args:
        df: Original dataframe
        mapping: Column mapping dict

    Returns:
        Dataframe with standardized column names
    """
    result = df.copy()
    rename_map = {}

    # Map risk hierarchy to Risk L1, Risk L2, Risk L3
    for i, col in enumerate(mapping.get('risk_hierarchy', [])):
        if col and col in df.columns:
            rename_map[col] = f'Risk L{i + 1}'

    # Map attack hierarchy to Attack L1, Attack L2, Attack L3
    for i, col in enumerate(mapping.get('attack_hierarchy', [])):
        if col and col in df.columns:
            rename_map[col] = f'Attack L{i + 1}'

    # Map single-column fields
    single_mappings = {
        'severity': 'Severity',
        'round': 'Eval round',
        'justification': 'Justification',
        'transcript': 'Transcript',
        'eval_id': 'Eval ID',
    }

    for key, standard_name in single_mappings.items():
        col = mapping.get(key)
        if col and col != '(none)' and col in df.columns:
            rename_map[col] = standard_name

    # Apply renaming
    if rename_map:
        result = result.rename(columns=rename_map)

    return result


def get_mapped_column_name(mapping: dict, role: str, level: int = 0) -> Optional[str]:
    """Get the standardized column name for a role.

    Args:
        mapping: Column mapping dict
        role: Role name (e.g., 'risk_hierarchy', 'severity')
        level: For hierarchies, which level (0, 1, or 2)

    Returns:
        Standardized column name or None
    """
    if role == 'risk_hierarchy':
        cols = mapping.get('risk_hierarchy', [])
        if level < len(cols):
            return f'Risk L{level + 1}'
        return None
    elif role == 'attack_hierarchy':
        cols = mapping.get('attack_hierarchy', [])
        if level < len(cols):
            return f'Attack L{level + 1}'
        return None
    else:
        single_mappings = {
            'severity': 'Severity',
            'round': 'Eval round',
            'justification': 'Justification',
            'transcript': 'Transcript',
            'eval_id': 'Eval ID',
        }
        col = mapping.get(role)
        if col and col != '(none)':
            return single_mappings.get(role)
        return None


def get_hierarchy_columns(mapping: dict, hierarchy_type: str) -> list:
    """Get the list of standardized column names for a hierarchy.

    Args:
        mapping: Column mapping dict
        hierarchy_type: 'risk' or 'attack'

    Returns:
        List of standardized column names (e.g., ['Risk L1', 'Risk L2'])
    """
    if hierarchy_type == 'risk':
        cols = mapping.get('risk_hierarchy', [])
        return [f'Risk L{i + 1}' for i in range(len(cols))]
    elif hierarchy_type == 'attack':
        cols = mapping.get('attack_hierarchy', [])
        return [f'Attack L{i + 1}' for i in range(len(cols))]
    return []


def save_column_mapping(mapping: dict, name: str) -> str:
    """Save column mapping for reuse.

    Args:
        mapping: Column mapping dict
        name: Name for this mapping

    Returns:
        Path to saved file
    """
    os.makedirs(MAPPINGS_DIR, exist_ok=True)

    # Sanitize filename
    safe_name = re.sub(r'[^\w\-]', '_', name)
    filepath = os.path.join(MAPPINGS_DIR, f'column_mapping_{safe_name}.json')

    with open(filepath, 'w') as f:
        json.dump(mapping, f, indent=2)

    return filepath


def load_column_mapping(name: str) -> Optional[dict]:
    """Load previously saved column mapping.

    Args:
        name: Name of the mapping to load

    Returns:
        Mapping dict or None if not found
    """
    safe_name = re.sub(r'[^\w\-]', '_', name)
    filepath = os.path.join(MAPPINGS_DIR, f'column_mapping_{safe_name}.json')

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def list_saved_mappings() -> list:
    """List all saved column mappings.

    Returns:
        List of mapping names
    """
    if not os.path.exists(MAPPINGS_DIR):
        return []

    mappings = []
    for f in os.listdir(MAPPINGS_DIR):
        if f.startswith('column_mapping_') and f.endswith('.json'):
            name = f[len('column_mapping_'):-len('.json')]
            mappings.append(name)

    return sorted(mappings)


def delete_column_mapping(name: str) -> bool:
    """Delete a saved column mapping.

    Args:
        name: Name of the mapping to delete

    Returns:
        True if deleted, False if not found
    """
    safe_name = re.sub(r'[^\w\-]', '_', name)
    filepath = os.path.join(MAPPINGS_DIR, f'column_mapping_{safe_name}.json')

    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
