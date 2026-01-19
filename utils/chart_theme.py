"""Chart styling to match AIUC-1 brand."""

# AIUC-1 Color Palette
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Purple
    'background': '#0f0f0f',
    'surface': '#1a1a1a',
    'border': '#2a2a2a',
    'text': '#ffffff',
    'text_muted': '#888888',

    # Risk category colors (matching AIUC-1 categories)
    'safety': '#ef4444',       # Red
    'security': '#f59e0b',     # Amber
    'reliability': '#10b981',  # Emerald
    'data_privacy': '#3b82f6', # Blue
    'accountability': '#8b5cf6', # Purple
    'society': '#ec4899',      # Pink

    # Severity colors
    'pass': '#10b981',
    'p4': '#fbbf24',
    'p3': '#f97316',
    'p2': '#ef4444',
    'p1': '#dc2626',
    'p0': '#991b1b',
}

# Sequential color scale for heatmaps (dark to indigo)
HEATMAP_COLORS = [
    [0, '#1a1a1a'],
    [0.25, '#3730a3'],
    [0.5, '#6366f1'],
    [0.75, '#818cf8'],
    [1, '#c7d2fe']
]

# Diverging color scale for comparison heatmaps
DIVERGING_COLORS = [
    [0, '#dc2626'],    # Negative - Red
    [0.5, '#1a1a1a'],  # Neutral - Dark
    [1, '#10b981']     # Positive - Green
]

# Severity color map for charts
SEVERITY_COLORS = {
    'PASS': COLORS['pass'],
    'P4': COLORS['p4'],
    'P3': COLORS['p3'],
    'P2': COLORS['p2'],
    'P1': COLORS['p1'],
    'P0': COLORS['p0'],
}


def get_plotly_layout():
    """Get standard Plotly layout matching AIUC-1 theme."""
    return {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['surface'],
        'font': {
            'family': 'Inter, sans-serif',
            'color': COLORS['text'],
            'size': 12
        },
        'title': {
            'font': {
                'size': 18,
                'color': COLORS['text']
            }
        },
        'xaxis': {
            'gridcolor': COLORS['border'],
            'linecolor': COLORS['border'],
            'tickfont': {'color': COLORS['text_muted']},
            'zerolinecolor': COLORS['border']
        },
        'yaxis': {
            'gridcolor': COLORS['border'],
            'linecolor': COLORS['border'],
            'tickfont': {'color': COLORS['text_muted']},
            'zerolinecolor': COLORS['border']
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': COLORS['text']}
        },
        'margin': {'t': 60, 'r': 20, 'b': 60, 'l': 60}
    }


def style_plotly_chart(fig):
    """Apply AIUC-1 styling to a Plotly figure."""
    layout = get_plotly_layout()
    fig.update_layout(**layout)
    return fig


def get_severity_color(severity: str) -> str:
    """Get color for a severity level."""
    return SEVERITY_COLORS.get(severity, COLORS['text_muted'])


def get_risk_color(risk_category: str) -> str:
    """Get color for a risk category."""
    # Normalize the category name
    category_lower = risk_category.lower().replace(' ', '_').replace('&', 'and')

    risk_map = {
        'safety': COLORS['safety'],
        'security': COLORS['security'],
        'reliability': COLORS['reliability'],
        'data_and_privacy': COLORS['data_privacy'],
        'data_privacy': COLORS['data_privacy'],
        'accountability': COLORS['accountability'],
        'society': COLORS['society'],
    }

    for key, color in risk_map.items():
        if key in category_lower:
            return color

    return COLORS['primary']


def get_heatmap_colorscale(diverging: bool = False):
    """Get colorscale for heatmaps.
    
    Args:
        diverging: If True, return diverging scale for comparisons
    
    Returns:
        Colorscale list for Plotly
    """
    if diverging:
        return DIVERGING_COLORS
    return HEATMAP_COLORS
