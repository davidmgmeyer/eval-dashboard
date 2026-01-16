"""Claude API integration for analyzing eval failures."""

from typing import Optional

import pandas as pd

# Try to import anthropic for API calls
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024


def get_category_insights(
    df: pd.DataFrame,
    category_column: str,
    category_value: str,
    api_key: Optional[str] = None
) -> str:
    """Analyze failures for a specific category using Claude.

    Args:
        df: Input dataframe with eval results
        category_column: Column name to filter by
        category_value: Value to filter for
        api_key: Anthropic API key (optional, uses env var if not provided)

    Returns:
        Analysis text from Claude, or error/warning message
    """
    if not ANTHROPIC_AVAILABLE:
        return "⚠️ Anthropic library not installed. Run `pip install anthropic` to enable insights."

    if not api_key:
        return "⚠️ API key required. Please enter your Anthropic API key in the sidebar."

    # Filter to specific category
    if category_column not in df.columns:
        return f"⚠️ Column '{category_column}' not found in data."

    category_df = df[df[category_column] == category_value]

    if len(category_df) == 0:
        return f"⚠️ No data found for {category_value}."

    # Get failures only
    if 'Severity' not in df.columns:
        return "⚠️ 'Severity' column not found in data."

    failures = category_df[category_df['Severity'] != 'PASS']

    if len(failures) == 0:
        return f"✅ No failures found for {category_value}. All evaluations passed!"

    # Sample justifications (up to 15)
    sample_size = min(15, len(failures))
    sampled = failures.sample(n=sample_size, random_state=42)

    # Build justifications text
    justifications = []
    for idx, row in sampled.iterrows():
        severity = row.get('Severity', 'Unknown')
        justification = row.get('Justification', 'No justification provided')
        justifications.append(f"[{severity}] {justification}")

    justifications_text = "\n\n".join(justifications)

    # Build prompt
    prompt = f"""Analyze these AI safety evaluation failures for the category "{category_value}".

Here are {sample_size} sample failure justifications (out of {len(failures)} total failures):

{justifications_text}

Please provide:

1. **Top 3 Root Causes** - What are the main reasons for these failures? Estimate the percentage of failures each cause represents.

2. **Severity Analysis** - Comment on the distribution of severity levels and what the most critical issues are.

3. **Improvement Recommendations** - Provide 3-5 specific, actionable recommendations to reduce failures in this category.

Keep your analysis concise and focused on actionable insights."""

    # Call Claude API
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"⚠️ API call failed: {str(e)}"


def analyze_failure_patterns(
    df: pd.DataFrame,
    api_key: Optional[str] = None
) -> str:
    """Analyze what types of attacks/risks the agent is most sensitive to.

    Args:
        df: Input dataframe with eval results
        api_key: Anthropic API key

    Returns:
        Analysis text from Claude
    """
    if not ANTHROPIC_AVAILABLE:
        return "⚠️ Anthropic library not installed. Run `pip install anthropic` to enable insights."

    if not api_key:
        return "⚠️ API key required. Please enter your Anthropic API key in the sidebar."

    if 'Severity' not in df.columns:
        return "⚠️ 'Severity' column not found in data."

    # Get failures only
    failures = df[df['Severity'] != 'PASS']

    if len(failures) == 0:
        return "✅ No failures found in the data. All evaluations passed!"

    # Analyze by attack type if available
    attack_analysis = ""
    if 'Attack L1' in df.columns:
        attack_stats = failures.groupby('Attack L1').agg(
            Count=('Severity', 'count'),
            P0_P1=('Severity', lambda x: ((x == 'P0') | (x == 'P1')).sum()),
        ).reset_index()
        attack_stats['Critical_Rate'] = (attack_stats['P0_P1'] / attack_stats['Count'] * 100).round(1)
        attack_stats = attack_stats.sort_values('Count', ascending=False)

        attack_lines = [f"- {row['Attack L1']}: {row['Count']} failures ({row['Critical_Rate']}% critical)"
                       for _, row in attack_stats.head(10).iterrows()]
        attack_analysis = "**Failures by Attack Type (L1):**\n" + "\n".join(attack_lines)

    # Analyze by risk type if available
    risk_analysis = ""
    if 'Risk L1' in df.columns:
        risk_stats = failures.groupby('Risk L1').agg(
            Count=('Severity', 'count'),
            P0_P1=('Severity', lambda x: ((x == 'P0') | (x == 'P1')).sum()),
        ).reset_index()
        risk_stats['Critical_Rate'] = (risk_stats['P0_P1'] / risk_stats['Count'] * 100).round(1)
        risk_stats = risk_stats.sort_values('Count', ascending=False)

        risk_lines = [f"- {row['Risk L1']}: {row['Count']} failures ({row['Critical_Rate']}% critical)"
                     for _, row in risk_stats.head(10).iterrows()]
        risk_analysis = "**Failures by Risk Type (L1):**\n" + "\n".join(risk_lines)

    # Severity distribution
    severity_dist = failures['Severity'].value_counts()
    severity_lines = [f"- {sev}: {count}" for sev, count in severity_dist.items()]
    severity_analysis = "**Severity Distribution:**\n" + "\n".join(severity_lines)

    # Cross-tabulation for attack/risk combinations if both available
    combo_analysis = ""
    if 'Attack L1' in df.columns and 'Risk L1' in df.columns:
        combo_stats = failures.groupby(['Attack L1', 'Risk L1']).size().reset_index(name='Count')
        combo_stats = combo_stats.sort_values('Count', ascending=False).head(10)
        combo_lines = [f"- {row['Attack L1']} × {row['Risk L1']}: {row['Count']} failures"
                      for _, row in combo_stats.iterrows()]
        combo_analysis = "**Top Attack/Risk Combinations:**\n" + "\n".join(combo_lines)

    prompt = f"""Analyze these AI agent safety evaluation failure patterns.

Total failures: {len(failures)} out of {len(df)} evaluations ({len(failures)/len(df)*100:.1f}% failure rate)

{severity_analysis}

{attack_analysis}

{risk_analysis}

{combo_analysis}

Please analyze:

1. **Vulnerability Profile** - What types of attacks is this agent most vulnerable to? Why might these be challenging?

2. **Risk Hotspots** - Which risk categories show the most failures? What does this tell us about the agent's weaknesses?

3. **Critical Patterns** - Are there specific attack/risk combinations that lead to critical (P0/P1) failures?

4. **Systemic Issues** - Do you see any systemic patterns that suggest fundamental issues rather than edge cases?

Keep your analysis focused and actionable."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"⚠️ API call failed: {str(e)}"


def analyze_specific_failures(
    df: pd.DataFrame,
    selected_indices: list,
    api_key: Optional[str] = None
) -> str:
    """Deep dive into specific selected failures.

    Args:
        df: Input dataframe with eval results
        selected_indices: List of row indices to analyze
        api_key: Anthropic API key

    Returns:
        Analysis text from Claude
    """
    if not ANTHROPIC_AVAILABLE:
        return "⚠️ Anthropic library not installed. Run `pip install anthropic` to enable insights."

    if not api_key:
        return "⚠️ API key required. Please enter your Anthropic API key in the sidebar."

    if not selected_indices:
        return "⚠️ No failures selected. Please select some failures to analyze."

    # Get selected rows
    selected = df.loc[selected_indices]

    # Build failure details
    failure_details = []
    for idx, row in selected.iterrows():
        detail = f"**Failure #{idx}**\n"

        # Add risk path if available
        risk_parts = []
        for col in ['Risk L1', 'Risk L2', 'Risk L3']:
            if col in row.index and pd.notna(row[col]):
                risk_parts.append(str(row[col]))
        if risk_parts:
            detail += f"- Risk: {' > '.join(risk_parts)}\n"

        # Add attack path if available
        attack_parts = []
        for col in ['Attack L1', 'Attack L2', 'Attack L3']:
            if col in row.index and pd.notna(row[col]):
                attack_parts.append(str(row[col]))
        if attack_parts:
            detail += f"- Attack: {' > '.join(attack_parts)}\n"

        # Add severity
        if 'Severity' in row.index:
            detail += f"- Severity: {row['Severity']}\n"

        # Add justification
        if 'Justification' in row.index and pd.notna(row['Justification']):
            justification = str(row['Justification'])[:500]  # Truncate long justifications
            detail += f"- Justification: {justification}\n"

        failure_details.append(detail)

    failures_text = "\n".join(failure_details)

    prompt = f"""Analyze these {len(selected_indices)} specific AI safety evaluation failures in detail:

{failures_text}

Please provide:

1. **Common Threads** - What patterns or themes connect these failures? Are there shared root causes?

2. **Individual Analysis** - For each failure, briefly explain why it likely occurred and what made it fail.

3. **Severity Assessment** - Do the severity levels assigned seem appropriate? Any that seem over/under-rated?

4. **Specific Fixes** - What concrete changes would prevent each type of failure?

Be specific and actionable in your recommendations."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS * 2,  # Allow longer response for detailed analysis
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"⚠️ API call failed: {str(e)}"


def generate_recommendations(
    df: pd.DataFrame,
    api_key: Optional[str] = None
) -> str:
    """Generate actionable recommendations to improve the agent.

    Args:
        df: Input dataframe with eval results
        api_key: Anthropic API key

    Returns:
        Recommendations text from Claude
    """
    if not ANTHROPIC_AVAILABLE:
        return "⚠️ Anthropic library not installed. Run `pip install anthropic` to enable insights."

    if not api_key:
        return "⚠️ API key required. Please enter your Anthropic API key in the sidebar."

    if 'Severity' not in df.columns:
        return "⚠️ 'Severity' column not found in data."

    # Calculate overall stats
    total = len(df)
    passes = len(df[df['Severity'] == 'PASS'])
    failures = df[df['Severity'] != 'PASS']
    pass_rate = passes / total * 100 if total > 0 else 0

    # Critical failures (P0, P1)
    critical = failures[failures['Severity'].isin(['P0', 'P1'])]

    # Find worst categories
    worst_risks = ""
    if 'Risk L2' in df.columns:
        risk_stats = df.groupby('Risk L2').agg(
            Total=('Severity', 'count'),
            Passes=('Severity', lambda x: (x == 'PASS').sum()),
        ).reset_index()
        risk_stats['Pass_Rate'] = (risk_stats['Passes'] / risk_stats['Total'] * 100).round(1)
        risk_stats = risk_stats[risk_stats['Total'] >= 5].sort_values('Pass_Rate').head(5)
        if len(risk_stats) > 0:
            worst_risk_lines = [f"- {row['Risk L2']}: {row['Pass_Rate']}% pass rate ({row['Total']} evals)"
                               for _, row in risk_stats.iterrows()]
            worst_risks = "**Worst Risk Categories (L2):**\n" + "\n".join(worst_risk_lines)

    worst_attacks = ""
    if 'Attack L1' in df.columns:
        attack_stats = df.groupby('Attack L1').agg(
            Total=('Severity', 'count'),
            Passes=('Severity', lambda x: (x == 'PASS').sum()),
        ).reset_index()
        attack_stats['Pass_Rate'] = (attack_stats['Passes'] / attack_stats['Total'] * 100).round(1)
        attack_stats = attack_stats[attack_stats['Total'] >= 5].sort_values('Pass_Rate').head(5)
        if len(attack_stats) > 0:
            worst_attack_lines = [f"- {row['Attack L1']}: {row['Pass_Rate']}% pass rate ({row['Total']} evals)"
                                 for _, row in attack_stats.iterrows()]
            worst_attacks = "**Most Effective Attack Types (L1):**\n" + "\n".join(worst_attack_lines)

    # Sample critical failure justifications
    critical_samples = ""
    if len(critical) > 0 and 'Justification' in critical.columns:
        sample_size = min(5, len(critical))
        sampled = critical.sample(n=sample_size, random_state=42)
        sample_lines = []
        for _, row in sampled.iterrows():
            sev = row.get('Severity', 'Unknown')
            just = str(row.get('Justification', 'No justification'))[:200]
            sample_lines.append(f"- [{sev}] {just}")
        critical_samples = "**Sample Critical Failure Justifications:**\n" + "\n".join(sample_lines)

    prompt = f"""Based on these AI safety evaluation results, provide actionable recommendations to improve the agent.

**Overall Performance:**
- Total evaluations: {total}
- Pass rate: {pass_rate:.1f}%
- Total failures: {len(failures)}
- Critical failures (P0/P1): {len(critical)}

{worst_risks}

{worst_attacks}

{critical_samples}

Please provide:

1. **Priority Actions** - What are the top 5 specific, actionable changes that would have the biggest impact on improving the pass rate?

2. **Quick Wins** - What 2-3 issues could be fixed quickly with minimal effort?

3. **Systemic Recommendations** - What broader changes to the agent's training, prompts, or architecture might help?

4. **Testing Recommendations** - What additional evaluation scenarios should be added to better understand the agent's weaknesses?

Be specific and practical. Avoid generic advice - focus on what the data tells us about THIS agent's specific issues."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS * 2,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"⚠️ API call failed: {str(e)}"


def generate_executive_summary(
    df: pd.DataFrame,
    api_key: Optional[str] = None
) -> str:
    """Generate a concise executive summary suitable for customer emails.

    Args:
        df: Input dataframe with eval results
        api_key: Anthropic API key

    Returns:
        Executive summary text from Claude
    """
    if not ANTHROPIC_AVAILABLE:
        return "⚠️ Anthropic library not installed. Run `pip install anthropic` to enable insights."

    if not api_key:
        return "⚠️ API key required. Please enter your Anthropic API key in the sidebar."

    if 'Severity' not in df.columns:
        return "⚠️ 'Severity' column not found in data."

    # Calculate stats
    total = len(df)
    passes = len(df[df['Severity'] == 'PASS'])
    failures = df[df['Severity'] != 'PASS']
    pass_rate = passes / total * 100 if total > 0 else 0

    # Critical failures
    critical = len(df[df['Severity'].isin(['P0', 'P1'])])

    # Best and worst categories
    strengths = []
    weaknesses = []

    if 'Risk L2' in df.columns:
        risk_stats = df.groupby('Risk L2').agg(
            Total=('Severity', 'count'),
            Passes=('Severity', lambda x: (x == 'PASS').sum()),
        ).reset_index()
        risk_stats['Pass_Rate'] = (risk_stats['Passes'] / risk_stats['Total'] * 100).round(1)
        risk_stats = risk_stats[risk_stats['Total'] >= 5]

        if len(risk_stats) > 0:
            best = risk_stats.nlargest(3, 'Pass_Rate')
            for _, row in best.iterrows():
                strengths.append(f"{row['Risk L2']} ({row['Pass_Rate']}%)")

            worst = risk_stats.nsmallest(3, 'Pass_Rate')
            for _, row in worst.iterrows():
                weaknesses.append(f"{row['Risk L2']} ({row['Pass_Rate']}%)")

    strengths_text = ", ".join(strengths) if strengths else "N/A"
    weaknesses_text = ", ".join(weaknesses) if weaknesses else "N/A"

    # Severity breakdown
    severity_counts = df['Severity'].value_counts().to_dict()
    severity_text = ", ".join([f"{k}: {v}" for k, v in severity_counts.items()])

    prompt = f"""Generate a concise executive summary of this AI safety evaluation that can be copy-pasted into a customer email.

**Evaluation Results:**
- Total evaluations: {total:,}
- Overall pass rate: {pass_rate:.1f}%
- Critical failures (P0/P1): {critical}
- Severity breakdown: {severity_text}

**Strongest Areas (highest pass rates):**
{strengths_text}

**Areas Needing Improvement (lowest pass rates):**
{weaknesses_text}

Write a professional, 1-paragraph executive summary (4-6 sentences) that covers:
1. Overall agent performance (good/needs work/concerning)
2. Key strengths
3. Key vulnerabilities
4. Most concerning failures
5. Top 3 recommended next steps

The tone should be professional and balanced - acknowledge both positives and areas for improvement.
Do NOT use markdown formatting or bullet points - write it as flowing prose suitable for an email body.
Start directly with the summary content, not with "Here is..." or similar."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"⚠️ API call failed: {str(e)}"


def get_comparison_insights(
    comparison_df: pd.DataFrame,
    api_key: Optional[str] = None
) -> str:
    """Analyze comparison data between two eval rounds using Claude.

    Args:
        comparison_df: Comparison dataframe with Change column
        api_key: Anthropic API key (optional, uses env var if not provided)

    Returns:
        Analysis text from Claude, or error/warning message
    """
    if not ANTHROPIC_AVAILABLE:
        return "⚠️ Anthropic library not installed. Run `pip install anthropic` to enable insights."

    if not api_key:
        return "⚠️ API key required. Please enter your Anthropic API key in the sidebar."

    if 'Change' not in comparison_df.columns:
        return "⚠️ 'Change' column not found in comparison data."

    if 'Category' not in comparison_df.columns:
        return "⚠️ 'Category' column not found in comparison data."

    # Find top improvements (positive change = higher pass rate)
    improvements = comparison_df[comparison_df['Change'] > 0].nlargest(5, 'Change')

    # Find top regressions (negative change = lower pass rate)
    regressions = comparison_df[comparison_df['Change'] < 0].nsmallest(5, 'Change')

    # Build improvements text
    if len(improvements) > 0:
        improvements_text = "\n".join([
            f"- {row['Category']}: +{row['Change']:.1f}% (R1: {row.get('PASS % R1', 'N/A'):.1f}% → R2: {row.get('PASS % R2', 'N/A'):.1f}%)"
            for _, row in improvements.iterrows()
        ])
    else:
        improvements_text = "No improvements found."

    # Build regressions text
    if len(regressions) > 0:
        regressions_text = "\n".join([
            f"- {row['Category']}: {row['Change']:.1f}% (R1: {row.get('PASS % R1', 'N/A'):.1f}% → R2: {row.get('PASS % R2', 'N/A'):.1f}%)"
            for _, row in regressions.iterrows()
        ])
    else:
        regressions_text = "No regressions found."

    # Overall stats
    total_categories = len(comparison_df)
    improved_count = len(comparison_df[comparison_df['Change'] > 0])
    regressed_count = len(comparison_df[comparison_df['Change'] < 0])
    unchanged_count = len(comparison_df[comparison_df['Change'] == 0])
    avg_change = comparison_df['Change'].mean()

    # Build prompt
    prompt = f"""Analyze this comparison between two AI safety evaluation rounds (R1 = baseline, R2 = new).

**Overall Statistics:**
- Total categories: {total_categories}
- Improved: {improved_count} categories
- Regressed: {regressed_count} categories
- Unchanged: {unchanged_count} categories
- Average change in pass rate: {avg_change:+.1f}%

**Top 5 Improvements (higher pass rate in R2):**
{improvements_text}

**Top 5 Regressions (lower pass rate in R2):**
{regressions_text}

Please provide:

1. **Executive Summary** - A 2-3 sentence overview of the overall performance change.

2. **Key Wins** - What improvements are most significant and why?

3. **Areas of Concern** - What regressions need immediate attention?

4. **Recommendations** - What should be prioritized for the next iteration?

Keep your analysis concise and focused on actionable insights."""

    # Call Claude API
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"⚠️ API call failed: {str(e)}"
