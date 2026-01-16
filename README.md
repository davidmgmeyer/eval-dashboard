# Eval Dashboard

A Streamlit application for analyzing AI agent safety evaluation results.

## Features

- **Single Round Analysis**: Upload a CSV file to analyze pass/fail statistics, attack distribution, failure breakdown, and get AI-powered insights
- **Round Comparison**: Compare two evaluation rounds side-by-side with automatic category mapping
- **AI Insights**: Use Claude to analyze failure patterns and get recommendations
- **Export Options**: Download results as CSV or copy to clipboard

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Configuration

The app supports configuration through environment variables, a `.env` file, or Streamlit secrets (for deployed apps).

### Local Development

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   # Anthropic API key for AI insights
   ANTHROPIC_API_KEY=sk-ant-your-key-here

   # Optional: Enable authentication
   ENABLE_AUTH=true
   AUTHORIZED_EMAILS=alice@company.com,bob@company.com
   ```

3. Run the app - it will automatically load settings from `.env`

### Streamlit Cloud Deployment

For deployment on Streamlit Cloud, use [Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management):

1. Go to your app's settings in Streamlit Cloud
2. Navigate to the "Secrets" section
3. Add your configuration in TOML format:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ENABLE_AUTH = "true"
   AUTHORIZED_EMAILS = "alice@company.com,bob@company.com"
   ```

### Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ANTHROPIC_API_KEY` | string | - | API key for Claude AI insights |
| `AUTHORIZED_EMAILS` | string | - | Comma-separated list of authorized emails |
| `APP_TITLE` | string | "Eval Dashboard" | Application title |
| `APP_ICON` | string | "📊" | Application icon (emoji) |
| `MAX_UPLOAD_SIZE_MB` | int | 50 | Maximum file upload size in MB |
| `ENABLE_AUTH` | bool | false | Enable email-based authentication |
| `DEBUG` | bool | false | Show debug information in sidebar |

### Priority Order

Configuration values are loaded in this order (first found wins):
1. Environment variables
2. Streamlit secrets (`st.secrets`)
3. Default values

## CSV Format

Your evaluation CSV should contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| Risk L1 | Top-level risk category | Safety, Security |
| Risk L2 | Second-level risk category | Physical harm, Data exposure |
| Risk L3 | Detailed risk category | Violence, PII leakage |
| Attack L1 | Top-level attack type | Direct, Indirect |
| Attack L2 | Second-level attack type | Jailbreak, Social engineering |
| Severity | Result grade | PASS, P0, P1, P2, P3, P4 |
| Justification | Explanation of the grade | "Model refused appropriately" |

Column names are flexible - variations like `risk_l1`, `Risk_L1`, or `risk l1` are accepted.

## Project Structure

```
eval-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment configuration
├── .gitignore            # Git ignore rules
├── config/
│   ├── __init__.py       # Config module exports
│   └── settings.py       # Settings class and loader
├── utils/
│   ├── __init__.py
│   ├── data_processing.py # CSV loading and statistics
│   ├── mappings.py       # Category mapping utilities
│   └── insights.py       # Claude API integration
├── data/
│   └── category_mappings.json # Saved category mappings
└── .streamlit/
    └── config.toml       # Streamlit theme configuration
```

## License

Internal use only.
