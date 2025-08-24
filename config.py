import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# TennisAbstract.com configuration
TENNIS_ABSTRACT_BASE_URL = "https://tennisabstract.com"
TENNIS_ABSTRACT_CHARTING_URL = "https://tennisabstract.com/charting/"

# Output configuration
OUTPUT_DIR = "analysis_output"
REPORTS_DIR = "reports"
CHARTS_DIR = "charts"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, REPORTS_DIR, CHARTS_DIR]:
    os.makedirs(directory, exist_ok=True) 