from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = DATA_DIR / "processed"
FIGURES_DIR = DATA_DIR / "processed" / "evaluation"

RANDOM_STATE = 42
N_SIMULATIONS = 1000

KAGGLE_MATCHES_DATASET = "martj42/international-football-results-from-1872-to-2017"
KAGGLE_RANKINGS_DATASET = "cashncarry/fifaworldranking"

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php"
WC2026_WIKI_PAGE = "2026_FIFA_World_Cup"
FIFA_RANKINGS_WIKI_PAGE = "FIFA_Men%27s_World_Ranking"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

HOST_NATIONS = ["United States", "Canada", "Mexico"]

N_GROUPS = 12
GROUP_LETTERS = [chr(ord("A") + i) for i in range(N_GROUPS)]
TEAMS_PER_GROUP = 4
ADVANCE_PER_GROUP = 2
BEST_THIRD_ADVANCE = 8

K_FACTORS = {
    "FIFA World Cup": 80,
    "FIFA World Cup qualification": 60,
    "Friendly": 40,
    "default": 50,
}

ELO_HOME_ADVANTAGE = 100
ELO_INITIAL_RATING = 1000
ELO_DRAW_FACTOR = 0.30

WC_GROUP_DRAW_RATE = 0.25

TEAM_NAME_MAPPING = {
    "USA": "United States",
    "USMNT": "United States",
    "U.S.A.": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Islamic Republic of Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "China PR": "China",
    "Curacao": "Curaçao",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Lucia": "Saint Lucia",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "North Macedonia": "FYR Macedonia",
    "Eswatini": "Swaziland",
    "Congo": "Congo Republic",
    "DR Congo": "Congo DR",
}

CONFEDERATIONS = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL", "Guyana": "CONMEBOL", "Suriname": "CONMEBOL",
    "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Italy": "UEFA", "Netherlands": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Serbia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA",
    "Denmark": "UEFA", "Poland": "UEFA", "Sweden": "UEFA", "Ukraine": "UEFA",
    "Turkey": "UEFA", "Czech Republic": "UEFA", "Scotland": "UEFA",
    "Wales": "UEFA", "Slovakia": "UEFA", "Romania": "UEFA", "Greece": "UEFA",
    "Hungary": "UEFA", "Republic of Ireland": "UEFA", "Norway": "UEFA",
    "Bosnia-Herzegovina": "UEFA", "Finland": "UEFA", "Israel": "UEFA",
    "Russia": "UEFA", "Georgia": "UEFA", "Albania": "UEFA", "Iceland": "UEFA",
    "Montenegro": "UEFA", "North Macedonia": "UEFA", "Slovenia": "UEFA",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Panama": "CONCACAF",
    "Honduras": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Haiti": "CONCACAF", "El Salvador": "CONCACAF", "Guatemala": "CONCACAF",
    "Curaçao": "CONCACAF", "Martinique": "CONCACAF", "Suriname": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "China": "AFC", "Qatar": "AFC", "Uzbekistan": "AFC",
    "Iraq": "AFC", "United Arab Emirates": "AFC", "Jordan": "AFC",
    "Oman": "AFC", "Syria": "AFC", "Bahrain": "AFC", "Kuwait": "AFC",
    "Palestine": "AFC", "Lebanon": "AFC", "Vietnam": "AFC", "Thailand": "AFC",
    "India": "AFC", "North Korea": "AFC", "Tajikistan": "AFC",
    "Nigeria": "CAF", "Senegal": "CAF", "Ghana": "CAF", "Cameroon": "CAF",
    "Algeria": "CAF", "Morocco": "CAF", "Tunisia": "CAF", "Egypt": "CAF",
    "South Africa": "CAF", "Mali": "CAF", "Ivory Coast": "CAF",
    "Congo DR": "CAF", "Burkina Faso": "CAF", "Guinea": "CAF",
    "Zambia": "CAF", "Cape Verde": "CAF", "Kenya": "CAF",
    "New Zealand": "OFC", "Fiji": "OFC", "Papua New Guinea": "OFC",
    "Solomon Islands": "OFC", "Tahiti": "OFC",
}

OUTCOME_HOME_WIN = 1
OUTCOME_DRAW = 0
OUTCOME_AWAY_WIN = -1

OUTCOME_LABELS = {1: "home_win", 0: "draw", -1: "away_win"}

NEURAL_NET_EPOCHS = 100
NEURAL_NET_PATIENCE = 10
NEURAL_NET_LAYERS = [128, 64, 32]
NEURAL_NET_DROPOUT = 0.3
NEURAL_NET_LEARNING_RATE = 1e-3

OPTUNA_TRIALS = 20
CV_FOLDS = 3