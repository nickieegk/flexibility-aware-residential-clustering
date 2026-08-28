from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = str(PROJECT_ROOT / "Data")
OUTPUT_PATH = str(PROJECT_ROOT / "Outputs")

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
