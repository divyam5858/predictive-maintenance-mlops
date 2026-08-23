import pandas as pd
from pathlib import Path


def load_sensor_data() -> pd.DataFrame:
    """Load the raw predictive maintenance sensor dataset."""

    data_path = Path("data/raw/sensor_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    return df