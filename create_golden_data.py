import pandas as pd
import numpy as np
from pathlib import Path

def generate_pareto_dataset():
    """
    Generates a controlled dataset for Pareto (80/20) analysis testing.
    Total rows: 200.
    Ground Truth: Top 20% of customers generate ~80% of revenue.
    """
    # Create directory if missing
    output_dir = Path("tests/golden/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_total = 200
    n_top = 40  # 20%
    n_bottom = 160  # 80%

    # 1. Top 20% - High spenders (Total ~80,000)
    top_customers = [f"VIP_{i:02d}" for i in range(n_top)]
    top_sales = rng.normal(2000, 100, n_top)

    # 2. Bottom 80% - Small spenders (Total ~20,000)
    bottom_customers = [f"REG_{i:03d}" for i in range(n_bottom)]
    bottom_sales = rng.normal(125, 20, n_bottom)

    # Combine
    data = {
        "customer_id": top_customers + bottom_customers,
        "sales_amount": np.concatenate([top_sales, bottom_sales]),
        "region": rng.choice(["North", "South", "East", "West"], n_total)
    }

    df = pd.DataFrame(data)

    # Shuffle to make it realistic
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    file_path = output_dir / "sales_pareto.csv"
    df.to_csv(file_path, index=False)

    # Verification
    total_rev = df['sales_amount'].sum()
    top_rev = df[df['customer_id'].str.startswith("VIP")]['sales_amount'].sum()
    print(f"Dataset created: {file_path}")
    print(f"Actual Concentration: {top_rev/total_rev:.2%} of revenue from 20% of customers.")


def generate_high_nulls_dataset():
    """Generate a deterministic dataset with a sparse campaign field."""
    output_dir = Path("tests/golden/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "high_nulls.csv"

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(1, 13)],
            "sales_amount": [1200, 980, 1450, 1100, 870, 1320, 1040, 760, 1510, 920, 1180, 990],
            "region": ["North", "South", "East", "West"] * 3,
            "campaign_code": [None] * 12,
        }
    )
    df.to_csv(file_path, index=False)
    print(f"Dataset created: {file_path}")

if __name__ == "__main__":
    generate_pareto_dataset()
    generate_high_nulls_dataset()
