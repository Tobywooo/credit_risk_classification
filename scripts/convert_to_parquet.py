
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "credit_risk_dataset_v2.xlsx"
OUT_DATA = ROOT / "data" / "credit_risk_dataset_v2.parquet"
OUT_DICT = ROOT / "data" / "credit_risk_data_dictionary.parquet"


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric columns to the smallest dtype that holds their range."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            # float32 is safe here: XGBoost casts to float32 anyway.
            # Guard against values outside float32 range just in case.
            if s.abs().max() < np.finfo(np.float32).max:
                out[col] = s.astype(np.float32)
    return out


def mb(x) -> float:
    return x / 1024 ** 2


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Not found: {SRC}")

    print(f"{SRC.name} ({mb(SRC.stat().st_size):.0f} MB)...")

    t = time.time()
    df = pd.read_excel(SRC, sheet_name="Customer Data")
    dd = pd.read_excel(SRC, sheet_name="Data Dictionary")
    print(f"  read in {time.time() - t:.0f}s -> {df.shape[0]:,} rows x {df.shape[1]} cols")

    before = df.memory_usage(deep=True).sum()
    df = downcast(df)
    after = df.memory_usage(deep=True).sum()
    print(f"  memory {mb(before):.0f} MB -> {mb(after):.0f} MB "
          f"({100 * (1 - after / before):.0f}% smaller)")

    t = time.time()
    df.to_parquet(OUT_DATA, compression="snappy", index=False)
    dd.to_parquet(OUT_DICT, index=False)
    print(f"  wrote parquet in {time.time() - t:.0f}s")

    print(f"\n{OUT_DATA.name}: {mb(OUT_DATA.stat().st_size):.0f} MB")
    print(f"{OUT_DICT.name}: {mb(OUT_DICT.stat().st_size):.1f} MB")
    
    t = time.time()
    check = pd.read_parquet(OUT_DATA)
    print(f"\nRe-read in {time.time() - t:.2f}s  shape {check.shape}")
    assert check.shape == df.shape, "shape mismatch after round trip"
    assert list(check.columns) == list(df.columns), "column mismatch after round trip"
    print("Round trip OK.")


if __name__ == "__main__":
    main()
