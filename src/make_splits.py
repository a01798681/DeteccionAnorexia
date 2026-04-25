from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    input_path = Path("data/data_train.xlsx")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)

    # División estratificada 80/20
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["class"],
        random_state=42
    )

    train_df.to_excel(output_dir / "train_split.xlsx", index=False)
    val_df.to_excel(output_dir / "validation_split.xlsx", index=False)

    print("Splits generados correctamente:")
    print(f"Train: {len(train_df)} filas")
    print(f"Validation: {len(val_df)} filas")


if __name__ == "__main__":
    main()