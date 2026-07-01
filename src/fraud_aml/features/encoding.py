from __future__ import annotations

from typing import Any

import pandas as pd


class FrequencyEncoder:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.maps: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> FrequencyEncoder:
        for col in self.columns:
            self.maps[col] = df[col].value_counts(dropna=True).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            mapping = self.maps.get(col, {})
            out[f"{col}_freq"] = df[col].map(mapping).fillna(0).astype("float32")
        return out


class TargetEncoder:
    def __init__(self, columns: list[str], *, smoothing: float = 20.0) -> None:
        self.columns = columns
        self.smoothing = smoothing
        self.prior = 0.0
        self.maps: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, target: Any) -> TargetEncoder:
        self.prior = float(target.mean())
        for col in self.columns:
            stats = target.groupby(df[col]).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.prior * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.maps[col] = smoothed.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            mapping = self.maps.get(col, {})
            out[f"{col}_te"] = df[col].map(mapping).fillna(self.prior).astype("float32")
        return out
