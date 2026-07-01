from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud_aml.config import Settings
from fraud_aml.decision import Decision, stricter
from fraud_aml.features.build import add_all
from fraud_aml.fraud_model.dataset import assemble_features
from fraud_aml.serving.rules import apply_rules
from fraud_aml.serving.schemas import ScoreRequest, ScoreResponse

_ONLINE_KEYS = ("win_count", "win_sum", "win_velocity", "win_delta")


class ScoringService:
    def __init__(self, bundle: dict[str, Any], settings: Settings, feast_store: Any = None) -> None:
        self.model = bundle["model"]
        self.freq = bundle["freq"]
        self.te = bundle["te"]
        self.feature_order = bundle["feature_order"]
        self.meta = bundle["meta"]
        self.settings = settings
        self.feast_store = feast_store

    def online_features(self, card1: str) -> dict[str, float]:
        if self.feast_store is None:
            return {}
        from fraud_aml.streaming.online_store import read_online

        raw = read_online(self.feast_store, card1)
        out: dict[str, float] = {}
        for key in _ONLINE_KEYS:
            value = raw.get(key)
            if isinstance(value, list):
                value = value[0] if value else None
            if value is not None:
                out[key] = float(value)
        return out

    def build_vector(self, request: ScoreRequest, online: dict[str, float]) -> pd.DataFrame:
        base: dict[str, Any] = {
            "TransactionID": request.TransactionID,
            "card1": request.card1,
            "TransactionAmt": request.TransactionAmt,
            "TransactionDT": request.TransactionDT,
            **request.features,
        }
        frame = add_all(pd.DataFrame([base]))
        if "win_delta" in online:
            frame["card_dt_delta"] = float(online["win_delta"])
        return assemble_features(frame, self.freq, self.te, self.feature_order)

    def _ml_decision(self, proba: float) -> Decision:
        if proba >= self.settings.block_score:
            return "block"
        if proba >= self.settings.review_score:
            return "review"
        return "allow"

    def score(self, request: ScoreRequest) -> ScoreResponse:
        online = self.online_features(str(request.card1))
        frame = self.build_vector(request, online)
        proba = float(self.model.predict_proba(frame)[:, 1][0])
        contributions = np.asarray(self.model.booster_.predict(frame, pred_contrib=True))[0][:-1]
        top = int(np.argmax(np.abs(contributions)))
        direction = "increases" if contributions[top] > 0 else "decreases"
        top_reason = f"{self.feature_order[top]} ({direction} risk)"

        rule = apply_rules(online, self.settings)
        decision = stricter(self._ml_decision(proba), rule.forced) or self._ml_decision(proba)
        version: str = self.meta.get("model_version", self.settings.model_version_label)
        return ScoreResponse(
            score=proba,
            decision=decision,
            fired_rules=rule.fired,
            top_reason=top_reason,
            model_version=version,
        )
