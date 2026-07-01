from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fraud_aml.decision import Decision


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    TransactionID: int
    card1: int
    TransactionAmt: float = Field(gt=0)
    TransactionDT: int = Field(ge=0)
    features: dict[str, Any] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    score: float
    decision: Decision
    fired_rules: list[str]
    top_reason: str
    model_version: str
