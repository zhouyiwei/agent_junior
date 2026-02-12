from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Supported intent categories for routing."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    REASONING = "reasoning"
    CODING = "coding"


INTENT_DESCRIPTIONS = {
    QueryIntent.SIMPLE: "A short, straightforward question with a single factual answer and minimal reasoning.",
    QueryIntent.MODERATE: "A conceptual explanation request that needs some detail but not deep technical rigor.",
    QueryIntent.COMPLEX: "A highly technical prompt requiring detailed theory, formalism, or advanced depth.",
    QueryIntent.REASONING: "A puzzle-like or multi-step logic question where careful reasoning is essential.",
    QueryIntent.CODING: "A request to design or implement code, algorithms, or debugging steps.",
}


class IntentClassification(BaseModel):
    """Classify the user's query intent using the defined intent categories."""
    intent: QueryIntent = Field(
        description=(
            "Intent category that best matches the query. "
            + " ".join([f"{k.value}: {v}" for k, v in INTENT_DESCRIPTIONS.items()])
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the intent classification (0.0 = not confident, 1.0 = very confident).",
    )
    reasoning: str = Field(
        description="Brief justification for the chosen intent."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "intent": "coding",
                    "confidence": 0.82,
                    "reasoning": "The user asks for code changes and debugging."
                }
            ]
        }
    }


class MissionCriticality(BaseModel):
    """Score how critical it is to be correct (0.0 = low stakes, 1.0 = high stakes)."""
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = low stakes, 1.0 = critical, must be correct",
    )
    reasoning: str = Field(
        description="Brief justification why this query is low or high stakes."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "score": 0.85,
                    "reasoning": "This is medical advice; errors could cause harm."
                }
            ]
        }
    }


class LatencyCriticality(BaseModel):
    """Score how time-sensitive the response is (0.0 = can wait, 1.0 = needs instant response)."""
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = can wait, 1.0 = needs instant response",
    )
    reasoning: str = Field(
        description="Brief justification why this query needs a fast response or can wait."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "score": 0.25,
                    "reasoning": "User asked for a detailed analysis; speed is less important."
                }
            ]
        }
    }


class RoutingDecision(BaseModel):
    """Select the best model_key and deployment given intent, criticality and time-sensitivity signals."""
    model_key: str = Field(
        description="Key from the model registry (e.g., gemma-3-4b)."
    )
    model_tier: str = Field(
        description="Selected model tier (small, medium, large, reasoning). Must match the chosen model_key tier."
    )
    deployment: Literal["edge", "cloud"] = Field(
        description="Deployment target."
    )
    reasoning: str = Field(
        description="Brief justification why this model/deployment was selected."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_key": "gemma-3-4b",
                    "model_tier": "small",
                    "deployment": "edge",
                    "reasoning": "high mission + low speed demands score indicates LARGE/REASONING on cloud."
                }
            ]
        }
    }
