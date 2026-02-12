"""
Custom Router Implementation.

================================================================================
CANDIDATE INSTRUCTIONS
================================================================================

This is the file where you should implement your final routing strategy.
Extend the BaseRouter class and implement the required methods.

You may add additional files in this solutions/ folder as needed:
- Helper modules (e.g., classifier.py, features.py)
- Data files (e.g., training data, lookup tables)
- Tests for your implementation

However, your FINAL router implementation must be in this file (custom_router.py)
and the main class must be named `CustomRouter`.

================================================================================
GOAL
================================================================================

Create a router that intelligently routes queries to the best model/deployment
combination based on query characteristics, optimizing for quality, latency, and cost.

================================================================================
BASELINE LIMITATIONS TO ADDRESS (pick one or more)
================================================================================

1. Query complexity detection - route simple queries to fast/cheap models
2. Intent classification - match query type to model strengths
3. Cost optimization - minimize cost while meeting quality thresholds
4. Latency optimization - meet latency SLAs by smart edge/cloud decisions
5. Quality optimization - maximize quality within budget constraints
6. Adaptive routing - learn from past performance
7. Fallback strategies - handle model failures gracefully

================================================================================
AVAILABLE RESOURCES
================================================================================

- MODEL_REGISTRY: Dict of model configurations (tier, costs, etc.)
- ModelTier: Enum (SMALL, MEDIUM, LARGE, REASONING)
- EDGE_COMPATIBLE_MODELS: List of models that can run on edge (SMALL tier only)
- get_latency_multiplier(): Get latency multiplier for model/deployment combo

================================================================================
CONSTRAINTS
================================================================================

- Only SMALL tier models can be deployed on edge
- Edge deployment has 0.2x latency multiplier (5x faster than cloud)
- Cloud deployment has 1.0x latency multiplier (baseline)
"""

from typing import Optional, List, Tuple, Dict, Any, TypedDict

from src.router import BaseRouter
from src.model_registry import MODEL_REGISTRY, ModelTier
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import RateLimitError
import time
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

from .models import (
    QueryIntent,
    IntentClassification,
    LatencyCriticality,
    MissionCriticality,
    RoutingDecision,
    INTENT_DESCRIPTIONS,
)


class CustomRouter(BaseRouter):
    """
    Custom router implementation for intelligent query routing.

    TODO: Implement your routing strategy here.

    Your router should decide:
    1. Which model to use for a given query
    2. Where to deploy (edge or cloud)

    Consider these factors:
    - Query complexity (simple vs complex)
    - Query type/intent (coding, reasoning, factual, creative, etc.)
    - Latency requirements
    - Cost constraints
    - Quality requirements
    """

    def __init__(
        self,
        routing_model_key: str = "gemma-3-4b",
        max_retries: int = 3,
    ):
        """
        Initialize the custom router.

        TODO: Implement initialization logic.

        Consider initializing:
        - Configuration parameters
        - Query classifier (if using ML-based classification)
        - Model performance cache (if doing adaptive routing)
        - Any pre-computed routing rules
        """
        super().__init__()

        if routing_model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown routing model: {routing_model_key}")
        if MODEL_REGISTRY[routing_model_key].tier != ModelTier.SMALL:
            raise ValueError("Routing agents must use SMALL tier models")

        self.routing_model_key = routing_model_key
        self.routing_model_id = MODEL_REGISTRY[routing_model_key].model_id

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.max_retries = max_retries
        self._llm = ChatOpenAI(
            model=self.routing_model_id,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.2,
            max_tokens=500,
        )

        self.graph = self._build_routing_graph()

    @property
    def name(self) -> str:
        """
        Return a descriptive name for this router.

        TODO: Return a name that describes your routing strategy.

        Examples:
        - "ComplexityAware"
        - "IntentBased"
        - "CostOptimized"
        - "AdaptiveQuality"
        """
        return "AgenticRouting"

    def route(
        self,
        query: str,
        available_models: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Route a query to a model and deployment.

        TODO: Implement your routing logic here.

        Args:
            query: The user query to route
            available_models: Optional list of models to choose from
                            (defaults to all models if None)

        Returns:
            Tuple of (model_key, deployment) where:
            - model_key: Key from MODEL_REGISTRY (e.g., "gemma-3-4b", "llama-3.3-70b")
            - deployment: "edge" or "cloud"

        IMPORTANT CONSTRAINTS:
        - Only SMALL tier models can be deployed on "edge"
        - All models can be deployed on "cloud"
        - If you return an invalid combination, the benchmark will fail
        """
        self.call_count += 1

        result = self.graph.invoke({"query": query, "available_models": available_models})
        decision = result["routing_decision"]
        self.routing_history.append((query, decision.model_key, decision.deployment))
        return (decision.model_key, decision.deployment)

    # =========================================================================
    # LLM-POWERED PIPELINE STEPS (Task 2)
    # =========================================================================

    def classify_intent(self, query: str) -> IntentClassification:
        """Step 1: Intent classification."""
        return self._invoke_structured(
            system_prompt=_INTENT_SYSTEM_PROMPT,
            user_prompt=_format_intent_prompt(query),
            output_type=IntentClassification,
        )

    def score_mission(
        self,
        query: str,
        intent: IntentClassification,
    ) -> MissionCriticality:
        """Step 2: Mission-criticality scoring."""
        return self._invoke_structured(
            system_prompt=_MISSION_SYSTEM_PROMPT,
            user_prompt=_format_mission_prompt(query, intent),
            output_type=MissionCriticality,
        )

    def score_latency(
        self,
        query: str,
        intent: IntentClassification,
    ) -> LatencyCriticality:
        """Step 3: Latency-criticality scoring."""
        return self._invoke_structured(
            system_prompt=_LATENCY_SYSTEM_PROMPT,
            user_prompt=_format_latency_prompt(query, intent),
            output_type=LatencyCriticality,
        )

    def make_decision(
        self,
        query: str,
        intent: IntentClassification,
        mission: MissionCriticality,
        latency: LatencyCriticality,
        available_models: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """Step 4: Final routing decision."""
        allowed_models = available_models or list(MODEL_REGISTRY.keys())
        decision = self._invoke_structured(
            system_prompt=_DECISION_SYSTEM_PROMPT,
            user_prompt=_format_decision_prompt(
                query=query,
                intent=intent,
                mission=mission,
                latency=latency,
                allowed_models=allowed_models,
            ),
            output_type=RoutingDecision,
        )
        error = self._if_error_in_decision(decision, allowed_models)
        if not error:
            return decision

        retry_prompt = (
            _format_decision_prompt(
                query=query,
                intent=intent,
                mission=mission,
                latency=latency,
                allowed_models=allowed_models,
            )
            + f"\nPrevious error: {error}\nFix the decision."
        )
        retry_decision = self._invoke_structured(
            system_prompt=_DECISION_SYSTEM_PROMPT,
            user_prompt=retry_prompt,
            output_type=RoutingDecision,
        )
        if not self._if_error_in_decision(retry_decision, allowed_models):
            return retry_decision
        return retry_decision


    # =========================================================================
    # OPTIONAL HELPER METHODS
    # Implement these if they help your routing strategy
    # =========================================================================

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to extract features for routing decisions.

        TODO: Implement query analysis (optional but recommended).

        Suggested features to extract:
        - length: Number of characters/words/tokens
        - complexity_score: Estimated complexity (simple/moderate/complex)
        - intent: Query type (factual, reasoning, coding, creative, etc.)
        - domain: Subject area (math, science, general, etc.)
        - requires_reasoning: Whether deep reasoning is needed
        - requires_code: Whether code generation is needed

        Returns:
            Dict with query features
        """
        # TODO: Implement query analysis
        # Example:
        # return {
        #     "length": len(query),
        #     "word_count": len(query.split()),
        #     "complexity": self._estimate_complexity(query),
        #     "intent": self._classify_intent(query),
        #     "has_code_keywords": any(kw in query.lower() for kw in ["code", "function", "implement"]),
        # }

        pass

    def _estimate_complexity(self, query: str) -> str:
        """
        Estimate query complexity.

        TODO: Implement complexity estimation (optional).

        Approaches:
        - Rule-based: keyword matching, length thresholds
        - ML-based: trained classifier
        - Hybrid: rules + simple heuristics

        Returns:
            One of: "simple", "moderate", "complex"
        """
        # TODO: Implement complexity estimation
        # Example rule-based approach:
        return "moderate"

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _invoke_structured(self, system_prompt: str, user_prompt: str, output_type):
        schema = output_type.model_json_schema()
        fields = ", ".join(schema.get("properties", {}).keys())
        merged_prompt = f"""{_BASE_SYSTEM_PROMPT}
{system_prompt}

{user_prompt}

Return ONLY valid JSON with fields: {fields}.
Do not return the schema. Return an instance object.
"""
        messages = [HumanMessage(content=merged_prompt)]
        llm = self._llm
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = llm.invoke(messages)
                return _parse_json_response(getattr(result, "content", str(result)), output_type)
            except Exception as exc:
                last_error = exc
                if _is_rate_limit_error(exc):
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    retry_prompt = (
                        f"{merged_prompt}\n\nPrevious error: {last_error}\n"
                        "Fix the output to satisfy the schema."
                    )
                    retry_result = llm.invoke([HumanMessage(content=retry_prompt)])
                    return _parse_json_response(
                        getattr(retry_result, "content", str(retry_result)), output_type
                    )
        raise RuntimeError(f"_invoke_structured failed after {self.max_retries} retries: {last_error}")



    @staticmethod
    def _if_error_in_decision(
        decision: RoutingDecision,
        allowed_models: List[str],
    ):
        if not decision:
            return "You must select one model from the allowed_models list."
        model_key = decision.model_key
        deployment = decision.deployment
        model_tier = decision.model_tier

        if model_key not in allowed_models:
            return "model_key is not from the allowed_models list."
        if deployment not in ("edge", "cloud"):
            return "deployment is neither edge or cloud."
        model_config = MODEL_REGISTRY.get(decision.model_key)
        if not model_config:
            return "model_key not found in model registry."
        if decision.deployment == "edge" and model_config.tier != ModelTier.SMALL:
            return "Only SMALL tier models can be deployed on edge."
        if decision.deployment == "cloud" and model_config.tier == ModelTier.SMALL:
            return "Small tier models have to be deployed on EDGE."
        if model_tier != model_config.tier.value:
            return ("Your selected model does not match to your selected model_tier. "
                    "Make a different model selection based on your model tier.")
        return None

    # =========================================================================
    # GRAPH (Task 3)
    # =========================================================================
    class RouterState(TypedDict, total=False):
        query: str
        available_models: Optional[List[str]]
        intent: IntentClassification
        mission_criticality: MissionCriticality
        latency_criticality: LatencyCriticality
        routing_decision: RoutingDecision

    def _build_routing_graph(self):
        graph = StateGraph(CustomRouter.RouterState)

        def classify_intent_node(state: CustomRouter.RouterState) -> Dict[str, Any]:
            intent = self.classify_intent(state["query"])
            return {"intent": intent}

        def score_mission_node(state: CustomRouter.RouterState) -> Dict[str, Any]:
            mission = self.score_mission(state["query"], state["intent"])
            return {"mission_criticality": mission}

        def score_latency_node(state: CustomRouter.RouterState) -> Dict[str, Any]:
            latency = self.score_latency(state["query"], state["intent"])
            return {"latency_criticality": latency}

        def make_decision_node(state: CustomRouter.RouterState) -> Dict[str, Any]:
            decision = self.make_decision(
                query=state["query"],
                intent=state["intent"],
                mission=state["mission_criticality"],
                latency=state["latency_criticality"],
                available_models=state.get("available_models"),
            )
            return {"routing_decision": decision}

        graph.add_node("classify_intent", classify_intent_node)
        graph.add_node("score_mission", score_mission_node)
        graph.add_node("score_latency", score_latency_node)
        graph.add_node("make_decision", make_decision_node)

        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "score_mission")
        graph.add_edge("classify_intent", "score_latency")
        graph.add_edge("score_mission", "make_decision")
        graph.add_edge("score_latency", "make_decision")

        return graph.compile()


    def _classify_intent(self, query: str) -> str:
        """
        Classify the intent/type of query.

        TODO: Implement intent classification (optional).

        Suggested categories:
        - "factual": Simple fact lookup
        - "reasoning": Requires logical reasoning
        - "coding": Code generation/debugging
        - "creative": Creative writing
        - "analysis": Data/text analysis
        - "math": Mathematical problems

        Returns:
            Intent category string
        """
        # TODO: Implement intent classification
        # Example keyword-based approach:
        # query_lower = query.lower()
        # if any(kw in query_lower for kw in ["write code", "implement", "function"]):
        #     return "coding"
        # elif any(kw in query_lower for kw in ["why", "explain", "reason"]):
        #     return "reasoning"
        # else:
        #     return "factual"

        pass

    def _select_tier(self, query_features: Dict[str, Any]) -> ModelTier:
        """
        Select the appropriate model tier based on query features.

        TODO: Implement tier selection logic (optional).

        Guidelines:
        - SMALL: Simple queries, low latency requirements
        - MEDIUM: Moderate complexity, balanced quality/speed
        - LARGE: Complex queries requiring high quality
        - REASONING: Queries requiring deep reasoning/math

        Args:
            query_features: Features extracted from _analyze_query()

        Returns:
            ModelTier enum value
        """
        # TODO: Implement tier selection
        # Example:
        # complexity = query_features.get("complexity", "moderate")
        # intent = query_features.get("intent", "factual")
        #
        # if intent == "reasoning" or complexity == "complex":
        #     return ModelTier.REASONING
        # elif complexity == "simple":
        #     return ModelTier.SMALL
        # else:
        #     return ModelTier.MEDIUM

        pass

    def _select_model(
        self,
        tier: ModelTier,
        available_models: Optional[List[str]] = None
    ) -> str:
        """
        Select a specific model from the target tier.

        TODO: Implement model selection within tier (optional).

        Considerations:
        - Cost: Choose cheaper model if quality requirements are met
        - Availability: Filter by available_models if provided
        - Specialization: Some models may be better for certain tasks

        Args:
            tier: Target model tier
            available_models: Optional filter list

        Returns:
            model_key from MODEL_REGISTRY
        """
        # TODO: Implement model selection
        # Example:
        # tier_models = get_models_by_tier(tier)
        # if available_models:
        #     tier_models = [m for m in tier_models if m in available_models]
        #
        # if not tier_models:
        #     # Fallback to any available model
        #     return available_models[0] if available_models else "gemma-3-4b"
        #
        # # Select cheapest model in tier
        # return min(tier_models, key=lambda m: MODEL_REGISTRY[m].cost_per_million_input)

        pass

    def update_from_feedback(
        self,
        query: str,
        model_key: str,
        deployment: str,
        quality_score: float,
        latency_ms: float
    ) -> None:
        """
        Update router state based on feedback from a completed request.

        TODO: Implement adaptive learning (optional, for advanced implementations).

        This method is called after each request completes with the actual
        quality and latency observed. Use this to:
        - Track model performance over time
        - Adjust routing decisions based on observed outcomes
        - Implement online learning

        Args:
            query: The original query
            model_key: Model that was used
            deployment: Deployment that was used
            quality_score: Quality score (0-10) from evaluation
            latency_ms: Actual latency in milliseconds
        """
        # TODO: Implement feedback processing for adaptive routing
        # Example:
        # key = f"{model_key}@{deployment}"
        # if key not in self.model_performance_cache:
        #     self.model_performance_cache[key] = {"quality": [], "latency": []}
        # self.model_performance_cache[key]["quality"].append(quality_score)
        # self.model_performance_cache[key]["latency"].append(latency_ms)

        pass


# =============================================================================
# PROMPTS
# =============================================================================

_BASE_SYSTEM_PROMPT = """You are a routing assistant."""

_INTENT_SYSTEM_PROMPT = """Given a query, classify its intent.
Consider:
- What kind of task the query requires?
"""

_MISSION_SYSTEM_PROMPT = """Score how critical it is to get a correct answer.
Consider:
- Is the user making a decision based on this?
- Could a wrong answer cause harm?
- Is this a factual question with a definitive answer?
- Is this casual/exploratory?
> 0.7 for high stakes queries, < 0.3 for low stakes queries
"""

_LATENCY_SYSTEM_PROMPT = """Given a query and its intent, score how time-sensitive the response is. 
Consider:
- Is this a quick lookup or a deep analysis?
- Would the user expect an instant response?
- Is latency more important than thoroughness?
> 0.7 for latency-critical queries, < 0.3 for queries that do not need a quick response
"""

_DECISION_SYSTEM_PROMPT = """Given the intent, mission score, and speed score, select the best (model_key, model_tier, deployment)
combination from the model registry.

You MUST apply exactly ONE routing rule below, in order, based only on the score bands:
1) high mission + low speed demand -> LARGE or REASONING tier, deployment = cloud
2) low mission + high speed demand -> SMALL tier, deployment = edge
3) high mission + high speed demand -> MEDIUM tier, deployment = cloud
4) low mission + low speed demand -> SMALL tier, deployment = edge
5) otherwise -> MEDIUM tier, deployment = cloud

Do NOT invent new rules. Do NOT contradict the rules. You MUST choose a model whose tier matches the rule’s tier.
A score is high if > 0.7, a score is low if < 0.3.

Selection procedure (follow exactly):
1) Determine if the mission and speed demand scores are high or low.
2) Determine the tier from the rule.
3) Filter Allowed Models to that tier only.
4) From that filtered list, select a model that is most like to have good quality and low cost.

Reasoning MUST include:
- Rule number used
- Tier required by the rule
- The chosen model’s tier (as shown in the Allowed Models list)
- A one-line confirmation that the tier matches the rule
"""


def _format_intent_prompt(query: str) -> str:
    intent_values = ", ".join([e.value for e in QueryIntent])
    intent_desc = " ".join(
        [f"{k.value}: {v}" for k, v in INTENT_DESCRIPTIONS.items()]
    )
    return f"""{intent_desc}

Use only these intent values:
{intent_values}.

Query:
{query}
"""


def _format_mission_prompt(query: str, intent: IntentClassification) -> str:
    intent_desc = INTENT_DESCRIPTIONS.get(intent.intent, intent.intent.value)
    return f"""Use a numeric score between 0.0 and 1.0. The higher the score is, the more critical it is to be correct.
Consider the query intent when scoring.

Query:
{query}

Intent:
{intent.intent}

Intent description:
{intent_desc}
"""


def _format_latency_prompt(query: str, intent: IntentClassification) -> str:
    intent_desc = INTENT_DESCRIPTIONS.get(intent.intent, intent.intent.value)
    return f"""Use a numeric score between 0.0 and 1.0. The higher the score is, the more quickly the customer needs 
    a response.
Consider the query intent when scoring.

Query:
{query}

Intent:
{intent.intent}

Intent description:
{intent_desc}
"""


def _format_decision_prompt(
    query: str,
    intent: IntentClassification,
    mission: MissionCriticality,
    latency: LatencyCriticality,
    allowed_models: List[str],
) -> str:
    tier_groups = {t: [] for t in ModelTier}
    allowed_set = set(allowed_models)
    for key, config in MODEL_REGISTRY.items():
        if key not in allowed_set:
            continue
        tier_groups[config.tier].append(key)
    tier_lines = []
    for tier in ModelTier:
        keys = tier_groups[tier]
        if keys:
            tier_lines.append(f"- {tier.value.upper()}: {', '.join(keys)}")
    tier_list_str = "\n".join(tier_lines) if tier_lines else "(none)"
    intent_desc = INTENT_DESCRIPTIONS.get(intent.intent, intent.intent.value)
    return f"""Choose only from the allowed models listed below.

Query:
{query}

Intent: {intent.intent}
Intent description: {intent_desc}
Mission score: {mission.score}
Speed demand score: {latency.score}

Allowed models grouped by tier (you MUST choose from the tier implied by the rule):
{tier_list_str}

Selection rule reminder:
- Determine the tier from the routing rule.
- Choose ONLY from that tier's list above.
"""


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    message = str(exc).lower()
    return "rate limit" in message or "429" in message


def _parse_json_response(text: str, model_cls):
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return model_cls.model_validate_json(cleaned)


# =============================================================================
# ALTERNATIVE ROUTER TEMPLATES
# Uncomment and modify one of these if it better fits your approach
# =============================================================================

# class ComplexityAwareRouter(BaseRouter):
#     """Router that routes based on query complexity."""
#
#     def __init__(self, complexity_threshold: int = 50):
#         super().__init__()
#         self.complexity_threshold = complexity_threshold
#
#     @property
#     def name(self) -> str:
#         return "ComplexityAware"
#
#     def route(self, query: str, available_models: Optional[List[str]] = None) -> Tuple[str, str]:
#         self.call_count += 1
#         # Simple heuristic: longer queries are more complex
#         if len(query) < self.complexity_threshold:
#             model_key, deployment = "gemma-3-4b", "edge"
#         else:
#             model_key, deployment = "llama-3.3-70b", "cloud"
#         self.routing_history.append((query[:50], model_key, deployment))
#         return (model_key, deployment)


# class IntentBasedRouter(BaseRouter):
#     """Router that routes based on query intent."""
#
#     INTENT_MODEL_MAP = {
#         "coding": "llama-3.3-70b",
#         "reasoning": "deepseek-r1-0528",
#         "simple": "gemma-3-4b",
#         "default": "mistral-small-24b",
#     }
#
#     def __init__(self):
#         super().__init__()
#
#     @property
#     def name(self) -> str:
#         return "IntentBased"
#
#     def _detect_intent(self, query: str) -> str:
#         query_lower = query.lower()
#         if any(kw in query_lower for kw in ["code", "function", "implement", "debug"]):
#             return "coding"
#         elif any(kw in query_lower for kw in ["prove", "derive", "calculate", "solve"]):
#             return "reasoning"
#         elif len(query.split()) < 10:
#             return "simple"
#         return "default"
#
#     def route(self, query: str, available_models: Optional[List[str]] = None) -> Tuple[str, str]:
#         self.call_count += 1
#         intent = self._detect_intent(query)
#         model_key = self.INTENT_MODEL_MAP.get(intent, "mistral-small-24b")
#         deployment = "edge" if MODEL_REGISTRY[model_key].tier == ModelTier.SMALL else "cloud"
#         self.routing_history.append((query[:50], model_key, deployment))
#         return (model_key, deployment)
