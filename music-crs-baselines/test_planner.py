from recommender_v2.planner import QueryPlanner
from recommender_v2.planner.ollama import OllamaQwenPlanner


planner = QueryPlanner(
    llm=OllamaQwenPlanner("qwen3:8b"),
    mode="llm",
)

query = "I want something more chill, similar to what you recommended before."

conversation_history = [
    {
        "role": "user",
        "content": "Recommend me some music for studying."
    },
    {
        "role": "music",
        "content": "TRACK_ID_123"
    },
]

anchor_track_ids = [
    "TRACK_ID_123"
]

state = planner.plan(
    query=query,
    conversation_history=conversation_history,
    anchor_track_ids=anchor_track_ids,
)

print("\nQUERY:")
print(query)

print("\nANCHORS:")
print(anchor_track_ids)

print("\nQUERY STATE:")
print(state.to_dict())