from solutions import CustomRouter
router = CustomRouter()
state = router.graph.invoke({"query": "How does a blockchain work and what are its main use cases?"})
print(state["intent"])
print(state["mission_criticality"])
print(state["latency_criticality"])
print(state["routing_decision"])