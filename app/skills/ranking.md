version: 2

# Ranking hubs

1. Call `rank_hubs`, using region and hazard filters that match the question. For "winter disruption", use the `winter` group. Use a component such as `ice_storm` only when the user asks for it.
2. Lead with the answer: the top hubs and their Expected Disruption Days (EDD) per year.
3. `rank_hubs` doesn't break scores down, so call `compare_hubs` on the top hubs, then say what separates them using the hazards behind their scores.
4. Report rank ranges. If two hubs' ranges overlap, say their order is not firm.
5. Mention catastrophic flags on any ranked hub, and say these are separate from EDD.
6. Report any `caveat` on a ranked hub.
7. Remind the reader once that closure-day values are assumptions.
