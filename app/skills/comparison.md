version: 3

# Comparing hubs

1. Call `compare_hubs` with the hubs and any hazards the user named. Hazard groups are `winter`, `hurricane`, `flood` and `severe_storm`.
2. Compare hazard by hazard, using each hub's EDD and national rank.
3. State which hub is more exposed on each hazard, giving each hub's figure and the gap from `difference_by_hazard`, then give the overall picture. With more than two hubs, that gap is highest minus lowest, so say so. Never state any other gap, ratio or percentage between hubs, because no tool returns one.
4. EDD counts closure days, not damage. When hurricanes are involved, say that a single storm can cost far more than its closure days suggest.
5. Report any `caveat` on a compared hub.
6. For hurricane history, you may call `yearly_history` with `hurricane_declarations`. Describe those as federal declarations, never as direct hits.
