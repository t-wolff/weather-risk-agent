version: 4

# Explaining a hub's score

1. Call `explain_hub`.
2. Check the question's premise against the result. If the user says a risk is high and the data doesn't show that, correct it plainly.
3. Name the two or three hazards that contribute most, with their closure days before blending and their annual frequencies. These don't add up to the hub's EDD.
4. Explain the arithmetic in plain words: how often it happens, multiplied by assumed closure days per event, blended with recent observed days. Quote the `blend` weights exactly as the tool returns them, never as percentages, and call them the weights on FEMA history and on observed days.
5. Give the tier, the tier thresholds, the rank and the rank range.
6. Mention catastrophic flags separately from EDD, and report any `caveat`.
