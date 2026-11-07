# Severity model key findings

- On unseen 2025 data the model reaches ROC AUC 0.65 (0.5 = guessing); gradient boosting scores 0.65, so the simpler logistic model is kept.
- The 20% of crashes the model rates riskiest contain 33% of KSI crashes.
- Strongest risk-raising factors: escooter = Yes (×3.9); motorcycle = Yes (×3.7); cyclist = Yes (×2.9); bus = Yes (×2.1); pedestrian = Yes (×2.1).
- Strongest risk-lowering factors: speed_limit = 20 mph (×0.48); surface = Ice, snow, flood / unknown (×0.51); road_class = Motorway (×0.67).
- Severity is hard to predict from scene conditions alone (speed at impact, seatbelts and vehicle type matter a lot and aren't in open data), so treat predictions as relative risk, not certainty.
