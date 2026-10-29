# Airport Delay Predictor & Station Health Score

Which US flights and airports are most often delayed, and how much of that can we predict before departure? This project combines SQL analysis, a hypothesis test, three prediction models, airport clustering, and a custom operations metric (the Station Health Score).

**Headline:** Delays build up through the day and are concentrated in evening departures and at busy airports such as Chicago (MDW 24.17%, ORD 23.92%) and Newark (EWR 23.25%). Using only information known before departure, the best model reaches a test ROC-AUC of 0.642. That is a modest signal, which is expected without weather or previous-flight data.

**Dataset:** [2015 Flight Delays and Cancellations (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays), random sample of 200,000 flights (seed 42).

## How to run
1. `pip install -r requirements.txt`
2. Download the data into `data/` (see `data/README.md`).
3. `jupyter notebook flight_ops.ipynb`, then "Restart and run all".

## Results

Model comparison (time-based split: train Jan-Sep, test Oct-Dec; delayed = departure delay > 15 minutes; class imbalance handled with class weights):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.246 | 0.467 | 0.323 | 0.634 |
| Random Forest | 0.304 | 0.204 | 0.244 | 0.642 |
| XGBoost | 0.278 | 0.245 | 0.260 | 0.638 |

**How to read this:** Random Forest ranks flights best by risk (highest ROC-AUC), but at the default 0.5 threshold it catches only about 1 in 5 actual delays (recall 0.204). Logistic Regression catches the most delays (recall 0.467, best F1) at the cost of more false alarms. Which model is "best" depends on whether an operations team prefers fewer missed delays or fewer false alarms. All three models are only moderately predictive, because only pre-departure information is used.

### Key findings
1. **Delays build up through the day.** The delay rate rises from 5.49% at 5 AM to 28.60% at 8 PM. A chi-square test shows evening flights (17-21h: 26.57% delayed, 95% CI [26.19%, 26.96%]) are delayed far more often than morning flights (5-9h: 9.21% delayed, 95% CI [8.97%, 9.45%]); chi-square = 5544.06, p < 0.001. This is a correlation: time of day is likely a proxy for delays knocking on from earlier flights, not a direct cause.
2. **Scheduled departure hour is the strongest predictor.** In the tuned Random Forest, departure hour has about 48.3% feature importance, followed by month (about 11.1%) and flight distance (about 10.5%).
3. **Airports split into two groups.** KMeans (k = 2, silhouette = 0.339, a moderate separation) gives a larger, more delay-prone group (Cluster 0: 27 airports, 20.8% mean delay rate, e.g. ATL, ORD, DFW) and a lower-delay group (Cluster 1: 39 airports, 15.6% mean delay rate, e.g. MSP, SEA, DTW).

## Station Health Score

```
score = 100 - (0.5 * delay_rate% + 2.0 * cancel_rate% + 0.3 * avg_delay_scaled)
```
The score is clipped to 0-100 (higher is healthier). Weights:
- **0.5 on delay rate:** how often flights leave more than 15 minutes late.
- **2.0 on cancellation rate:** cancellations are rare but hurt customers most, so each percentage point counts four times as much as a delay point.
- **0.3 on average delay length:** average delay in minutes, rescaled 0-100 across airports.

The weights are judgement calls and can be changed by an ops team (for example, a higher cancellation weight during holiday travel). The score does not account for weather, airport size, or passengers affected.

Healthiest stations: HNL (94.7), OGG (88.1), ANC (86.6). Least healthy: ORD (52.0), LGA (53.4), EWR (54.1).

## Limitations
- 2015 data only; patterns may have changed.
- No weather or previous-flight (knock-on delay) information, which likely explains the modest model performance.
- Sampled data (about 200k flights), not the full year.
- A "delay" is defined as more than 15 minutes of departure delay.

## Next steps
- Add weather data.
- Add previous-flight delay as a feature.
- Try real-time scoring.
