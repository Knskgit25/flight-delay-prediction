# Airport Delay Predictor & Station Health Score

Delays are heavily concentrated in late afternoon and evening departures and at congested hub stations like Chicago (MDW 24.17%, ORD 23.92%) and Newark (EWR 23.25%). Using only pre-departure features known prior to scheduled departure, our model achieves a test ROC-AUC of 0.642 to anticipate flight delay risk.

**Dataset:** [2015 Flight Delays and Cancellations (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays), random sample of 200,000 flights (seed 42).

## How to run
1. `pip install -r requirements.txt`
2. Download the data into `data/` (see `data/README.md`).
3. `jupyter notebook flight_ops.ipynb` then "Restart and run all".

## Results
Model comparison (time-based split: train Jan-Sep, test Oct-Dec; delayed = departure delay > 15 min):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.246 | 0.467 | 0.323 | 0.634 |
| Random Forest | 0.304 | 0.204 | 0.244 | 0.642 |
| XGBoost | 0.278 | 0.245 | 0.260 | 0.638 |

### Key findings
1. Delay rates compound continuously over the day from 5.49% at 5 AM to 28.60% at 8 PM; a chi-square test confirms evening flights (17–21h, 26.57% delay rate, 95% CI [26.19%, 26.96%]) have a statistically significantly higher delay rate than morning flights (5–9h, 9.21% delay rate, 95% CI [8.97%, 9.45%]) with $\chi^2 = 5544.06$ and $p = 0.0$.
2. Scheduled departure hour is the single most dominant predictor (~48.3% feature importance), followed by month (~11.1%) and flight distance (~10.5%), with hyperparameter-tuned Random Forest achieving the best test ROC-AUC of 0.642.
3. KMeans clustering ($k=2$, silhouette = 0.339) effectively segments airports into large, delay-prone hub complexes (Cluster 0: 27 airports, 20.8% mean delay rate, e.g., ATL, ORD, DFW) versus high-reliability regional stations (Cluster 1: 39 airports, 15.6% mean delay rate, e.g., MSP, SEA, DTW).

### Station Health Score
`score = 100 - (0.5 * delay_rate% + 2.0 * cancel_rate% + 0.3 * avg_delay_scaled)`, clipped to 0-100.
Weights penalize frequent delays (0.5), severe customer-impacting cancellations (2.0, fourfold weight), and normalized delay duration (0.3); the top 3 healthiest stations are HNL (94.7), OGG (88.1), and ANC (86.6), while the bottom 3 least healthy are ORD (52.0), LGA (53.4), and EWR (54.1).

## Limitations
- 2015 data only; patterns may have changed.
- No weather or previous-flight (knock-on delay) information.
- Sampled data (~200k flights), not the full year.
- A "delay" is defined as more than 15 minutes of departure delay.

## Next steps
- Add weather data.
- Add previous-flight delay as a feature.
- Try real-time scoring.
