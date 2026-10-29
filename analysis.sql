-- name: q1_delay_by_airline
SELECT a.AIRLINE AS airline,
       COUNT(*) AS flights,
       ROUND(100.0 * AVG(f.delayed), 2) AS delay_rate_pct
FROM flights f
JOIN airlines a ON f.airline = a.IATA_CODE
WHERE f.cancelled = 0
GROUP BY a.AIRLINE
ORDER BY delay_rate_pct DESC;

-- name: q2_delay_by_hour
SELECT sched_hour AS hour,
       COUNT(*) AS flights,
       ROUND(100.0 * AVG(delayed), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
GROUP BY sched_hour
ORDER BY sched_hour;

-- name: q3_worst_origin_airports
SELECT origin,
       COUNT(*) AS flights,
       ROUND(100.0 * AVG(delayed), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
GROUP BY origin
HAVING COUNT(*) >= 500
ORDER BY delay_rate_pct DESC
LIMIT 5;

-- name: q4_rank_airlines_in_top5_airports
WITH top5 AS (
    SELECT origin
    FROM flights
    WHERE cancelled = 0
    GROUP BY origin
    HAVING COUNT(*) >= 500
    ORDER BY AVG(delayed) DESC
    LIMIT 5
),
stats AS (
    SELECT f.origin, f.airline,
           COUNT(*) AS flights,
           100.0 * AVG(f.delayed) AS delay_rate_pct
    FROM flights f
    JOIN top5 t ON f.origin = t.origin
    WHERE f.cancelled = 0
    GROUP BY f.origin, f.airline
    HAVING COUNT(*) >= 30
)
SELECT origin, airline, flights,
       ROUND(delay_rate_pct, 2) AS delay_rate_pct,
       RANK() OVER (PARTITION BY origin ORDER BY delay_rate_pct DESC) AS rank_in_airport
FROM stats
ORDER BY origin, rank_in_airport;

-- name: q5_daily_delay_with_change
WITH daily AS (
    SELECT flight_date,
           100.0 * AVG(delayed) AS delay_rate_pct
    FROM flights
    WHERE cancelled = 0
    GROUP BY flight_date
)
SELECT flight_date,
       ROUND(delay_rate_pct, 2) AS delay_rate_pct,
       ROUND(delay_rate_pct - LAG(delay_rate_pct) OVER (ORDER BY flight_date), 2) AS change_vs_prev_day
FROM daily
ORDER BY flight_date;
