-- Completion vs termination by trial phase (the development funnel).
SELECT
  phase_clean AS phase,
  COUNT(*) AS n_trials,
  SUM(CASE WHEN overall_status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed,
  SUM(CASE WHEN overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED') THEN 1 ELSE 0 END) AS failed,
  ROUND(100.0 * SUM(CASE WHEN overall_status = 'COMPLETED' THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN overall_status IN ('COMPLETED','TERMINATED','WITHDRAWN','SUSPENDED')
                          THEN 1 ELSE 0 END), 0), 1) AS completion_pct_of_resolved
FROM trials
GROUP BY phase_clean
ORDER BY n_trials DESC;
