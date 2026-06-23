-- The label leak in one query: what share of "successes" never actually completed?
-- The bundled target counts RECRUITING + ACTIVE_NOT_RECRUITING as success. Those trials
-- are concentrated in recent years, so calendar recency predicts the label by construction.
SELECT
  trial_era,
  SUM(CASE WHEN trial_outcome = 1 THEN 1 ELSE 0 END) AS positives,
  SUM(CASE WHEN overall_status IN ('RECRUITING','ACTIVE_NOT_RECRUITING') THEN 1 ELSE 0 END) AS in_progress_positives,
  ROUND(100.0 * SUM(CASE WHEN overall_status IN ('RECRUITING','ACTIVE_NOT_RECRUITING') THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN trial_outcome = 1 THEN 1 ELSE 0 END), 0), 1) AS pct_positives_in_progress
FROM trials
GROUP BY trial_era
ORDER BY CASE trial_era
  WHEN '1990s' THEN 1 WHEN '2000s' THEN 2 WHEN '2010-2014' THEN 3
  WHEN '2015-2019' THEN 4 WHEN '2020+' THEN 5 ELSE 6 END;
