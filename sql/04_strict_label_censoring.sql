-- Why even the strict label has a milder time effect: right-censoring.
-- Completion takes years, so among recently-resolved trials terminations dominate.
SELECT
  trial_era,
  COUNT(*) AS resolved_trials,
  SUM(CASE WHEN overall_status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed,
  SUM(CASE WHEN overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED') THEN 1 ELSE 0 END) AS terminated,
  ROUND(100.0 * SUM(CASE WHEN overall_status = 'COMPLETED' THEN 1 ELSE 0 END) / COUNT(*), 1) AS completed_pct
FROM trials
WHERE overall_status IN ('COMPLETED','TERMINATED','WITHDRAWN','SUSPENDED')
GROUP BY trial_era
ORDER BY CASE trial_era
  WHEN '1990s' THEN 1 WHEN '2000s' THEN 2 WHEN '2010-2014' THEN 3
  WHEN '2015-2019' THEN 4 WHEN '2020+' THEN 5 ELSE 6 END;
