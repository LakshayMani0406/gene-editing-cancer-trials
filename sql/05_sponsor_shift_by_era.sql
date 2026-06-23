-- Industry's growing share of trials over time (the commercialization pivot).
SELECT
  trial_era,
  COUNT(*) AS n_trials,
  SUM(CASE WHEN sponsor_class_clean = 'Industry' THEN 1 ELSE 0 END) AS industry,
  ROUND(100.0 * SUM(CASE WHEN sponsor_class_clean = 'Industry' THEN 1 ELSE 0 END) / COUNT(*), 1) AS industry_pct
FROM trials
GROUP BY trial_era
ORDER BY CASE trial_era
  WHEN '1990s' THEN 1 WHEN '2000s' THEN 2 WHEN '2010-2014' THEN 3
  WHEN '2015-2019' THEN 4 WHEN '2020+' THEN 5 ELSE 6 END;
