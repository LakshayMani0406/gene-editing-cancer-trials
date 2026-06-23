-- Trial volume and completion by cancer type, split hematologic vs solid.
SELECT
  cancer_type,
  tumor_category,
  COUNT(*) AS n_trials,
  SUM(CASE WHEN overall_status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed,
  ROUND(100.0 * SUM(CASE WHEN overall_status = 'COMPLETED' THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN overall_status IN ('COMPLETED','TERMINATED','WITHDRAWN','SUSPENDED')
                          THEN 1 ELSE 0 END), 0), 1) AS completion_pct_of_resolved
FROM trials
GROUP BY cancer_type, tumor_category
HAVING n_trials >= 20
ORDER BY n_trials DESC
LIMIT 20;
