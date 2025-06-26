SELECT YEAR(gl_date) AS year, SUM(amount) AS total_amount
FROM [omar.rme1].[dbo].[cost_dist]
WHERE project_no = '60'
GROUP BY YEAR(gl_date)
WITH ROLLUP;