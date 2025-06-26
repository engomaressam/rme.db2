SELECT
    YEAR(gl_date) AS Year,
    SUM(amount) AS TotalAmount
FROM
    [omar.rme1].[dbo].[cost_dist]
WHERE
    YEAR(gl_date) IN (2018, 2019, 2020, 2021, 2022, 2023, 2024)
GROUP BY
    YEAR(gl_date)
ORDER BY
    YEAR(gl_date);
