SELECT
    YEAR(date) AS Year,
    SUM(amount) AS TotalAmount
FROM
    [omar.rme1].[dbo].[mat_mov]
WHERE
    YEAR(date) IN (2018, 2019)
GROUP BY
    YEAR(date)
ORDER BY
    YEAR(date);
