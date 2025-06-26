SELECT 
    YEAR(creation_date) AS year, 
    COUNT(*) AS row_count
FROM 
    po_followup
GROUP BY 
    YEAR(creation_date)
ORDER BY 
    year;