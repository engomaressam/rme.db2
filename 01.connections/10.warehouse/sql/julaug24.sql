SELECT * 
FROM cost_dist
WHERE 
    (MONTH(gl_date) = 7 OR MONTH(gl_date) = 8) 
    AND YEAR(gl_date) = 2024;