SELECT 
    line_desc, 
    unit, 
    qty, 
    amount
FROM 
    cost_dist
WHERE 
    project_no = '123' 
    AND gl_date BETWEEN '2023-01-01' AND '2023-12-31';