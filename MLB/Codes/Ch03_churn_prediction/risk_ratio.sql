SELECT
    gender,
    avg(churn),
    avg(churn) - global_churn,
    avg(churn) - global_churn -- global_churn如何计算的？
FROM data
GROUP BY gender;