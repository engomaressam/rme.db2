require('dotenv').config();
const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json()); // Allows API to read JSON requests

// 🔹 Connect to MySQL on Linux server
const db = mysql.createConnection({
    host: "10.10.11.242",
    user: "gamal",
    password: "password123Y$",
    database: "RME_TEST"
});

db.connect(err => {
    if (err) {
        console.error("❌ MySQL Connection Failed:", err);
    } else {
        console.log("✅ MySQL Connected to RME_TEST");
    }
});

// 🔹 Test API Route
app.get("/", (req, res) => {
    res.send("✅ API is running!");
});

// 🔹 Get Unique Project Names with Total Amount
app.get("/projects-summary", (req, res) => {
    const query = `
        SELECT PROJECT_NAME, SUM(AMOUNT) AS total_amount 
        FROM RME_Projects_Cost_Dist_Line_Report 
        GROUP BY PROJECT_NAME
        ORDER BY total_amount DESC
        LIMIT 20;`;

    db.query(query, (err, result) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else {
            res.json(result);
        }
    });
});

// 🔹 Start API Server on Port 4000
app.listen(4000, () => {
    console.log("🚀 API Server running on http://localhost:4000");
});
