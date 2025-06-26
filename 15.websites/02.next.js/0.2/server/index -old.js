require('dotenv').config();
const express = require('express');
const mysql = require('mysql2'); // Use mysql2 for better compatibility
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json()); // Allows API to read JSON requests

// 🔹 Connect to MySQL on Linux server
const db = mysql.createConnection({
    host: "10.10.11.242",
    user: "gamal",
    password: "password123Y$",
    database: "RME_TEST",
    port: 3306, // Ensure MySQL port is correct
    multipleStatements: true // Allow multiple queries
});

db.connect(err => {
    if (err) {
        console.error("❌ MySQL Connection Failed:", err);
        process.exit(1); // Exit process if DB fails to connect
    } else {
        console.log("✅ MySQL Connected to RME_TEST");
    }
});

// 🔹 Test API Route
app.get("/", (req, res) => {
    res.send("✅ API is running!");
});

// 🔹 Get All Expenditures
app.get("/expenditures", (req, res) => {
    db.query("SELECT * FROM RME_TEST.RME_Projects_Cost_Dist_Line_Report", (err, result) => {
        if (err) {
            console.error("❌ MySQL Query Error:", err);
            res.status(500).json({ error: err.message });
        } else {
            res.json(result);
        }
    });
});

// 🔹 Start API Server on Port 4000
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
    console.log(`🚀 API Server running on http://localhost:${PORT}`);
});
