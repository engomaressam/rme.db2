import mysql from "mysql2/promise";
import dotenv from "dotenv";
dotenv.config();

// Create MySQL connection pool
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  connectTimeout: 10000, // Set timeout for connection attempts
  debug: process.env.NODE_ENV === "development",
  enableKeepAlive: true,
  keepAliveInitialDelay: 10000,
  multipleStatements: false,
});

// Function to execute SQL queries
async function runMySQLQuery(sql, params = [], maxRetries = 3) {
  let connection;
  let attempts = 0;

  // Extract LIMIT & OFFSET for special handling
  if (sql.includes("LIMIT") && sql.includes("OFFSET")) {
    let limit = params[0];
    let offset = params[1];

    // Ensure limit & offset are numbers
    if (isNaN(limit) || isNaN(offset)) {
      throw new Error("LIMIT and OFFSET must be valid numbers.");
    }

    // Convert them to integers and replace placeholders
    sql = sql.replace("LIMIT ?", `LIMIT ${parseInt(limit, 10)}`);
    sql = sql.replace("OFFSET ?", `OFFSET ${parseInt(offset, 10)}`);

    // Remove limit/offset parameters from the array
    params = params.slice(2);
  }

  while (attempts < maxRetries) {
    try {
      connection = await pool.getConnection();
      console.log(
        `🔍 Running SQL Query (Attempt ${attempts + 1}):`,
        sql,
        params
      );

      const [rows] = await connection.execute(sql, params);
      return rows;
    } catch (error) {
      console.error("MySQL Query Error:", {
        message: error.message,
        sql: sql,
        params: params,
        stack: error.stack,
      });

      if (attempts >= maxRetries - 1) {
        throw new Error("Failed to execute query after multiple attempts");
      }

      console.log(`Retrying query (${attempts + 1}/${maxRetries})...`);
      attempts++;
    } finally {
      if (connection) connection.release();
    }
  }
}

export { runMySQLQuery };
