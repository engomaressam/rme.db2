import { runMySQLQuery } from "../config/db.js";
import bcrypt from "bcryptjs";

import jwt from "jsonwebtoken";

const resolvers = {
  Mutation: {
    // User Registration (Only Admins Can Create Users)
    createUser: async (_, { username, email, password, role }, { user }) => {
      // Temporarily allow initial admin creation
      const users = await runMySQLQuery("SELECT * FROM RME_TEST.users");
      if (users.length > 0 && (!user || user.role !== "admin")) {
        throw new Error("Unauthorized");
      }

      const hashedPassword = await bcrypt.hash(password, 10);
      await runMySQLQuery(
        "INSERT INTO RME_TEST.users (username, email, password, role) VALUES (?, ?, ?, ?)",
        [username, email, hashedPassword, role]
      );

      return { success: true, message: "User created successfully" };
    },

    // User Login
    loginUser: async (_, { email, password }) => {
      const users = await runMySQLQuery(
        "SELECT * FROM RME_TEST.users WHERE email = ?",
        [email]
      );
      if (!users.length) {
        throw new Error("❌ User not found");
      }

      const user = users[0];
      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) {
        throw new Error("Invalid credentials");
      }

      const token = jwt.sign(
        { id: user.id, role: user.role },
        process.env.JWT_SECRET,
        {
          expiresIn: process.env.JWT_EXPIRES_IN,
        }
      );

      return {
        token,
        user: {
          id: user.id,
          username: user.username,
          email: user.email,
          role: user.role,
        },
      };
    },

    // Mutation to insert a new expenditure into the database
    insertExpenditure: async (
      _,
      {
        expenditureItemId,
        transactionSource,
        projectId,
        projectNum,
        projectName,
        projectType,
        taskId,
        taskNum,
        taskName,
        quantity,
        uom,
        lineDesc,
      }
    ) => {
      try {
        const result = await runMySQLQuery(
          `INSERT INTO RME_TEST.pa_cost_distribution_lines_all (
            EXPENDITURE_ITEM_ID,
            TRANSACTION_SOURCE,
            PROJECT_ID,
            PROJECT_NUM,
            PROJECT_NAME,
            PROJECT_TYPE,
            TASK_ID,
            TASK_NUM,
            TASK_NAME,
            QUANTITY,
            UOM,
            LINE_DESC
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            expenditureItemId,
            transactionSource,
            projectId,
            projectNum,
            projectName,
            projectType,
            taskId,
            taskNum,
            taskName,
            quantity,
            uom,
            lineDesc,
          ]
        );
        console.log("Expenditure inserted:", result);
        return "Expenditure inserted successfully";
      } catch (error) {
        console.error("Error inserting expenditure:", error);
        throw new Error("Failed to insert expenditure.");
      }
    },
  },

  Query: {
    ping: () => "pong",
    // Query to fetch all users from the database
    getUsers: async (_, __, { user }) => {
      if (!user || user.role !== "admin") {
        throw new Error("Unauthorized");
      }
      return await runMySQLQuery(
        "SELECT id, username, email, role FROM RME_TEST.users"
      );
    },

    // Get User by Email (Logged-in users only)
    getUserByEmail: async (_, { email }, { user }) => {
      if (!user) {
        throw new Error("❌ Unauthorized: Please log in");
      }

      console.log(`🔍 Searching for user with email: ${email}`);

      const users = await runMySQLQuery(
        "SELECT id, username, email, role FROM RME_TEST.users WHERE email = ?",
        [email]
      );

      if (!users.length) {
        console.warn("⚠️ User not found");
        return null;
      }

      return users[0];
    },

    // Query to fetch all expenditures from the database
    getAllExpenditures: async () => {
      try {
        console.log("Fetching all expenditures");

        const data = await runMySQLQuery(
          "SELECT * FROM RME_TEST.pa_cost_distribution_lines_all"
        );

        if (!data.length) {
          console.warn("No expenditures found in the database.");
          return [];
        }

        // Helper function to map database row to expenditure object
        const mapRowToExpenditure = (row) => ({
          expenditureItemId: row.EXPENDITURE_ITEM_ID ?? null,
          transactionSource: row.TRANSACTION_SOURCE ?? null,
          projectId: row.PROJECT_ID ?? null,
          projectNum: row.PROJECT_NUM ?? null,
          projectName: row.PROJECT_NAME ?? null,
          projectType: row.PROJECT_TYPE ?? null,
          taskId: row.TASK_ID ?? null,
          taskNum: row.TASK_NUM ?? null,
          taskName: row.TASK_NAME ?? null,
          quantity: row.QUANTITY ?? null,
          uom: row.UOM ?? null,
          lineDesc: row.LINE_DESC ?? null,
        });

        return data.map(mapRowToExpenditure);
      } catch (error) {
        console.error("Error fetching expenditures:", error);
        throw new Error("Failed to fetch expenditures.");
      }
    },

    getExpenditureById: async (_, { id }) => {
      try {
        console.log(`Fetching expenditure with ID: ${id}`);

        const results = await runMySQLQuery(
          "SELECT * FROM RME_TEST.pa_cost_distribution_lines_all WHERE EXPENDITURE_ITEM_ID = ?",
          [id]
        );

        if (results.length === 0) {
          console.warn(`No expenditure found with ID: ${id}`);
          return null;
        }

        console.log("Expenditure retrieved:", results[0]);

        return {
          expenditureItemId: results[0].EXPENDITURE_ITEM_ID ?? null,
          transactionSource: results[0].TRANSACTION_SOURCE ?? null,
          projectId: results[0].PROJECT_ID ?? null,
          projectNum: results[0].PROJECT_NUM ?? null,
          projectName: results[0].PROJECT_NAME ?? null,
          projectType: results[0].PROJECT_TYPE ?? null,
          taskId: results[0].TASK_ID ?? null,
          taskNum: results[0].TASK_NUM ?? null,
          taskName: results[0].TASK_NAME ?? null,
          quantity: results[0].QUANTITY ?? null,
          uom: results[0].UOM ?? null,
          lineDesc: results[0].LINE_DESC ?? null,
        };
      } catch (error) {
        console.error(`Error fetching expenditure with ID ${id}:`, error);
        throw new Error("Failed to fetch expenditure.");
      }
    },

    // Query to fetch all RECEIPTS from the database
    getReceipts: async () => {
      try {
        console.log("Fetching all receipts");

        const data = await runMySQLQuery(
          "SELECT * FROM RME_TEST.receipts_2"
        );

        if (!data.length) {
          console.warn("No receipts found in the database.");
          return [];
        }

        // Helper function to map database row to receipt object
        const mapRowToReceipt = (row) => ({
          receiptId: row.RECEIPT_ID ?? undefined, // Change null to undefined to handle potential null values
          receiptNum: row.RECEIPT_NUMBER ?? null,
          orgId: row.ORG_ID ?? null,
          receiptProjNam: row.RECEIPT_PRJ_NAME ?? null,
          receiptProjCode: row.RECEIPT_PRJ_CODE ?? null,
          receiptAmount: row.RECEIPT_AMOUNT ?? null,
          receiptDate: row.RECEIPT_DATE ?? null,
          invoiceNum: row.INV_NUM ?? null,
          appliedAmount: row.AMOUNT_APPLIED ?? null,
          attribute1: row.ATTRIBUTE1 ?? null,
          newCalcTotalAdj: row.NEW_CALCULATED_TOTAL_ADJ ?? null,
          currency: row.CURRENCY ?? null,
          transactionAmount: row.TRANSACTION_AMOUNT ?? null,
          totalAfterTax: row.TOTAL_AFTER_TAX ?? null,
          calcAmountToCollect: row.CALCULATED_AMOUNT_TO_COLLECT ?? null,
          totalAmountApplied: row.TOTAL_AMOUNT_APPLIED ?? null,
          trxPrjNam: row.TRX_PRJ_NAME ?? null,
          trxPrjCode: row.TRX_PRJ_CODE ?? null,
          transStatus: row.STATUS ?? null,
          transType: row.TYPE ?? null,
          customerNum: row.CUSTOMER_NO ?? null,
          customerNam: row.CUSTOMER_NAME ?? null,
        });

        return data.map(mapRowToReceipt);
      } catch (error) {
        console.error("Error fetching receipts:", error);
        throw new Error("Failed to fetch receipts.");
      }
    },
  },
};

export default resolvers;
