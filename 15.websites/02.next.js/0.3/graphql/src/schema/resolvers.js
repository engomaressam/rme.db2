import { runMySQLQuery } from "../config/db.js";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

const resolvers = {
  Mutation: {
    // ✅ User Registration (Only Admins Can Create Users)
    createUser: async (_, { username, email, password, role }, { user }) => {
      try {
        // Check if user is admin
        const users = await runMySQLQuery("SELECT * FROM RME_TEST.users");
        if (users.length > 0 && (!user || user.role !== "admin")) {
          throw new Error("❌ Unauthorized: Only Admins can create users.");
        }

        // Hash password securely
        const hashedPassword = await bcrypt.hash(password, 10);
        await runMySQLQuery(
          "INSERT INTO RME_TEST.users (username, email, password, role) VALUES (?, ?, ?, ?)",
          [username, email, hashedPassword, role]
        );

        return { success: true, message: "✅ User created successfully" };
      } catch (error) {
        console.error("🚨 Error creating user:", error);
        throw new Error("❌ Failed to create user.");
      }
    },

    // ✅ User Login
    loginUser: async (_, { email, password }) => {
      try {
        const users = await runMySQLQuery(
          "SELECT * FROM RME_TEST.users WHERE email = ?",
          [email]
        );
        if (!users.length) {
          throw new Error("❌ User not found.");
        }

        const user = users[0];
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
          throw new Error("❌ Invalid credentials.");
        }

        // Generate JWT Token
        const token = jwt.sign(
          { id: user.id, role: user.role },
          process.env.JWT_SECRET,
          { expiresIn: process.env.JWT_EXPIRES_IN }
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
      } catch (error) {
        console.error("🚨 Error logging in:", error);
        throw new Error("❌ Login failed.");
      }
    },

    // ✅ Insert New Expenditure
    insertExpenditure: async (_, args) => {
      try {
        const {
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
        } = args;

        await runMySQLQuery(
          `INSERT INTO RME_TEST.pa_cost_distribution_lines_all (
            EXPENDITURE_ITEM_ID, TRANSACTION_SOURCE, PROJECT_ID, PROJECT_NUM,
            PROJECT_NAME, PROJECT_TYPE, TASK_ID, TASK_NUM, TASK_NAME,
            QUANTITY, UOM, LINE_DESC
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

        return "✅ Expenditure inserted successfully.";
      } catch (error) {
        console.error("🚨 Error inserting expenditure:", error);
        throw new Error("❌ Failed to insert expenditure.");
      }
    },
  },

  Query: {
    // ✅ Ping Test
    ping: () => "pong",

    // ✅ Fetch All Users (Admin Only)
    getUsers: async (_, __, { user }) => {
      try {
        if (!user || user.role !== "admin") {
          throw new Error("❌ Unauthorized: Only Admins can fetch users.");
        }
        return await runMySQLQuery(
          "SELECT id, username, email, role FROM RME_TEST.users"
        );
      } catch (error) {
        console.error("🚨 Error fetching users:", error);
        throw new Error("❌ Failed to fetch users.");
      }
    },

    // ✅ Get User by Email
    getUserByEmail: async (_, { email }, { user }) => {
      try {
        if (!user) {
          throw new Error("❌ Unauthorized: Please log in.");
        }

        const users = await runMySQLQuery(
          "SELECT id, username, email, role FROM RME_TEST.users WHERE email = ?",
          [email]
        );

        return users.length ? users[0] : null;
      } catch (error) {
        console.error("🚨 Error fetching user:", error);
        throw new Error("❌ Failed to fetch user.");
      }
    },

    // ✅ Fetch All Expenditures
    getAllExpenditures: async () => {
      try {
        const data = await runMySQLQuery(
          "SELECT * FROM RME_TEST.pa_cost_distribution_lines_all"
        );

        return data.map((row) => ({
          expenditureItemId: row?.EXPENDITURE_ITEM_ID,
          transactionSource: row?.TRANSACTION_SOURCE,
          projectId: row?.PROJECT_ID,
          projectNum: row?.PROJECT_NUM,
          projectName: row?.PROJECT_NAME,
          projectType: row?.PROJECT_TYPE,
          taskId: row?.TASK_ID,
          taskNum: row?.TASK_NUM,
          taskName: row?.TASK_NAME,
          quantity: row?.QUANTITY,
          uom: row?.UOM,
          lineDesc: row?.LINE_DESC,
        }));
      } catch (error) {
        console.error("🚨 Error fetching expenditures:", error);
        throw new Error("❌ Failed to fetch expenditures.");
      }
    },

    // ✅ Get Expenditure by ID
    getExpenditureById: async (_, { id }) => {
      try {
        const results = await runMySQLQuery(
          "SELECT * FROM RME_TEST.pa_cost_distribution_lines_all WHERE EXPENDITURE_ITEM_ID = ?",
          [id]
        );

        return results.length ? results[0] : null;
      } catch (error) {
        console.error(`🚨 Error fetching expenditure (ID: ${id}):`, error);
        throw new Error("❌ Failed to fetch expenditure.");
      }
    },

    // ✅ Fetch All Receipts
    getReceipts: async () => {
      try {
        const data = await runMySQLQuery("SELECT * FROM RME_TEST.Receipts_Report");

        return data.map((row) => ({
          receiptId: row?.RECEIPT_ID,
          receiptNum: row?.RECEIPT_NUMBER,
          orgId: row?.ORG_ID,
          receiptProjNam: row?.RECEIPT_PRJ_NAME,
          receiptProjCode: row?.RECEIPT_PRJ_CODE,
          receiptAmount: row?.RECEIPT_AMOUNT,
          receiptDate: row?.RECEIPT_DATE,
          invoiceNum: row?.INV_NUM,
          appliedAmount: row?.AMOUNT_APPLIED,
          attribute1: row?.ATTRIBUTE1,
          newCalcTotalAdj: row?.NEW_CALCULATED_TOTAL_ADJ,
          currency: row?.CURRENCY,
          transactionAmount: row?.TRANSACTION_AMOUNT,
          totalAfterTax: row?.TOTAL_AFTER_TAX,
          calcAmountToCollect: row?.CALCULATED_AMOUNT_TO_COLLECT,
          totalAmountApplied: row?.TOTAL_AMOUNT_APPLIED,
          trxPrjNam: row?.TRX_PRJ_NAME,
          trxPrjCode: row?.TRX_PRJ_CODE,
          transStatus: row?.STATUS,
          transType: row?.TYPE,
          customerNum: row?.CUSTOMER_NO,
          customerNam: row?.CUSTOMER_NAME,
        }));
      } catch (error) {
        console.error("🚨 Error fetching receipts:", error);
        throw new Error("❌ Failed to fetch receipts.");
      }
    },
  },
};

export default resolvers;
