import { gql } from "apollo-server-express";

const typeDefs = gql`
  # User Schema

  type User {
    id: ID!
    username: String!
    email: String!
    role: String!
  }

  type AuthResponse {
    token: String!
    user: User!
  }

  type Query {
    ping: String!
    getUsers: [User] # Only Admins Can Fetch Users
    getUserByEmail(email: String!): User # Fetch user by email
  }

  type Mutation {
    createUser(
      username: String!
      email: String!
      password: String!
      role: String!
    ): Response
    loginUser(email: String!, password: String!): AuthResponse
  }

  type Response {
    success: Boolean!
    message: String!
  }

  # Expenditure Schema

  type Query {
    getAllExpenditures(limit: Int = 100, offset: Int = 0): [Expenditure]

    getExpenditureById(id: ID!): Expenditure
  }

  type Mutation {
    insertExpenditure(
      expenditureItemId: ID!
      transactionSource: String!
      projectId: Int!
      projectNum: Int!
      projectName: String!
      projectType: String!
      taskId: Int!
      taskNum: String!
      taskName: String!
      quantity: Float!
      uom: String!
      lineDesc: String!
    ): String

    updateExpenditure(
      expenditureItemId: ID!
      transactionSource: String
      projectId: Int
      projectNum: Int
      projectName: String
      projectType: String
      taskId: Int
      taskNum: String
      taskName: String
      quantity: Float
      uom: String
      lineDesc: String
    ): String

    deleteExpenditure(id: ID!): String
  }

  type Expenditure {
    expenditureItemId: ID
    transactionSource: String
    projectId: Int
    projectNum: Int
    projectName: String
    projectType: String
    taskId: Int
    taskNum: String
    taskName: String
    quantity: Float
    uom: String
    lineDesc: String
  }

  # RECEIPT Schema
  type Receipt {
    receiptId: ID
    receiptNum: String
    orgId: Int
    receiptProjNam: String
    receiptProjCode: String
    receiptAmount: String
    receiptDate: String
    invoiceNum: String
    appliedAmount: String
    attribute1: String
    newCalcTotalAdj: String
    currency: String
    transactionAmount: String
    totalAfterTax: String
    calcAmountToCollect: String
    totalAmountApplied: String
    trxPrjNam: String
    trxPrjCode: String
    transStatus: String
    transType: String
    customerNum: Int
    customerNam: String
  }

  type Query {
    getReceipts: [Receipt]
  }
`;

export default typeDefs;
