import { gql } from "@apollo/client";

export const GET_USER_BY_EMAIL = gql`
  query GetUserByEmail($email: String!) {
    getUserByEmail(email: $email) {
      name
    }
  }
`;

export const GET_EXPENDITURES = gql`
  query {
    getAllExpenditures {
      expenditureItemId
      transactionSource
      projectId
      projectNum
      projectName
      projectType
      taskId
      taskNum
      taskName
      quantity
      uom
      lineDesc
    }
  }
`;

// get Receipts
export const GET_RECEIPTS = gql`
  query {
    getReceipts {
      receiptId
      receiptNum
      orgId
      receiptProjNam
      receiptProjCode
      receiptAmount
      receiptDate
      invoiceNum
      appliedAmount
      attribute1
      newCalcTotalAdj
      currency
      transactionAmount
      totalAfterTax
      calcAmountToCollect
      totalAmountApplied
      trxPrjNam
      trxPrjCode
      transStatus
      transType
      customerNum
      customerNam
    }
  }
`;
