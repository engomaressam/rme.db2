import { gql } from "@apollo/client";

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
