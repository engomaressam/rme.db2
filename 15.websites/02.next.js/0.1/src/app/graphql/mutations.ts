import { gql } from "@apollo/client";

export const CREATE_EXPENDITURE = gql`
  mutation (
    $expenditureItemId: ID!
    $transactionSource: String!
    $projectId: Int!
    $projectNum: Int!
    $projectName: String!
    $projectType: String!
    $taskId: Int!
    $taskNum: String!
    $taskName: String!
    $quantity: Float!
    $uom: String!
    $lineDesc: String!
  ) {
    insertExpenditure(
      expenditureItemId: $expenditureItemId
      transactionSource: $transactionSource
      projectId: $projectId
      projectNum: $projectNum
      projectName: $projectName
      projectType: $projectType
      taskId: $taskId
      taskNum: $taskNum
      taskName: $taskName
      quantity: $quantity
      uom: $uom
      lineDesc: $lineDesc
    )
  }
`;
