"use client";

import { useQuery } from "@apollo/client";
import { useState } from "react";
import { GET_RECEIPTS } from "../graphql/queries";
import DownloadExcel from "./DownloadExcel";
import {
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Input,
} from "@heroui/react";

// Define the Receipt type
interface Receipt {
  receiptId: string;
  receiptNum: string;
  orgId: string;
  receiptProjNam: string;
  receiptProjCode: string;
  receiptAmount: string;
  receiptDate: string;
  invoiceNum: string;
  appliedAmount: string;
  attribute1: string;
  newCalcTotalAdj: string;
  currency: string;
  transactionAmount: string;
  totalAfterTax: string;
  calcAmountToCollect: string;
  totalAmountApplied: string;
  trxPrjNam: string;
  trxPrjCode: string;
  transStatus: string;
  transType: string;
  customerNum: string;
  customerNam: string;
}

export default function ReceiptList() {
  const { data, loading, error } = useQuery(GET_RECEIPTS);

  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState("");
  const itemsPerPage = 5;

  if (loading) return <p className="text-center text-gray-500">Loading...</p>;
  if (error)
    return <p className="text-center text-red-500">Error: {error.message}</p>;

  // Extract unique project names for filtering
  const projectNames: string[] = Array.from(
    new Set(
      (data.getReceipts as Receipt[]).map(
        (receipt: Receipt) => receipt.receiptProjNam
      )
    )
  ).filter((project) =>
    project.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Filter receipts based on selected project
  const filteredReceipts = searchQuery
    ? data.getReceipts.filter(
        (receipt: Receipt) => receipt.receiptProjNam === searchQuery
      )
    : data.getReceipts;

  const totalPages = Math.ceil(filteredReceipts.length / itemsPerPage);
  const currentReceipts = filteredReceipts.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <div className="relative overflow-x-auto shadow-md sm:rounded-lg p-4">
      {/* Download Button */}
      <div className="flex flex-col space-y-2 pb-4 bg-white dark:bg-gray-900">
        <DownloadExcel data={data.getReceipts} filename="Receipts.xlsx" />

        {/* Search Filter */}
        <Dropdown>
          <DropdownTrigger>
            <Input
              id="receipt-search"
              type="text"
              label="Filter by Project"
              labelPlacement="inside"
              placeholder="Search for a project..."
              className="w-80"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </DropdownTrigger>
          <DropdownMenu
            disallowEmptySelection
            aria-label="Project Dropdown"
            selectedKeys={[searchQuery]}
            selectionMode="single"
            className="rounded-b-lg overflow-auto max-h-60 w-80"
            onSelectionChange={(selected) => {
              setSearchQuery(Array.from(selected as Set<string>)[0]);
            }}
          >
            {projectNames.length > 0 ? (
              projectNames.map((project) => (
                <DropdownItem
                  key={project}
                  onPress={() => setSearchQuery(project)}
                >
                  {project}
                </DropdownItem>
              ))
            ) : (
              <DropdownItem key="no-matching" isDisabled>
                No matching projects
              </DropdownItem>
            )}
          </DropdownMenu>
        </Dropdown>
      </div>

      {/* Data Table Using Hero UI */}
      <Table isStriped aria-label="Receipts Table">
        <TableHeader>
          <TableColumn>Receipt Number</TableColumn>
          <TableColumn>orgId</TableColumn>
          <TableColumn>Project Name</TableColumn>
          <TableColumn>receiptProjCode</TableColumn>
          <TableColumn>receiptAmount</TableColumn>
          <TableColumn>receiptDate</TableColumn>
          <TableColumn>invoiceNum</TableColumn>
          <TableColumn>appliedAmount</TableColumn>
          <TableColumn>attribute1</TableColumn>
          <TableColumn>newCalcTotalAdj</TableColumn>
          <TableColumn>currency</TableColumn>
          <TableColumn>transactionAmount</TableColumn>
          <TableColumn>totalAfterTax</TableColumn>
          <TableColumn>calcAmountToCollect</TableColumn>
          <TableColumn>totalAmountApplied</TableColumn>
          <TableColumn>trxPrjNam</TableColumn>
          <TableColumn>trxPrjCode</TableColumn>
          <TableColumn>transStatus</TableColumn>
          <TableColumn>transType</TableColumn>
          <TableColumn>customerNum</TableColumn>
          <TableColumn>customerNam</TableColumn>
        </TableHeader>
        <TableBody>
          {currentReceipts.map((receipt: Receipt) => (
            <TableRow key={`${receipt.receiptId}-${receipt.receiptNum}`}>
              <TableCell>{receipt.receiptNum}</TableCell>
              <TableCell>{receipt.orgId}</TableCell>
              <TableCell>{receipt.receiptProjNam}</TableCell>
              <TableCell>{receipt.receiptProjCode}</TableCell>
              <TableCell className="text-center">
                {receipt.receiptAmount}
              </TableCell>
              <TableCell>{receipt.receiptDate}</TableCell>
              <TableCell>{receipt.invoiceNum}</TableCell>
              <TableCell>{receipt.appliedAmount}</TableCell>
              <TableCell>{receipt.attribute1}</TableCell>
              <TableCell>{receipt.newCalcTotalAdj}</TableCell>
              <TableCell>{receipt.currency}</TableCell>
              <TableCell>{receipt.transactionAmount}</TableCell>
              <TableCell>{receipt.totalAfterTax}</TableCell>
              <TableCell>{receipt.calcAmountToCollect}</TableCell>
              <TableCell>{receipt.totalAmountApplied}</TableCell>
              <TableCell>{receipt.trxPrjNam}</TableCell>
              <TableCell>{receipt.trxPrjCode}</TableCell>
              <TableCell>{receipt.transStatus}</TableCell>
              <TableCell>{receipt.transType}</TableCell>
              <TableCell>{receipt.customerNum}</TableCell>
              <TableCell>{receipt.customerNam}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {currentReceipts.length === 0 && (
        <p className="text-center text-gray-500 mt-4">No receipts available.</p>
      )}

      {/* Pagination */}
      <div className="flex justify-center items-center space-x-2 mt-4">
        <button
          onClick={() => setCurrentPage(1)}
          disabled={currentPage === 1}
          className={`px-4 py-2 rounded-md ${
            currentPage === 1
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-500 text-white hover:bg-blue-600"
          }`}
        >
          First
        </button>

        <button
          onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
          disabled={currentPage === 1}
          className={`px-4 py-2 rounded-md ${
            currentPage === 1
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-500 text-white hover:bg-blue-600"
          }`}
        >
          Previous
        </button>

        <span className="px-4 py-2 bg-gray-200 rounded-md">
          Page {currentPage} of {totalPages}
        </span>

        <button
          onClick={() =>
            setCurrentPage((prev) => Math.min(prev + 1, totalPages))
          }
          disabled={currentPage === totalPages}
          className={`px-4 py-2 rounded-md ${
            currentPage === totalPages
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-500 text-white hover:bg-blue-600"
          }`}
        >
          Next
        </button>

        <button
          onClick={() => setCurrentPage(totalPages)}
          disabled={currentPage === totalPages}
          className={`px-4 py-2 rounded-md ${
            currentPage === totalPages
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-500 text-white hover:bg-blue-600"
          }`}
        >
          Last
        </button>
      </div>
    </div>
  );
}
