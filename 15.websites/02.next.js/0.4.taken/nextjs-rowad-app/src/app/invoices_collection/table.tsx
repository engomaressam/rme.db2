"use client";

import { useQuery } from "@apollo/client";
import { useState, useRef, useEffect } from "react";
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
  Input,
  Button,
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
  const [filters, setFilters] = useState<Record<string, string>>({
    receiptProjNam: "",
    receiptDate: "",
    invoiceNum: "",
    attribute1: "",
    trxPrjNam: "",
    transStatus: "",
    customerNam: "",
  });

  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;

  const dropdownRef = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const [showDropdown, setShowDropdown] = useState<{ [key: string]: boolean }>(
    {}
  );

  useEffect(() => {
    // Close dropdown when clicking outside
    const handleClickOutside = (event: MouseEvent) => {
      Object.keys(showDropdown).forEach((key) => {
        if (
          showDropdown[key] &&
          dropdownRef.current[key] &&
          !dropdownRef.current[key]?.contains(event.target as Node)
        ) {
          setShowDropdown((prev) => ({ ...prev, [key]: false }));
        }
      });
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showDropdown]);

  if (loading) return <p className="text-center text-gray-500">Loading...</p>;
  if (error)
    return <p className="text-center text-red-500">Error: {error.message}</p>;

  // Extract unique values for filtering
  const uniqueValues = (key: keyof Receipt) => {
    return Array.from(
      new Set(data.getReceipts.map((receipt: Receipt) => receipt[key]))
    ).filter(Boolean);
  };

  // Filtered data based on selected filters
  const filteredReceipts = data.getReceipts.filter((receipt: Receipt) => {
    return Object.entries(filters).every(([key, value]) => {
      if (!value) return true;
      const receiptValue = receipt[key as keyof Receipt] ?? ""; // Ensure it's a string
      return receiptValue.toString().toLowerCase().includes(value.toLowerCase());
    });
  });

  // Pagination logic
  const totalPages = Math.ceil(filteredReceipts.length / itemsPerPage);
  const currentReceipts = filteredReceipts.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Handle filter change (Typing)
  const handleInputChange = (key: keyof Receipt, value: string) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
    setShowDropdown((prev) => ({ ...prev, [key]: true }));
    setCurrentPage(1);
  };

  // Handle selecting an option from dropdown
  const handleDropdownSelect = (key: keyof Receipt, value: string) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
    setShowDropdown((prev) => ({ ...prev, [key]: false }));
  };

  return (
    <div className="relative overflow-x-auto shadow-md sm:rounded-lg p-4 bg-white dark:bg-gray-900">
      {/* Action Buttons */}
      <div className="flex justify-between items-center pb-4">
        <DownloadExcel data={filteredReceipts} filename="Receipts.xlsx" />
      </div>

      {/* Filters with Labels, Search, and Live Dropdown */}
      <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-4">
        {Object.keys(filters).map((filterKey) => (
          <div key={filterKey} className="relative">
            <label className="block text-gray-700 dark:text-gray-300 mb-2">
              Filter by {filterKey.replace(/([A-Z])/g, " $1").trim()}
            </label>
            <Input
              type="text"
              placeholder={`Search ${filterKey
                .replace(/([A-Z])/g, " $1")
                .trim()}`}
              className="w-full"
              value={filters[filterKey as keyof Receipt] || ""}
              onChange={(e) =>
                handleInputChange(filterKey as keyof Receipt, e.target.value)
              }
            />
            {showDropdown[filterKey] && (
              <div
                ref={(el) => {
                  dropdownRef.current[filterKey] = el;
                }}
                className="absolute z-10 w-full bg-white border rounded-md shadow-md max-h-40 overflow-auto"
              >
                {uniqueValues(filterKey as keyof Receipt)
                  .filter((val) =>
                    (val as string)
                      .toLowerCase()
                      .includes(
                        filters[filterKey as keyof Receipt].toLowerCase()
                      )
                  )
                  .map((value) => (
                    <div
                      key={value as string}
                      className="px-4 py-2 cursor-pointer hover:bg-gray-200"
                      onClick={() =>
                        handleDropdownSelect(
                          filterKey as keyof Receipt,
                          value as string
                        )
                      }
                    >
                      {value as string}
                    </div>
                  ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Data Table */}
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
          {currentReceipts.map((receipt: Receipt, index: number) => (
            <TableRow key={`${receipt.receiptId}-${receipt.receiptNum}-${index}`}>
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

      {/* Pagination Controls */}
      <div className="flex justify-center items-center space-x-2 mt-4">
        <Button
          onPress={() => setCurrentPage(1)}
          isDisabled={currentPage === 1}
          className="px-4 py-2"
        >
          First
        </Button>
        <Button
          onPress={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
          isDisabled={currentPage === 1}
          className="px-4 py-2"
        >
          Previous
        </Button>
        <span className="px-4 py-2 bg-gray-200 rounded-md">
          Page {currentPage} of {totalPages}
        </span>
        <Button
          onPress={() =>
            setCurrentPage((prev) => Math.min(prev + 1, totalPages))
          }
          isDisabled={currentPage === totalPages}
          className="px-4 py-2"
        >
          Next
        </Button>
        <Button
          onPress={() => setCurrentPage(totalPages)}
          isDisabled={currentPage === totalPages}
          className="px-4 py-2"
        >
          Last
        </Button>
      </div>
    </div>
  );
}
