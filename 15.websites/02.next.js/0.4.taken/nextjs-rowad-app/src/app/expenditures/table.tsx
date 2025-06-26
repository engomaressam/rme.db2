"use client";

import { useQuery } from "@apollo/client";
import { useState } from "react";
import { GET_EXPENDITURES } from "../graphql/queries";
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

// Define the Expenditure type
interface Expenditure {
  expenditureItemId: string;
  transactionSource: string;
  projectId: string;
  projectName: string;
  taskName: string;
  quantity: number;
  uom: string;
  lineDesc: string;
}

export default function ExpenditureList() {
  const { data, loading, error } = useQuery(GET_EXPENDITURES, {
    pollInterval: 5000, // Poll every 5 seconds
  });

  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState(""); // Search query for dropdown
  const [selectedProject, setSelectedProject] = useState(""); // Selected project filter
  const [selectedRow, setSelectedRow] = useState<string | null>(null); // Selected row tracking
  const itemsPerPage = 10; // Number of items per page

  if (loading) return <p className="text-center text-gray-500">Loading...</p>;
  if (error)
    return <p className="text-center text-red-500">Error: {error.message}</p>;

  // Extract unique project names for the dropdown
  const projectNames: string[] = Array.from(
    new Set(
      (data.getAllExpenditures as Expenditure[]).map(
        (exp: Expenditure) => exp.projectName
      )
    )
  ).filter((project) =>
    project.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Filter expenditures based on selected project
  const filteredExpenditures = selectedProject
    ? data.getAllExpenditures.filter(
        (expenditure: Expenditure) =>
          expenditure.projectName === selectedProject
      )
    : data.getAllExpenditures;

  const totalPages = Math.ceil(filteredExpenditures.length / itemsPerPage);
  const currentExpenditures = filteredExpenditures.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <div className="relative overflow-x-auto shadow-md sm:rounded-lg p-4">
      {/* Project Dropdown with Label & Search */}
      <div className="flex flex-col space-y-2 pb-4 bg-white dark:bg-gray-900">
        {/* Download Button */}
        <DownloadExcel
          data={data.getAllExpenditures}
          filename="Expenditures.xlsx"
        />

        <Dropdown>
          <DropdownTrigger>
            <Input
              id="project-search"
              type="text"
              label="Filter by Project"
              labelPlacement="inside"
              placeholder="Search for a project..."
              className="w-80"
              value={selectedProject || searchQuery} // Hold selected project until cleared
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setSelectedProject(""); // Reset selection when typing
              }}
            />
          </DropdownTrigger>
          <DropdownMenu
            disallowEmptySelection
            aria-label="Project Dropdown"
            selectedKeys={[selectedProject]}
            selectionMode="single"
            className="
            rounded-b-lg
            overflow-auto 
            max-h-60 
            w-80
            "
            onSelectionChange={(selected) => {
              setSelectedProject(Array.from(selected as Set<string>)[0]);
              setSearchQuery(""); // Clear search input after selection
            }}
          >
            <>
              <DropdownItem key="all" onPress={() => setSelectedProject("")}>
                All Projects
              </DropdownItem>
              {projectNames.length > 0 ? (
                projectNames.map((project) => (
                  <DropdownItem
                    key={project}
                    onPress={() => {
                      setSelectedProject(project);
                      setSearchQuery(""); // Clear search input after selection
                    }}
                  >
                    {project}
                  </DropdownItem>
                ))
              ) : (
                <DropdownItem key="no-matching" isDisabled>
                  No matching projects
                </DropdownItem>
              )}
            </>
          </DropdownMenu>
        </Dropdown>
      </div>

      {/* Data Table Using Hero UI */}
      <Table isStriped aria-label="Expenditure Table">
        <TableHeader>
          <TableColumn>Project Name</TableColumn>
          <TableColumn>Task</TableColumn>
          <TableColumn>Transaction Source</TableColumn>
          <TableColumn className="text-center">Quantity</TableColumn>
          <TableColumn>UOM</TableColumn>
          <TableColumn>Description</TableColumn>
        </TableHeader>
        <TableBody>
          {currentExpenditures.map((expenditure: Expenditure) => (
            <TableRow
              key={expenditure.expenditureItemId}
              className={`cursor-pointer ${
                selectedRow === expenditure.expenditureItemId
                  ? "bg-blue-200"
                  : ""
              }`}
              onClick={() =>
                setSelectedRow(
                  selectedRow === expenditure.expenditureItemId
                    ? null
                    : expenditure.expenditureItemId
                )
              }
            >
              <TableCell>{expenditure.projectName}</TableCell>
              <TableCell>{expenditure.taskName}</TableCell>
              <TableCell>{expenditure.transactionSource}</TableCell>
              <TableCell className="text-center">
                {expenditure.quantity}
              </TableCell>
              <TableCell>{expenditure.uom}</TableCell>
              <TableCell>{expenditure.lineDesc}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {currentExpenditures.length === 0 && (
        <p className="text-center text-gray-500 mt-4">No data available.</p>
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
