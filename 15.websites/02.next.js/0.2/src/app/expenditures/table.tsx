"use client";

import { useQuery } from "@apollo/client";
import { useState } from "react";
import { GET_EXPENDITURES } from "../graphql/queries";

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
  const [searchQuery, setSearchQuery] = useState(""); // Search query for the dropdown
  const [selectedProject, setSelectedProject] = useState(""); // Selected project filter
  const [isDropdownVisible, setDropdownVisible] = useState(false); // Controls dropdown visibility
  const itemsPerPage = 10; // Set the number of items per page

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
    <div className="relative overflow-x-auto shadow-md sm:rounded-lg">
      {/* Project Dropdown with Search */}
      <div className="pb-4 bg-white dark:bg-gray-900">
        <label htmlFor="project-filter" className="sr-only">
          Filter by Project Name
        </label>
        <div className="relative mt-1">
          <input
            type="text"
            placeholder="Search project name..."
            className="w-full p-2 text-sm border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 
            focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:focus:ring-blue-500 
            dark:focus:border-blue-500"
            value={selectedProject || searchQuery} // Hold selected project until cleared
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setSelectedProject(""); // Reset selection when typing
              setDropdownVisible(true); // Show dropdown when typing
            }}
            onFocus={() => setDropdownVisible(true)} // Show dropdown on focus
          />

          {/* Dropdown List */}
          {isDropdownVisible && searchQuery && projectNames.length > 0 && (
            <div className="absolute z-10 w-full bg-white border border-gray-300 rounded-lg mt-1 max-h-40 overflow-y-auto shadow-lg">
              {projectNames.map((project) => (
                <div
                  key={project}
                  className="p-2 cursor-pointer hover:bg-gray-200"
                  onClick={() => {
                    setSelectedProject(project);
                    setSearchQuery(""); // Clear search input after selection
                    setDropdownVisible(false); // Hide dropdown after selection
                  }}
                >
                  {project}
                </div>
              ))}
            </div>
          )}

          {/* Clear Selection Button */}
          {selectedProject && (
            <button
              onClick={() => {
                setSelectedProject("");
                setSearchQuery("");
                setDropdownVisible(false);
              }}
              className="absolute top-2 right-3 text-gray-500 hover:text-red-500"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Data Table */}
      <table className="w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
        <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
          <tr>
            <th scope="col" className="px-6 py-3">
              Project Name
            </th>
            <th scope="col" className="px-6 py-3">
              Task
            </th>
            <th scope="col" className="px-6 py-3">
              Transaction Source
            </th>
            <th scope="col" className="px-6 py-3 text-center">
              Quantity
            </th>
            <th scope="col" className="px-6 py-3">
              UOM
            </th>
            <th scope="col" className="px-6 py-3">
              Description
            </th>
          </tr>
        </thead>
        <tbody>
          {currentExpenditures.map(
            (expenditure: Expenditure, index: number) => (
              <tr
                key={expenditure.expenditureItemId}
                className={`bg-white border-b dark:bg-gray-800 dark:border-gray-700 border-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition ${
                  index % 2 === 0 ? "bg-gray-50" : ""
                }`}
              >
                <th
                  scope="row"
                  className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white"
                >
                  {expenditure.projectName}
                </th>
                <td className="px-6 py-4">{expenditure.taskName}</td>
                <td className="px-6 py-4">{expenditure.transactionSource}</td>
                <td className="px-6 py-4 text-center">
                  {expenditure.quantity}
                </td>
                <td className="px-6 py-4">{expenditure.uom}</td>
                <td className="px-6 py-4">{expenditure.lineDesc}</td>
              </tr>
            )
          )}
        </tbody>
      </table>

      {currentExpenditures.length === 0 ? (
        <p className="text-center text-gray-500 mt-4">No data available.</p>
      ) : null}

      <div className="flex justify-between mt-4">
        <button
          onClick={() => setCurrentPage(1)}
          disabled={currentPage === 1}
          className="bg-gray-300 px-4 py-2 rounded"
        >
          First
        </button>

        <button
          onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
          disabled={currentPage === 1}
          className="bg-gray-300 px-4 py-2 rounded"
        >
          Previous
        </button>
        <span>
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={() =>
            setCurrentPage((prev) => Math.min(prev + 1, totalPages))
          }
          disabled={currentPage === totalPages}
          className="bg-gray-300 px-4 py-2 rounded"
        >
          Next
        </button>
        <button
          onClick={() => setCurrentPage(totalPages)}
          disabled={currentPage === totalPages}
          className="bg-gray-300 px-4 py-2 rounded"
        >
          Last
        </button>
      </div>
    </div>
  );
}
