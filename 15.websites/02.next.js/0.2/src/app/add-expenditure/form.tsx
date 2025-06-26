"use client";

import { useMutation } from "@apollo/client";
import React, { useState } from "react";
import { CREATE_EXPENDITURE } from "../graphql/mutations";

interface FormData {
  expenditureItemId: string;
  transactionSource: string;
  projectId: string;
  projectNum: string;
  projectName: string;
  projectType: string;
  taskId: string;
  taskNum: string;
  taskName: string;
  quantity: string;
  uom: string;
  lineDesc: string;
}

const ExpenditureForm: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    expenditureItemId: "",
    transactionSource: "",
    projectId: "",
    projectNum: "",
    projectName: "",
    projectType: "",
    taskId: "",
    taskNum: "",
    taskName: "",
    quantity: "",
    uom: "",
    lineDesc: "",
  });

  const [createExpenditure, { loading, error }] = useMutation(
    CREATE_EXPENDITURE,
    {
      onCompleted: () => {
        alert("Expenditure created successfully!");
        resetForm();
      },
      onError: (err) => {
        alert(`An error occurred: ${err.message}`);
      },
    }
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: ["projectId", "projectNum", "taskId", "quantity"].includes(name)
        ? value === ""
          ? ""
          : Number(value) // Convert to number if field is numeric
        : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log("Form data:", formData);
    try {
      const variables = {
        expenditureItemId: formData.expenditureItemId,
        transactionSource: formData.transactionSource,
        projectId: parseInt(formData.projectId),
        projectNum: parseInt(formData.projectNum),
        projectName: formData.projectName,
        projectType: formData.projectType,
        taskId: parseInt(formData.taskId),
        taskNum: formData.taskNum,
        taskName: formData.taskName,
        quantity: parseFloat(formData.quantity),
        uom: formData.uom,
        lineDesc: formData.lineDesc,
      };
      console.log("Variables:", variables);
      await createExpenditure({ variables });
    } catch (error) {
      console.error("Error creating expenditure:", error);
    }
  };

  const resetForm = () => {
    setFormData({
      expenditureItemId: "",
      transactionSource: "",
      projectId: "",
      projectNum: "",
      projectName: "",
      projectType: "",
      taskId: "",
      taskNum: "",
      taskName: "",
      quantity: "",
      uom: "",
      lineDesc: "",
    });
  };

  return (
    <div className="p-4 border">
      <h2 className="text-xl font-bold">Add Expenditure</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        {Object.entries(formData).map(([key, value]) => (
          <input
            key={key}
            type={key === "quantity" || key.includes("Id") ? "number" : "text"}
            name={key}
            placeholder={
              key.charAt(0).toUpperCase() +
              key
                .slice(1)
                .replace(/([A-Z])/g, " $1")
                .trim()
            }
            value={value}
            onChange={handleChange}
            className="border p-2 w-full"
            required
          />
        ))}
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2"
          disabled={loading}
        >
          {loading ? "Creating..." : "Create Expenditure"}
        </button>
      </form>
      {error && <p className="text-red-500">Error: {error.message}</p>}
    </div>
  );
};

export default ExpenditureForm;
