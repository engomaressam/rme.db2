"use client";

import { useMutation } from "@apollo/client";
import React, { useState, useEffect } from "react";
import { CREATE_EXPENDITURE } from "../graphql/mutations";
import { Input, Button, Card, CardBody, Spinner, Alert } from "@heroui/react";

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

  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [createExpenditure, { loading }] = useMutation(CREATE_EXPENDITURE, {
    onCompleted: () => {
      setSuccessMessage("Expenditure created successfully!");
      resetForm();
    },
    onError: (err) => {
      setErrorMessage(`An error occurred: ${err.message}`);
    },
  });

  useEffect(() => {
    if (successMessage || errorMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage(null);
        setErrorMessage(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [successMessage, errorMessage]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: ["projectId", "projectNum", "taskId", "quantity"].includes(name)
        ? value === "" || isNaN(Number(value))
          ? ""
          : parseFloat(value) || 0 // Ensure numeric values don’t break
        : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const variables = {
        expenditureItemId: formData.expenditureItemId,
        transactionSource: formData.transactionSource,
        projectId: parseInt(formData.projectId) || 0,
        projectNum: parseInt(formData.projectNum) || 0,
        projectName: formData.projectName,
        projectType: formData.projectType,
        taskId: parseInt(formData.taskId) || 0,
        taskNum: formData.taskNum,
        taskName: formData.taskName,
        quantity: parseFloat(formData.quantity) || 0,
        uom: formData.uom,
        lineDesc: formData.lineDesc,
      };
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
    <Card shadow="lg" className="p-6 bg-white dark:bg-gray-800">
      <CardBody>
        <h2 className="text-2xl font-bold mb-6 text-gray-800 dark:text-white">
          Add Expenditure
        </h2>

        {/* Controlled Alert for Success/Error */}
        {successMessage && (
          <Alert
            color="success"
            className="mb-4"
            onClose={() => setSuccessMessage(null)}
          >
            {successMessage}
          </Alert>
        )}
        {errorMessage && (
          <Alert
            color="danger"
            className="mb-4"
            onClose={() => setErrorMessage(null)}
          >
            {errorMessage}
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {Object.entries(formData).map(([key, value]) => (
            <Input
              key={key}
              type={
                key === "quantity" || key.includes("Id") ? "number" : "text"
              }
              name={key}
              placeholder={key
                .replace(/([A-Z])/g, " $1")
                .replace(/^./, (str) => str.toUpperCase())}
              value={value}
              onChange={handleChange}
              className="w-full"
              required
            />
          ))}

          <Button
            type="submit"
            color="primary"
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <Spinner size="sm" color="white" />
            ) : (
              "Create Expenditure"
            )}
          </Button>
        </form>
      </CardBody>
    </Card>
  );
};

export default ExpenditureForm;
