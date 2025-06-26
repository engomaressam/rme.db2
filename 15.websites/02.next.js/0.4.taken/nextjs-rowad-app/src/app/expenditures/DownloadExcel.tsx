"use client";

import ExcelJS from "exceljs";
import { saveAs } from "file-saver";
import { Button } from "@heroui/react";

interface DownloadExcelProps {
  data: any[];
  filename?: string;
}

export default function DownloadExcel({
  data,
  filename = "Expenditures.xlsx",
}: DownloadExcelProps) {
  const handleExport = async () => {
    if (data.length === 0) {
      alert("No data available to export!");
      return;
    }

    // 🔹 Create Excel Workbook & Worksheet
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet("Expenditures");

    // 🔹 Define Column Headers
    worksheet.columns = [
      { header: "Expenditure ID", key: "expenditureItemId", width: 20 },
      { header: "Project Name", key: "projectName", width: 30 },
      { header: "Task", key: "taskName", width: 30 },
      { header: "Transaction Source", key: "transactionSource", width: 30 },
      { header: "Quantity", key: "quantity", width: 15 },
      { header: "UOM", key: "uom", width: 15 },
      { header: "Description", key: "lineDesc", width: 40 },
    ];

    // 🔹 Style Header Row
    worksheet.getRow(1).eachCell((cell) => {
      cell.font = { bold: true, color: { argb: "FFFFFF" } };
      cell.fill = {
        type: "pattern",
        pattern: "solid",
        fgColor: { argb: "4F81BD" }, // Blue header background
      };
      cell.alignment = { horizontal: "center", vertical: "middle" };
    });

    // 🔹 Add Data Rows
    data.forEach((row) => worksheet.addRow(row));

    // 🔹 Create Excel File & Trigger Download
    const buffer = await workbook.xlsx.writeBuffer();
    const blob = new Blob([buffer], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    saveAs(blob, filename);
  };

  return (
    <Button
      color="secondary"
      variant="flat"
      onPress={handleExport}
      className="w-80 px-6 py-3 text-lg bg-primary-500 text-primary-50"
    >
      Download Excel
    </Button>
  );
}
