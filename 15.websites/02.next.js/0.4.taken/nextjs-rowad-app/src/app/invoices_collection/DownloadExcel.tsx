"use client";

import ExcelJS from "exceljs";
import { saveAs } from "file-saver";
import { Button } from "@heroui/react";

interface DownloadExcelProps {
  data: any[]; // Pass only filtered data
  filename?: string;
}

export default function DownloadExcel({
  data,
  filename = "Receipts.xlsx",
}: DownloadExcelProps) {
  const handleExport = async () => {
    if (data.length === 0) {
      alert("No data available to export!");
      return;
    }

    // 🔹 Create Excel Workbook & Worksheet
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet("Receipts");

    // 🔹 Define Column Headers
    worksheet.columns = [
      { header: "Receipt Number", key: "receiptNum", width: 20 },
      { header: "Organization ID", key: "orgId", width: 15 },
      { header: "Project Name", key: "receiptProjNam", width: 30 },
      { header: "Project Code", key: "receiptProjCode", width: 20 },
      { header: "Amount", key: "receiptAmount", width: 15 },
      { header: "Date", key: "receiptDate", width: 20 },
      { header: "Invoice Number", key: "invoiceNum", width: 20 },
      { header: "Applied Amount", key: "appliedAmount", width: 20 },
      { header: "Attribute 1", key: "attribute1", width: 20 },
      { header: "Total Adjustment", key: "newCalcTotalAdj", width: 20 },
      { header: "Currency", key: "currency", width: 10 },
      { header: "Transaction Amount", key: "transactionAmount", width: 20 },
      { header: "Total After Tax", key: "totalAfterTax", width: 20 },
      { header: "Amount To Collect", key: "calcAmountToCollect", width: 20 },
      { header: "Total Applied", key: "totalAmountApplied", width: 20 },
      { header: "Transaction Project", key: "trxPrjNam", width: 30 },
      { header: "Transaction Project Code", key: "trxPrjCode", width: 20 },
      { header: "Status", key: "transStatus", width: 15 },
      { header: "Transaction Type", key: "transType", width: 15 },
      { header: "Customer Number", key: "customerNum", width: 20 },
      { header: "Customer Name", key: "customerNam", width: 30 },
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

    // 🔹 Add Data Rows (Filtered Data)
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
      className="px-6 py-3 text-lg bg-blue-500 text-white hover:bg-blue-600 rounded-lg"
    >
      Download Excel
    </Button>
  );
}
