"use client";

import Link from "next/link";
import Image from "next/image";

export default function Navbar() {
  return (
    <nav className="bg-white shadow-md p-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <Link href="/">
          <Image src="/rowad-AI-logo.png" alt="Logo" width={150} height={40} />
        </Link>

        {/* Navigation Links */}
        <div className="space-x-6">
          <Link href="/" className="text-gray-700 hover:text-blue-500">
            Home
          </Link>
          <Link
            href="/add-expenditure"
            className="text-gray-700 hover:text-blue-500"
          >
            Add Expenditure
          </Link>
          <Link
            href="/expenditures"
            className="text-gray-700 hover:text-blue-500"
          >
            View Expenditures
          </Link>
        </div>
      </div>
    </nav>
  );
}
