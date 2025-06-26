"use client";

import { Providers } from "./providers"; // Wraps Apollo, Auth & HeroUI
import Navbar from "./components/Navbar";
import "./styles/globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-grey-100 min-h-screen">
        <Providers>
          <Navbar />
          <main className="p-4">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
