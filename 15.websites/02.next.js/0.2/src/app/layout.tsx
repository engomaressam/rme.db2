"use client";

import { ApolloProvider } from "@apollo/client";
import client from "./graphql/client";
import Navbar from "./components/Navbar";

import "./styles/globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ApolloProvider client={client}>
      <html lang="en">
        <body className="bg-grey-100">
          <Navbar />
          <main className="p-4">{children}</main>
        </body>
      </html>
    </ApolloProvider>
  );
}
