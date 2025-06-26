"use client";

import { HeroUIProvider } from "@heroui/react";
import { ApolloProvider } from "@apollo/client";
import client from "./graphql/client";
import { AuthProvider } from "./utils/AuthProvider";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <ApolloProvider client={client}>
        <HeroUIProvider>{children}</HeroUIProvider>
      </ApolloProvider>
    </AuthProvider>
  );
}
