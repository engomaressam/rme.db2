import { ApolloClient, InMemoryCache, HttpLink } from "@apollo/client";
import { setContext } from "@apollo/client/link/context";

const GRAPHQL_ENDPOINT = process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT;

if (!GRAPHQL_ENDPOINT) {
  throw new Error(
    "GraphQL endpoint is missing. Check NEXT_PUBLIC_GRAPHQL_ENDPOINT."
  );
}

const httpLink = new HttpLink({
  uri: GRAPHQL_ENDPOINT,
  credentials: "include", // Allow cookies & authentication headers
});

const authLink = setContext((_, { headers }) => {
  if (typeof window === "undefined") {
    return { headers }; // Don't access localStorage on the server
  }

  const token = localStorage.getItem("auth_token");

  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : "",
    },
  };
});

const client = new ApolloClient({
  link: authLink.concat(httpLink),
  cache: new InMemoryCache(),
  ssrMode: typeof window === "undefined", // Enable SSR Mode for Next.js
});

export default client;
