import { ApolloClient, InMemoryCache, HttpLink } from "@apollo/client";

const client = new ApolloClient({
  link: new HttpLink({
    uri: process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT, // Your GraphQL API
    credentials: "include",
  }),
  cache: new InMemoryCache(),
  ssrMode: typeof window === "undefined", // Enable SSR Mode for server-side fetching
});

export default client;
