import dotenv from "dotenv";
import express from "express";
import helmet from "helmet";
import { ApolloServer } from "apollo-server-express";
import typeDefs from "./schema/typeDefs.js";
import resolvers from "./schema/resolvers.js";
import cors from "cors";
import mysql from "mysql2/promise";

dotenv.config();

const app = express();

// Security middleware
app.use(helmet());

// CORS configuration
const corsOptions = {
  origin: [
    "http://localhost:3000",
    "https://studio.apollographql.com",
    "http://10.10.12.223:3000",
  ],
  credentials: true,
  methods: "GET,POST,OPTIONS",
  allowedHeaders: "Content-Type, Authorization",
};
app.use(cors(corsOptions));

// MySQL Database Connection
const db = await mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

// GraphQL Server
const server = new ApolloServer({
  typeDefs,
  resolvers,
  introspection: true,  // ✅ Enables schema inspection
  playground: true,  // ✅ Enables the GraphQL Playground
  context: ({ req }) => {
    return { user: req.user };
  },
});

async function startServer() {
  await server.start();
  server.applyMiddleware({
    app,
    path: "/graphql",
    cors: corsOptions,
  });

  const PORT = process.env.PORT || 5001;
  app.listen(PORT, () => {
    console.log(`🚀 Server ready at http://localhost:${PORT}/graphql`);
    console.log(`🌍 Accessible via http://${process.env.SERVER_HOST}:${PORT}/graphql`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
