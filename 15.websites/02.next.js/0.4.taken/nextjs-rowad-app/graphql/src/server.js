import dotenv from "dotenv";
import express from "express";
import helmet from "helmet";
// import { rateLimit } from "express-rate-limit";
import { ApolloServer } from "apollo-server-express";
import typeDefs from "./schema/typeDefs.js";
import resolvers from "./schema/resolvers.js";
import cors from "cors";
import jwt from "jsonwebtoken";

dotenv.config();

const app = express();

// Security middleware
app.use(helmet());

// Rate limiting
// const limiter = rateLimit({
//   windowMs: 15 * 60 * 1000, // 15 minutes
//   max: 100, // limit each IP to 100 requests per windowMs
// });
// app.use(limiter);

// CORS configuration
const corsOptions = {
  origin: [
    "http://localhost:3000",
    "https://studio.apollographql.com",
    "http://10.10.12.223:3000",
  ],
  credentials: true,
  methods: "GET,POST,OPTIONS", // Allowed methods
  allowedHeaders: "Content-Type, Authorization", // Allowed headers
};
app.use(cors(corsOptions));

const dynamicCors = (req, res, next) => {
  const origin = req.header("Origin");
  const allowedOrigins = ["https://example.com", "https://anotherdomain.com"];
  if (allowedOrigins.includes(origin)) {
    res.header("Access-Control-Allow-Origin", origin);
  }
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept, Authorization"
  );
  next();
};

app.use(dynamicCors);

// Middleware to Extract User from Token
app.use((req, res, next) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (token) {
    try {
      req.user = jwt.verify(token, process.env.JWT_SECRET);
    } catch (err) {
      console.warn("Invalid token:", err.message);
    }
  }
  next();
});

// Apollo Server with Context for Authentication
const server = new ApolloServer({
  typeDefs,
  resolvers,
  playground: true,
  introspection: true,
  context: ({ req }) => {
    // Add authentication/authorization context here
    return { user: req.user };
  },
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send("Something broke!");
});

// Process error handlers
process.on("uncaughtException", (err) => {
  console.error("Uncaught Exception:", err);
  process.exit(1);
});

process.on("unhandledRejection", (err) => {
  console.error("Unhandled Rejection:", err);
  process.exit(1);
});

async function startServer() {
  try {
    await server.start();
    server.applyMiddleware({
      app,
      path: "/graphql",
      cors: {
        origin: [
          "http://localhost:3000",
          "https://studio.apollographql.com",
          "http://10.10.12.223:3000",
        ],
        credentials: true,
      },
    });

    const PORT = process.env.PORT || 4000;
    app.listen(PORT, () => {
      console.log(`🚀 Server ready at http://localhost:${PORT}/graphql`);
    });
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
}

startServer();
