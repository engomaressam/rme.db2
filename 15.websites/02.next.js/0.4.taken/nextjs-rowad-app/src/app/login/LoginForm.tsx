"use client";

import { useMutation } from "@apollo/client";
import { LOGIN_USER } from "../graphql/mutations";
import { useState, useEffect } from "react";
import { useAuth } from "../utils/AuthProvider";
import { Card, CardBody, Button, Input, Alert } from "@heroui/react";

export default function LoginForm() {
  const { login } = useAuth();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    name: "", // Added name field
  });

  const [errorMessage, setErrorMessage] = useState("");
  const [loginMutation, { loading }] = useMutation(LOGIN_USER);

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(""), 3000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage("");

    if (!formData.email || !formData.password) {
      setErrorMessage("Both email and password are required.");
      return;
    }

    try {
      const { data } = await loginMutation({ variables: formData });

      if (data?.loginUser?.token) {
        login(data.loginUser.token, formData.name, formData.email); // Store token, name, and email, then redirect
      } else {
        setErrorMessage("Invalid credentials. Try again.");
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Something went wrong.");
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100 px-4">
      <Card className="p-6 w-full max-w-md">
        <CardBody>
          <h2 className="text-2xl font-bold text-center mb-4">Login</h2>

          {/* Error Alert (Auto-hides in 3 seconds) */}
          {errorMessage && (
            <Alert
              variant="solid"
              color="danger"
              onClose={() => setErrorMessage("")}
            >
              {errorMessage}
            </Alert>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              type="email"
              label="Email"
              placeholder="Enter your email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
            />
            <Input
              type="password"
              label="Password"
              placeholder="Enter your password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
            />
            <Button
              type="submit"
              variant="solid"
              color="primary"
              isLoading={loading}
              className="w-full"
            >
              {loading ? "Logging in..." : "Login"}
            </Button>
          </form>
        </CardBody>
      </Card>
    </div>
  );
}
