"use client";

import { Button, Card, CardBody } from "@heroui/react";
import Container from "./components/Container";

export default function Home() {
  return (
    <Container>
      <Card shadow="lg" className="p-6 bg-white dark:bg-gray-800">
        <CardBody>
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
            Welcome to Expenditure Manager
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            Easily add and view your expenditures.
          </p>
          <div className="flex space-x-4">
            <Button
              as="a"
              href="/add-expenditure"
              color="primary"
              variant="flat"
            >
              Add Expenditure
            </Button>
            <Button
              as="a"
              href="/expenditures"
              color="secondary"
              variant="flat"
            >
              View Expenditures
            </Button>
          </div>
        </CardBody>
      </Card>
    </Container>
  );
}
