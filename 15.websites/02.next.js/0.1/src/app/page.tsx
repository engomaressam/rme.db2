import Container from "./components/Container";

export default function Home() {
  return (
    <Container>
      <h1 className="text-3xl font-bold text-gray-800 mb-6">
        Welcome to Expenditure Manager
      </h1>
      <p className="text-gray-600 mb-4">
        Easily add and view your expenditures.
      </p>
      <div className="flex space-x-4">
        <a
          href="/add-expenditure"
          className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
        >
          Add Expenditure
        </a>
        <a
          href="/expenditures"
          className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600"
        >
          View Expenditures
        </a>
      </div>
    </Container>
  );
}
