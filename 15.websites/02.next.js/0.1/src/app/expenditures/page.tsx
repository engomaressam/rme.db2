import Container from "../components/Container";
import ExpenditureTable from "./table";

export default function ExpendituresPage() {
  return (
    <Container>
      <h1 className="text-2xl font-bold text-gray-800 mb-4">
        All Expenditures
      </h1>
      <ExpenditureTable />
    </Container>
  );
}
