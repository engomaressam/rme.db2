import Container from "../components/Container";
import ExpenditureForm from "./form";

export default function AddExpenditurePage() {
  return (
    <Container>
      <h1 className="text-2xl font-bold text-gray-800 mb-4">Add Expenditure</h1>
      <ExpenditureForm />
    </Container>
  );
}
