import Container from "../components/Container";
import InvoicesCollectionTable from "./table";

export default function ExpendituresPage() {
  return (
    <Container>
      {/* ✅ Expenditure Table */}
      <InvoicesCollectionTable />
    </Container>
  );
}
