import DownloadExcel from "./DownloadExcel";
import Container from "../components/Container";
import ExpenditureTable from "./table";

export default function ExpendituresPage() {
  return (
    <Container>
      {/* ✅ Expenditure Table */}
      <ExpenditureTable />
    </Container>
  );
}
