const projectDropdown = document.getElementById('project-dropdown');
const tableContainer = document.getElementById('app-container');

projectDropdown.addEventListener('change', async function () {
  const selectedProject = this.value;

  // Fetch data for the selected project
  const tableData = await getTableData(selectedProject);

  // Display the table data
  if (tableData && tableData.length > 0) {
    const tableHTML = createTable(tableData);
    tableContainer.innerHTML = tableHTML;
  } else {
    // Handle no data case (optional: display a message)
    tableContainer.innerHTML = '<p>No data found for this project.</p>';
  }
});

async function getTableData(project) {
  // Replace with your actual logic for fetching data from Excel files
  // This is a placeholder example using fetch (adapt based on your needs)
  const response = await fetch(`/data/${project}.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch data for project ${project}`);
  }
  return await response.json();
}

function createTable(tableData) {
  // Function to build the HTML table structure based on your data
  let tableHTML = '<table>';
  tableHTML += '<tr>';
  for (const header of Object.keys(tableData[0])) {
    tableHTML += `<th>${header}</th>`;
  }
  tableHTML += '</tr>';

  for (const row of tableData) {
    tableHTML += '<tr>';
    for (const value of Object.values(row)) {
      tableHTML += `<td>${value}</td>`;
    }
    tableHTML += '</tr>';
  }

  tableHTML += '</table>';
  return tableHTML;
}
