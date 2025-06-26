import re

def create_sql_fix():
    """Fix the SQL generation logic in rev.20 to correctly use GROUP BY."""
    file_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

    # Find the create_sql_query_with_llm function and its system_message
    func_pattern = r'(def create_sql_query_with_llm\(.*?\):.*?system_message = f"""You are a SQL query generation system.*?""")'
    match = re.search(func_pattern, code, re.DOTALL)

    if not match:
        print("Could not find the system_message in create_sql_query_with_llm.")
        return None

    original_section = match.group(1)

    # Add the explicit GROUP BY instruction
    group_by_instruction = ("\\n\\nIMPORTANT SQL RULE: When a query involves aggregation (like SUM, COUNT, MAX, AVG) on one column "
                            "and selects another column, you MUST use a GROUP BY clause on the selected non-aggregated column(s). "
                            "For example, to 'find the supplier with the highest amount', the query must be 'SELECT \"VENDOR_NAME\" ... GROUP BY \"VENDOR_NAME\" ORDER BY SUM(\"LINE_AMOUNT\") DESC'.")

    # Insert the instruction into the system_message f-string
    # We add it before the final triple quote
    modified_section = original_section[:-3] + group_by_instruction + '"""'

    modified_code = code.replace(original_section, modified_section)

    # Change the port to avoid conflicts
    modified_code = modified_code.replace("port=7869", "port=7871")
    modified_code = modified_code.replace("http://localhost:7869", "http://localhost:7871")

    new_file_path = file_path.replace(".py", ".sqlfix.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)

    print(f"SQL generation fix saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Creating fix for the SQL GROUP BY issue...")
    fixed_file = create_sql_fix()
    if fixed_file:
        print(f"\nFix completed. The updated code is in: {fixed_file}")
        print("\nTo use this version:")
        print(f'python "{fixed_file}"')
        print("\nThis version should resolve the SQL error, but may still detect the wrong project.")
        print("We will diagnose the project detection issue next.")
