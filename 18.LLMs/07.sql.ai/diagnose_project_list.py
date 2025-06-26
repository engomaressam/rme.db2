import sys
import importlib.util
import difflib

def load_module_from_file(file_path, module_name):
    """Loads a python module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"Error: Could not load spec for module {module_name} from {file_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading module {module_name}: {e}")
        return None

def run_diagnostics():
    """Runs diagnostics on the project list and difflib matching."""
    print("--- Running Project Detection Diagnostics for rev.20 ---")
    rev20_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    print(f"Loading module from: {rev20_path}")
    rev20 = load_module_from_file(rev20_path, "rev20")
    
    if not rev20:
        return

    print("\n1. Initializing unique project list...")
    try:
        rev20.initialize_unique_lists()
        project_list = rev20.UNIQUE_PROJECTS
        print(f"Initialization complete. Found {len(project_list)} unique projects.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    print("\n2. Checking for 'RING ROAD BRIDGE-0122' in the list...")
    project_to_find = "RING ROAD BRIDGE-0122"
    if project_to_find in project_list:
        print(f"  [SUCCESS] '{project_to_find}' was FOUND in the loaded project list.")
    else:
        print(f"  [FAILURE] '{project_to_find}' was NOT FOUND in the loaded project list.")
        print("  This is the primary reason for the matching failure.")

    print("\n3. Performing difflib check...")
    search_term = "Ring Road"
    print(f"   Searching for close matches of '{search_term}'")
    
    matches = difflib.get_close_matches(search_term, project_list, n=5, cutoff=0.4)

    print(f"   Found {len(matches)} match(es) with cutoff=0.4:")
    if not matches:
        print("   No matches found.")
    else:
        for match in matches:
            ratio = difflib.SequenceMatcher(None, search_term.lower(), match.lower()).ratio()
            print(f"     - '{match}' (score: {ratio:.4f})")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    run_diagnostics()
