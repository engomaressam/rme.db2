import sys
import os
import importlib.util
import json
import difflib
from typing import Dict, List, Tuple, Optional, Any

# Test question that works in rev.16 but not in rev.20
TEST_QUESTION = "whats the supplier with the highest amount in ring road project"

def load_module_from_file(file_path, module_name):
    """Load a Python module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def compare_entity_detection():
    """Compare entity detection between rev.16 and rev.20"""
    # Paths to the two versions
    rev16_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py"
    rev20_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Load the modules
    print("Loading rev.16 module...")
    rev16 = load_module_from_file(rev16_path, "rev16")
    
    print("Loading rev.20 module...")
    rev20 = load_module_from_file(rev20_path, "rev20")
    
    # Initialize the unique lists in both modules
    print("\nInitializing unique lists for both versions...")
    rev16.initialize_unique_lists()
    rev20.initialize_unique_lists()
    
    # Compare the unique project lists
    print("\n=== COMPARING UNIQUE PROJECT LISTS ===")
    print(f"Rev.16 projects: {len(rev16.UNIQUE_PROJECTS)}")
    print(f"Rev.20 projects: {len(rev20.UNIQUE_PROJECTS)}")
    
    # Find projects with "ring road" in their name
    rev16_ring_road = [p for p in rev16.UNIQUE_PROJECTS if "ring road" in p.lower()]
    rev20_ring_road = [p for p in rev20.UNIQUE_PROJECTS if "ring road" in p.lower()]
    
    print(f"\nRing Road projects in rev.16: {len(rev16_ring_road)}")
    for p in rev16_ring_road:
        print(f"  - {p}")
    
    print(f"\nRing Road projects in rev.20: {len(rev20_ring_road)}")
    for p in rev20_ring_road:
        print(f"  - {p}")
    
    # Test LLM entity detection
    print("\n=== TESTING LLM ENTITY DETECTION ===")
    print(f"Test question: '{TEST_QUESTION}'")
    
    print("\nRev.16 LLM detection:")
    rev16_llm_entities = rev16.detect_entities_with_llm(TEST_QUESTION)
    print(f"  Result: {rev16_llm_entities}")
    
    print("\nRev.20 LLM detection:")
    rev20_llm_entities = rev20.detect_entities_with_llm(TEST_QUESTION)
    print(f"  Result: {rev20_llm_entities}")
    
    # Test direct entity extraction
    print("\n=== TESTING DIRECT ENTITY EXTRACTION ===")
    
    print("\nRev.16 direct extraction:")
    rev16_project, rev16_confidence = rev16.extract_entity_from_question(TEST_QUESTION, "project", rev16.UNIQUE_PROJECTS)
    print(f"  Project: {rev16_project}")
    print(f"  Confidence: {rev16_confidence}")
    
    print("\nRev.20 direct extraction:")
    rev20_project, rev20_confidence = rev20.extract_entity_from_question(TEST_QUESTION, "project", rev20.UNIQUE_PROJECTS)
    print(f"  Project: {rev20_project}")
    print(f"  Confidence: {rev20_confidence}")
    
    # Test full question processing
    print("\n=== TESTING FULL QUESTION PROCESSING ===")
    
    print("\nRev.16 processing:")
    try:
        rev16_answer, rev16_query, rev16_columns, rev16_results, rev16_entities = rev16.process_question(TEST_QUESTION)
        print(f"  Detected entities: {rev16_entities}")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    print("\nRev.20 processing:")
    try:
        rev20_answer, rev20_query, rev20_columns, rev20_results, rev20_entities = rev20.process_question(TEST_QUESTION)
        print(f"  Detected entities: {rev20_entities}")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Compare the difflib behavior
    print("\n=== COMPARING DIFFLIB BEHAVIOR ===")
    
    # Test with "Ring Road" as input
    test_input = "Ring Road"
    
    print(f"\nDifflib matches for '{test_input}' in rev.16:")
    rev16_matches = difflib.get_close_matches(test_input, rev16.UNIQUE_PROJECTS, n=3, cutoff=0.5)
    for match in rev16_matches:
        print(f"  - {match} (score: {difflib.SequenceMatcher(None, test_input, match).ratio():.2f})")
    
    print(f"\nDifflib matches for '{test_input}' in rev.20:")
    rev20_matches = difflib.get_close_matches(test_input, rev20.UNIQUE_PROJECTS, n=3, cutoff=0.5)
    for match in rev20_matches:
        print(f"  - {match} (score: {difflib.SequenceMatcher(None, test_input, match).ratio():.2f})")

if __name__ == "__main__":
    compare_entity_detection()
