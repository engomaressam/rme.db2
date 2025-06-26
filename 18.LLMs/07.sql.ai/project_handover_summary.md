# PO Query AI Chatbot Project Summary

## Project Overview
- **Project**: PO (Purchase Order) Query AI chatbot 
- **Current Version**: Rev.14
- **Database**: MySQL on `10.10.11.242`, database `RME_TEST`
- **Main Table**: `po_followup_merged` (contains merged data from `RME_PO_Follow_Up_Report` and `po_terms`)
- **LLM**: Ollama Gemma3 running locally on GPU

## Recent Progress

### Data Verification
- Confirmed data integrity of merged `po_followup_merged` table
- Verified all 861,403 rows from original table were preserved
- Identified 107,823 unique PO numbers in merged table
- Confirmed terms integration from `po_terms` into `po_followup_merged`

### Chatbot Development
- Created Rev.14 with improved entity detection for projects and suppliers
- Implemented 3-tier entity matching system:
  1. Direct matching for exact project/supplier names
  2. Keyword matching for significant words in names
  3. N-gram based fuzzy matching as fallback
- Fixed SQL query generation to prevent markdown syntax errors
- Added entity detection display to interface to show users what entities were detected

## Current Files
- `14.po.followup.query.ai.rev.14.on.gpu.gemma3.py` - Latest chatbot with entity detection
- Various data validation scripts (e.g., `check_terms_direct.py`, `terms_percentage.py`)

## Technical Details

### Database Connection
```python
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}
```

### Core Functionality
- Entity detection to match partial project/supplier names to database values
- SQL query generation via Ollama LLM
- Natural language answer generation from query results
- Display of detected entities so users can see what was matched

## Outstanding Tasks
1. Test Rev.14 with varied user queries to validate entity detection
2. Fine-tune entity matching thresholds if needed
3. Add detailed test cases for different query types
4. Consider moving database credentials to environment variables for security
5. Improve SQL error handling and feedback

## Current Challenges
- Entity detection sometimes fails on partial or differently formatted project/supplier names
- SQL query generation may occasionally add unwanted markdown formatting
- Need additional test cases to evaluate overall system performance

## Next Session Goals
- Validate Rev.14 fixes for entity detection
- Test with more complex queries
- Consider adding query templates for common question types
- Potentially create a dev mode that shows confidence scores for entity detection

This summary represents the current state of the project as of June 16, 2025.
