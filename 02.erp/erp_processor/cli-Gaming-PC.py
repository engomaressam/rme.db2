"""
Command-line interface for the ERP Processor.
"""
import argparse
import logging
import sys
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('erp_processor.log')
    ]
)
logger = logging.getLogger(__name__)

def process_cost_distribution(input_file: str, output_file: str) -> None:
    """Process cost distribution data."""
    logger.info(f"Processing cost distribution from {input_file} to {output_file}")
    # TODO: Implement cost distribution processing
    logger.info("Cost distribution processing completed")

def process_material_movement(input_file: str, output_file: str) -> None:
    """Process material movement data."""
    logger.info(f"Processing material movement from {input_file} to {output_file}")
    # TODO: Implement material movement processing
    logger.info("Material movement processing completed")

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ERP Data Processing Tool')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Cost distribution command
    cost_parser = subparsers.add_parser('cost-dist', help='Process cost distribution')
    cost_parser.add_argument('input_file', help='Input file path')
    cost_parser.add_argument('output_file', help='Output file path')
    
    # Material movement command
    matmov_parser = subparsers.add_parser('mat-mov', help='Process material movement')
    matmov_parser.add_argument('input_file', help='Input file path')
    matmov_parser.add_argument('output_file', help='Output file path')
    
    # Add version flag
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 0.1.0',
        help='Show version and exit'
    )
    
    # Add verbose flag
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args(args)

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    if args is None:
        args = sys.argv[1:]
    
    try:
        parsed_args = parse_args(args)
        
        if parsed_args.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
        
        if parsed_args.command == 'cost-dist':
            process_cost_distribution(parsed_args.input_file, parsed_args.output_file)
        elif parsed_args.command == 'mat-mov':
            process_material_movement(parsed_args.input_file, parsed_args.output_file)
        elif not parsed_args.command:
            logger.error("No command specified. Use -h for help.")
            return 1
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
            
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
