"""
Ollama API Status Checker
This script checks the status of the Ollama API and tests available models.
"""

import requests
import json
import time
import logging
import argparse
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ollama_status.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OllamaStatus")

def check_ollama_api(api_url):
    """Check if the Ollama API is accessible"""
    try:
        logger.info(f"Checking Ollama API at {api_url}...")
        response = requests.get(f"{api_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                logger.info(f"Ollama API is accessible. Found {len(models)} models:")
                for model in models:
                    logger.info(f"  - {model.get('name')}")
                return True, models
            else:
                logger.warning("Ollama API is accessible but no models found")
                return True, []
        else:
            logger.error(f"Ollama API returned status code {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Ollama API at {api_url}")
        return False, []
    except Exception as e:
        logger.error(f"Error checking Ollama API: {str(e)}")
        return False, []

def test_model(api_url, model_name):
    """Test a specific model with a simple prompt"""
    try:
        logger.info(f"Testing model {model_name}...")
        
        # Simple test prompt
        prompt = "Hello, please respond with a short greeting."
        
        # Prepare the request
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # Set timeout
        timeout = 30
        
        # Send the request
        start_time = time.time()
        response = requests.post(
            f"{api_url}/api/generate",
            json=request_data,
            timeout=timeout
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json.get("response", "")
            
            logger.info(f"Model {model_name} responded in {response_time:.2f} seconds")
            logger.info(f"Response: {response_text[:100]}..." if len(response_text) > 100 else f"Response: {response_text}")
            return True, response_time, response_text
        else:
            logger.error(f"Model {model_name} test failed with status code {response.status_code}")
            return False, response_time, None
    except requests.exceptions.Timeout:
        logger.error(f"Timeout testing model {model_name}")
        return False, timeout, None
    except Exception as e:
        logger.error(f"Error testing model {model_name}: {str(e)}")
        return False, 0, None

def main():
    parser = argparse.ArgumentParser(description="Check Ollama API status and test models")
    parser.add_argument("--url", default="http://10.10.12.202:11434", help="Ollama API URL")
    parser.add_argument("--test-all", action="store_true", help="Test all available models")
    parser.add_argument("--model", help="Specific model to test")
    
    args = parser.parse_args()
    
    # Check if API is accessible
    api_accessible, models = check_ollama_api(args.url)
    
    if not api_accessible:
        logger.error("Ollama API is not accessible")
        sys.exit(1)
    
    # Test models
    if args.test_all and models:
        logger.info("Testing all available models...")
        results = []
        
        for model in models:
            model_name = model.get("name")
            success, response_time, _ = test_model(args.url, model_name)
            results.append({
                "model": model_name,
                "success": success,
                "response_time": response_time
            })
        
        # Print summary
        logger.info("\nTest Results Summary:")
        for result in results:
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            logger.info(f"{result['model']}: {status} - Response time: {result['response_time']:.2f}s")
    
    elif args.model:
        # Test specific model
        success, response_time, response_text = test_model(args.url, args.model)
        if success:
            logger.info(f"Model {args.model} test passed")
        else:
            logger.error(f"Model {args.model} test failed")
            sys.exit(1)
    
    else:
        logger.info("No models tested. Use --test-all to test all models or --model to test a specific model.")

if __name__ == "__main__":
    main()
