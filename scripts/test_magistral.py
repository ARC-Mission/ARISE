#!/usr/bin/env python3
"""
Test script for Magistral API connectivity.
Verifies that the reasoning LLM is accessible and working.
"""

import os
import sys

def test_magistral_api():
    """Test Magistral API connection."""
    
    print("\n" + "="*50)
    print("Testing Magistral API Connection")
    print("="*50 + "\n")
    
    # Check API key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not set")
        print("\nTo set the API key:")
        print("  export MISTRAL_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://console.mistral.ai/")
        return False
    
    print(f"✓ API key found (length: {len(api_key)})")
    
    # Try importing the SDK
    try:
        from mistralai import Mistral
        print("✓ mistralai SDK imported")
    except ImportError:
        print("❌ mistralai SDK not installed")
        print("\nInstall with: pip install mistralai")
        return False
    
    # Test connection
    print("\nTesting API connection...")
    
    try:
        client = Mistral(api_key=api_key)
        
        # Simple test query
        response = client.chat.complete(
            model="magistral-medium-latest",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond briefly."
                },
                {
                    "role": "user",
                    "content": "Say 'Magistral is ready' and nothing else."
                }
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✓ API response: {result}")
        
        # Test reasoning capabilities
        print("\nTesting reasoning mode...")
        
        response = client.chat.complete(
            model="magistral-medium-latest",
            messages=[
                {
                    "role": "user",
                    "content": "What is 15 + 27? Think step by step."
                }
            ],
            max_tokens=200
        )
        
        result = response.choices[0].message.content
        print(f"✓ Reasoning response received ({len(result)} chars)")
        
        if "42" in result:
            print("✓ Correct answer found in response")
        
        return True
        
    except Exception as e:
        print(f"❌ API error: {e}")
        return False


def test_local_magistral():
    """Test local Magistral deployment (Ollama or vLLM)."""
    
    print("\n" + "="*50)
    print("Testing Local Magistral Deployment")
    print("="*50 + "\n")
    
    # Test Ollama
    print("Checking Ollama...")
    import subprocess
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ Ollama is running")
            if "magistral" in result.stdout.lower():
                print("✓ Magistral model found in Ollama")
            else:
                print("⚠ Magistral not found. Install with:")
                print("  ollama pull hf.co/mistralai/Magistral-Small-2506_gguf:Q4_K_M")
        else:
            print("⚠ Ollama not available")
            
    except FileNotFoundError:
        print("⚠ Ollama not installed")
    except subprocess.TimeoutExpired:
        print("⚠ Ollama timed out")
    except Exception as e:
        print(f"⚠ Error checking Ollama: {e}")
    
    # Test vLLM endpoint
    print("\nChecking vLLM endpoint...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✓ vLLM server is running")
        else:
            print(f"⚠ vLLM returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠ vLLM server not running on localhost:8000")
    except Exception as e:
        print(f"⚠ Error checking vLLM: {e}")
    
    return True


def main():
    print("\n" + "="*60)
    print("   ARC-M Magistral Connectivity Test")
    print("="*60)
    
    # Test API
    api_ok = test_magistral_api()
    
    # Test local
    local_ok = test_local_magistral()
    
    # Summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    
    if api_ok:
        print("✓ Magistral API: Ready")
    else:
        print("✗ Magistral API: Not configured")
    
    print("\nRecommendation:")
    if api_ok:
        print("  Use API mode for development (faster, no GPU needed)")
        print("  Set magistral.deployment: 'api' in config")
    else:
        print("  1. Get API key from https://console.mistral.ai/")
        print("  2. Set MISTRAL_API_KEY environment variable")
        print("  OR")
        print("  Deploy Magistral Small locally with Ollama/vLLM")


if __name__ == "__main__":
    main()
