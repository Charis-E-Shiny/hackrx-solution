"""
Test script for the HackRx LLM Query-Retrieval System
Run this to verify the system is working correctly
"""

import requests
import json
import os
import time

# Configuration
API_BASE_URL = "http://localhost:5000"
BEARER_TOKEN = "94adf1bd4f8978d3029fd88cb36c7f95a255d331301920caa50830f5f6c3ba26"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = requests.get(f"{API_BASE_URL}/health", headers=headers)
    
    if response.status_code == 200:
        print("✅ Health check passed")
        print(f"Response: {response.json()}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_main_query():
    """Test the main query endpoint with sample data"""
    print("\n🔍 Testing main query endpoint...")
    
    # Sample request matching the HackRx specification
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("📤 Sending request...")
    print(f"URL: {test_request['documents'][:50]}...")
    print(f"Questions: {len(test_request['questions'])} questions")
    
    start_time = time.time()
    response = requests.post(
        f"{API_BASE_URL}/api/v1/hackrx/run",
        headers=headers,
        json=test_request,
        timeout=120  # 2 minutes timeout
    )
    end_time = time.time()
    
    print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Query processing successful!")
        print(f"📊 Processed {len(result['answers'])} questions")
        
        # Display results
        for i, (question, answer) in enumerate(zip(test_request['questions'], result['answers'])):
            print(f"\n❓ Question {i+1}: {question}")
            print(f"💬 Answer: {answer}")
            
        return True
    else:
        print(f"❌ Query processing failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_authentication():
    """Test authentication with invalid token"""
    print("\n🔍 Testing authentication...")
    
    headers = {"Authorization": "Bearer invalid-token"}
    response = requests.get(f"{API_BASE_URL}/health", headers=headers)
    
    if response.status_code == 401:
        print("✅ Authentication test passed (correctly rejected invalid token)")
        return True
    else:
        print(f"❌ Authentication test failed: Expected 401, got {response.status_code}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting HackRx LLM Query-Retrieval System Tests")
    print("=" * 60)
    
    # Test basic connectivity
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print(f"⚠️  Server returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running with: uvicorn main:app --host 0.0.0.0 --port 5000")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_health_check():
        tests_passed += 1
    
    if test_authentication():
        tests_passed += 1
    
    if test_main_query():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")

if __name__ == "__main__":
    main()