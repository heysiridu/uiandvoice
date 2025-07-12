#!/usr/bin/env python3
"""
Simple test script to verify Gemini API configuration
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_api():
    """Test if Gemini API is properly configured"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please create a .env file with your Gemini API key")
        return False
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model instance
        model = genai.GenerativeModel('gemini-pro')
        
        # Test with a simple prompt
        response = model.generate_content("Hello! Please respond with 'Gemini API is working correctly!'")
        
        print("✅ Gemini API is working correctly!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        print("\nPossible solutions:")
        print("1. Check if your API key is correct")
        print("2. Ensure the Gemini API is enabled in Google Cloud Console")
        print("3. Check your internet connection")
        print("4. Verify billing is set up in Google Cloud Console")
        return False

if __name__ == "__main__":
    print("Testing Gemini API configuration...")
    test_gemini_api() 