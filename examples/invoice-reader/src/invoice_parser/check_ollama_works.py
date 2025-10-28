#!/usr/bin/env python3
"""
Simple debug script to test Ollama connection and response time.
"""

import time
import ollama
import json


def image2text(image_path: str, model: str) -> str | None:
    """Test Ollama with energy bill image to extract monthly amount."""    
    print(f"Testing Ollama connection with image: {image_path}")
    print(f"Using model: {model}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # pull model
        ollama.pull(model=model)

        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': 'What is the monthly amount shown on this energy bill? Please provide the amount, currency and type of bill in a concise format.',
                    'images': [image_path]
                }
            ]
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        response = response['message']['content']
        
        print(f"✅ Success!")
        print(f"Response time: {response_time:.2f} seconds")
        print(f"Response: {response}")
        
        return response
    
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"❌ Error after {response_time:.2f} seconds:")
        print(f"Error: {e}")


def text2json(text: str, model: str) -> dict | None:
    """Extract structured data from text using LFM2-1.2B-Extract model."""
    if not extracted_text:
        return None
        
    print(f"Extracting structured data from text...")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        system_prompt = f"""Identify and extract the following information. Present as a JSON object.
        
        utility: Type of utility (e.g., electricity, water, gas).
        amount: Amount shown on the bill. Only provide the numeric value.
        currency: Currency of the amount (e.g., USD, EUR).
        """
        
        # pull model
        ollama.pull(model=model)

        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': text
                }
            ]
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        response_content = response['message']['content']
        
        # Try to parse the JSON response
        try:
            json_data = json.loads(response_content)
            print(f"✅ Extraction successful!")
            print(f"Response time: {response_time:.2f} seconds")
            print(f"Extracted data: {json_data}")
            return json_data
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Response was not valid JSON")
            print(f"Raw response: {response_content}")
            return None
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"❌ Error after {response_time:.2f} seconds:")
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    
    extracted_text = image2text(
        image_path="inputs/Sample-electric-Bill-2023.jpg",
        model="hf.co/LiquidAI/LFM2-VL-1.6B-GGUF:F16",
    )

    json = text2json(
        text=extracted_text,
        model="hf.co/LiquidAI/LFM2-1.2B-Extract-GGUF:F16",    
    )

