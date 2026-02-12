import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIService:
    def __init__(self):
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/models"
        
        if not self.hf_token:
            print("Warning: HUGGING_FACE_TOKEN not set. AI features will be limited.")

    async def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize text using Hugging Face models"""
        if not self.hf_token:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        try:
            response = requests.post(
                f"{self.api_url}/facebook/bart-large-cnn",
                headers=headers,
                json={
                    "inputs": text,
                    "parameters": {
                        "max_length": max_length,
                        "min_length": 30,
                        "do_sample": False
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result[0]["summary_text"] if isinstance(result, list) and len(result) > 0 else text[:max_length] + "..."
            else:
                print(f"Hugging Face API error: {response.status_code} - {response.text}")
                return text[:max_length] + "..."
        except Exception as e:
            print(f"Error calling Hugging Face API: {str(e)}")
            return text[:max_length] + "..."

    async def classify_sentiment(self, text: str) -> List[Dict]:
        """Classify sentiment of text using Hugging Face models"""
        if not self.hf_token:
            return [{"label": "NEUTRAL", "score": 0.5}]
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        try:
            response = requests.post(
                f"{self.api_url}/cardiffnlp/twitter-roberta-base-sentiment-latest",
                headers=headers,
                json={"inputs": text},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result[0] if isinstance(result, list) and len(result) > 0 else []
            else:
                print(f"Hugging Face API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error calling Hugging Face API: {str(e)}")
            return []

    async def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using Hugging Face models"""
        if not self.hf_token:
            return f"Generated text would appear here based on prompt: {prompt}"
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        try:
            response = requests.post(
                f"{self.api_url}/gpt2",
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_length,
                        "return_full_text": False
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result[0]["generated_text"] if isinstance(result, list) and len(result) > 0 else ""
            else:
                print(f"Hugging Face API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Error calling Hugging Face API: {str(e)}")
            return ""

# Create a global instance
ai_service = AIService()