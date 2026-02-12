#!/usr/bin/env python3
"""
Entrypoint for Hugging Face Spaces
"""
import os
import sys
from app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # Hugging Face Spaces uses port 7860
    uvicorn.run(app, host="0.0.0.0", port=port)