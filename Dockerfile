FROM python:3.10-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend_py/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend_py/ /code/

# Make sure the app runs on port 7860 which is expected by Hugging Face Spaces
ENV PORT=7860

# Expose the port
EXPOSE 7860

CMD ["python", "app_simple.py"]