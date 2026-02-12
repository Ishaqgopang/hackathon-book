# Use the official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY backend_py/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY backend_py/ .

# Expose the port that the application will run on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "-m", "src.main"]