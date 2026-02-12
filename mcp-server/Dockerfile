# Use the official Node.js runtime as the base image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json (if available) to the working directory
COPY mcp-server/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy the rest of the application code to the working directory
COPY mcp-server/ .

# Expose the port that the application will run on
EXPOSE 8080

# Define the command to run the application
CMD ["npm", "start"]