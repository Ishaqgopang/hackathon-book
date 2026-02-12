# MongoDB Setup Guide

This guide will help you set up MongoDB for the Hackathon Book Backend.

## Option 1: Install MongoDB Locally (Windows)

### Method A: Using MongoDB MSI Installer
1. Go to [MongoDB Download Center](https://www.mongodb.com/try/download/community)
2. Download the MongoDB Community Server for Windows
3. Run the installer and follow the installation wizard
4. During installation, make sure to check "Install MongoDB as a Service"
5. Complete the installation

### Method B: Using Chocolatey (if you have it installed)
```bash
choco install mongodb
```

### Start MongoDB Service
After installation, MongoDB should start automatically. If not:
1. Open Command Prompt as Administrator
2. Run: `net start MongoDB`

## Option 2: Use MongoDB Atlas (Cloud)

### Create a Free Cluster
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas/database)
2. Sign up for a free account
3. Click "Build a Database"
4. Choose "Free" tier (M0 Sandbox)
5. Select your preferred cloud provider and region
6. Name your cluster (e.g., "hackathon-book-cluster")
7. Click "Create Cluster"

### Configure Database Access
1. Go to "Database Access" in the left sidebar
2. Click "Add New Database User"
3. Choose "Password" authentication method
4. Enter a username and password (remember these for your .env file)
5. Grant "Read and write to any database" privileges
6. Click "Add User"

### Configure Network Access
1. Go to "Network Access" in the left sidebar
2. Click "Add IP Address"
3. Click "Allow Access from Anywhere" (0.0.0.0/0) for development
4. Click "Confirm"

### Get Connection String
1. Go to "Clusters" and click "Connect" on your cluster
2. Choose "Connect your application"
3. Select "Python" as the driver and version "3.6 or later"
4. Copy the connection string
5. Replace `<username>`, `<password>`, and `<cluster-url>` with your credentials

Example connection string format:
```
mongodb+srv://<username>:<password>@<cluster-url>/hackathon-book?retryWrites=true&w=majority
```

## Configure Environment Variables

Create a `.env` file in the `backend_py` directory:

```env
MONGODB_URI=your_mongodb_connection_string
PORT=5000
HUGGING_FACE_TOKEN=your_huggingface_token_here
```

Replace `your_mongodb_connection_string` with:
- For local: `mongodb://localhost:27017/hackathon-book`
- For Atlas: Your MongoDB Atlas connection string

## Verify MongoDB Connection

### For Local Installation
1. Open Command Prompt
2. Type `mongo` (or `mongosh` for newer versions)
3. You should see the MongoDB shell prompt

### For Atlas
1. In your MongoDB Atlas dashboard, go to your cluster
2. You should see active connections when your application connects

## Troubleshooting

### Common Issues:
1. **Connection refused**: Make sure MongoDB service is running
2. **Authentication failed**: Verify your username/password in the connection string
3. **Network access**: Ensure your IP address is whitelisted in Atlas

### Check MongoDB Status (Windows):
```bash
net start | findstr MongoDB
```

### Restart MongoDB Service (Windows):
```bash
net stop MongoDB
net start MongoDB
```

## Using the Application

Once MongoDB is set up:
1. Make sure your `.env` file has the correct `MONGODB_URI`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python -m src.main`

The application will automatically detect if MongoDB is available and use it. If MongoDB is not available, it will fall back to the mock database implementation.