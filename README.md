# Dhamm AI Chatbot

## Overview
Dhamm AI Chatbot is a conversational AI application designed to assist users with civil engineering concepts using Bloom's Taxonomy. It leverages various AI models and frameworks to provide accurate and helpful responses tailored to the user's cognitive level and sentiment.

## Recent Changes and Improvements
- **Fixed Vector Store Initialization**: Improved the initialization process to ensure proper creation and loading of the vector database
- **Enhanced Error Handling**: Added comprehensive error handling throughout the application
- **Updated Memory Implementation**: Fixed deprecated ConversationBufferMemory implementation
- **Updated Conversation Chain Method**: Replaced deprecated `__call__` with `invoke` method
- **Production Deployment**: Added Gunicorn configuration for robust production deployment
- **Port Configuration**: Changed default port to 8000 to avoid conflicts with macOS AirPlay
- **Added Health Check Endpoint**: Created root endpoint for monitoring application status
- **Improved Text Chunking**: Enhanced text chunking strategy for better retrieval performance
- **Fixed API Path Issues**: Ensured all API endpoints are correctly prefixed and accessible
- **Environment Configuration**: Improved environment variable handling with better validation

## Features
- **Bloom's Taxonomy Integration**: Automatically detects the cognitive level of user questions
- **Sentiment Analysis**: Analyzes user sentiment to provide more empathetic responses
- **Contextual Memory**: Maintains conversation history for more coherent interactions
- **Dynamic Response Generation**: Tailors responses based on cognitive level and sentiment
- **Source Document Retrieval**: Retrieves relevant chunks from lecture transcripts

## Detailed Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Dhamm_AI_Chatbot-main
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory with the following API keys:

```
# Get a GROQ API key from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Get a Google API key with Generative AI API access from https://console.cloud.google.com
GOOGLE_API_KEY=your_google_api_key_here
```

> **Important:** Both API keys must be valid and have access to their respective services:
> - GROQ API key needs access to the `llama3-70b-8192` model
> - Google API key needs access to the Generative AI API and `models/text-embedding-004` model

### 5. Initialize Vector Store
Before running the application, you need to initialize the vector store with your transcript data:

```bash
python3 vectordb.py
```

This will:
- Create a vector store from the `cleaned_transcript.txt` file
- Split the text into appropriate chunks
- Generate embeddings using Google's API
- Store the vectors in the `vectordb` directory

You should see output confirming the number of vectors created.

## Running the Application

### Development Mode
For development, you can run the Flask application directly:

```bash
# Navigate to project directory
cd /path/to/Dhamm_AI_Chatbot-main

# Activate virtual environment
source venv/bin/activate

# Run the Flask application (port 8000)
python3 app5.py
```

#### macOS-specific Note
On macOS, port 5000 is commonly used by AirPlay Receiver. If you encounter a "Port 5000 is in use" error, you have two options:

1. Edit `app5.py` to use a different port (recommended):
   ```python
   # Change this line at the bottom of app5.py
   app.run(host='0.0.0.0', port=8000, debug=False)
   ```

2. Disable AirPlay Receiver:
   - Go to System Preferences → General → AirDrop & Handoff
   - Disable the 'AirPlay Receiver' service

### Running in Background
To run the application in the background:

```bash
cd /path/to/Dhamm_AI_Chatbot-main
source venv/bin/activate
nohup python3 app5.py > app.log 2>&1 &
```

You can check the logs with:
```bash
tail -f app.log
```

### Production Mode
For production use, use Gunicorn:

```bash
# Basic configuration
gunicorn app5:app --bind 0.0.0.0:8000

# Recommended configuration for better performance
gunicorn app5:app --bind 0.0.0.0:8000 --workers 4 --worker-class gevent --timeout 120 --log-level info
```

To run the server in the background (recommended for production):

```bash
# Kill any existing instances first
lsof -i :8000 | awk 'NR>1 {print $2}' | xargs kill -9 2>/dev/null || true

# Start Gunicorn with optimal settings
cd /path/to/Dhamm_AI_Chatbot-main && source venv/bin/activate && gunicorn app5:app --bind 0.0.0.0:8000 --workers 4 --worker-class gevent --timeout 120 --log-level info > gunicorn.log 2>&1 &
```

To verify the server is running correctly:

```bash
# Check running Gunicorn processes
ps aux | grep gunicorn | grep -v grep

# Test the health endpoint
curl http://localhost:8000/
```

The application will be accessible at [http://localhost:8000/](http://localhost:8000/).

## API Endpoints

### 1. Health Check

- **URL:** `/`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "ok",
    "message": "Dhamm AI Chatbot API is running"
  }
  ```

### 2. Chat Endpoint

- **URL:** `/api/chat`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "question": "What are the basic principles of civil engineering?",
    "show_chunks": false
  }
  ```
- **Response:**
  ```json
  {
    "answer": "The basic principles of civil engineering include...",
    "cognitive_level": "understand",
    "cognitive_description": "Explain ideas or concepts",
    "sentiment": "neutral",
    "chunks": []
  }
  ```

### 3. Clear Conversation History

- **URL:** `/api/clear`
- **Method:** `POST`
- **Response:**
  ```json
  {
    "message": "Conversation history cleared"
  }
  ```

### 4. Toggle Source Chunks Visibility

- **URL:** `/api/toggle-chunks`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "show_chunks": true
  }
  ```
- **Response:**
  ```json
  {
    "message": "Show chunks set to true",
    "show_chunks": true
  }
  ```

## Configuration Options

The following configuration options can be modified in `app5.py`:

- `EMBEDDING_MODEL`: The Google AI embedding model to use (default: "models/text-embedding-004")
- `VECTORDB_DIR`: Directory to store vector embeddings (default: "/vectordb")
- `COLLECTION_NAME`: Name of the Chroma collection (default: "chroma")
- `K`: Number of relevant chunks to retrieve (default: 2)
- `TRANSCRIPT_FILE`: Path to the transcript file (default: "cleaned_transcript.txt")

## Testing the API

### Using curl

After starting the server, you can test the API endpoints using curl:

```bash
# Test health check endpoint
curl http://localhost:8000/

# Test chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is civil engineering?"}'

# Enable showing source chunks
curl -X POST http://localhost:8000/api/toggle-chunks \
  -H "Content-Type: application/json" \
  -d '{"show_chunks": true}'

# Clear conversation history
curl -X POST http://localhost:8000/api/clear
```

### Using the Test Interface

A simple test interface is available at:

```
http://localhost:8000/test
```

This provides a basic UI to interact with the chatbot API.

## Troubleshooting Guide

### Vector Store Issues

**Problem**: Vector store initialization fails

**Solution**:
1. Ensure your Google API key is correct and has access to embedding models
2. Check if `cleaned_transcript.txt` exists and has content
3. Delete the `vectordb` directory and run `python3 vectordb.py` again
4. Verify that the `vectordb` directory is created with content

### API Key Issues

**Problem**: "API key not valid" errors

**Solution**:
1. Check that your API keys are correctly formatted in the `.env` file (no quotes, spaces, or extra characters)
2. Verify that your API keys have the necessary permissions
3. For Google API key, ensure it has access to the Generative AI API
4. For GROQ API key, ensure it has access to the LLaMA model

### Server Startup Issues

**Problem**: Server fails to start or port is already in use

**Solution**:
1. Check which process is using the port: `lsof -i :8000`
2. Kill any existing server processes: `pkill -f gunicorn` or `pkill -f python3`
3. Modify the port in app5.py (recommended for macOS users):
   ```python
   # Change line 433 from
   app.run(host='0.0.0.0', port=5000, debug=False)
   # to
   app.run(host='0.0.0.0', port=8000, debug=False)
   ```
4. Or run with an environment variable: `FLASK_RUN_PORT=8000 python3 app5.py`
5. For macOS users: The built-in AirPlay server uses port 5000 by default. Either change the port as shown above or disable AirPlay Receiver in System Preferences.

### Memory/Performance Issues

**Problem**: Application is slow or crashes due to memory issues

**Solution**:
1. Reduce the chunk size in `vectordb.py`
2. Reduce the number of workers in Gunicorn command
3. Process larger transcripts in smaller batches

### Python Not Found

**Problem**: `python: command not found` error

**Solution**:
1. On macOS or some Linux distributions, use `python3` instead of `python`:
   ```bash
   python3 app5.py
   ```

### Module Not Found Errors

**Problem**: `ModuleNotFoundError: No module named 'xyz'`

**Solution**:
1. Ensure your virtual environment is activated: `source venv/bin/activate`
2. Install the missing package: `pip install xyz`
3. If using a new dependency, add it to requirements.txt: `pip freeze > requirements.txt`

### API Response Issues

**Problem**: API returns error or doesn't respond

**Solution**:
1. Check if the server is running (should see Flask output in terminal or logs)
2. Verify the correct URL is being used (`http://localhost:8000/api/chat`)
3. Ensure you're sending valid JSON with the correct `Content-Type` header
4. Check for any error messages in the server logs: `tail -f app.log`

### HTTP 500 Internal Server Errors

**Problem**: API returns HTTP 500 errors when processing certain requests

**Solution**:
1. Update deprecated LangChain method calls:
   ```python
   # Replace this (deprecated)
   response = conversation({"question": question})
   
   # With this (current method)
   response = conversation.invoke({"question": question})
   ```
2. Add better error handling around API calls:
   ```python
   try:
       response = conversation.invoke({"question": question})
       # Process response
   except Exception as e:
       app.logger.error(f"Error in conversation chain: {type(e).__name__}: {str(e)}")
       # Create a fallback response
   ```
3. Check the logs for specific errors: `tail -f gunicorn.log`
4. Restart the server after making code changes

## React Frontend and Azure Deployment

### Connecting React Frontend to Flask Backend

#### 1. Setup React Project

If you don't have an existing React app:

```bash
# Create a new React app
npx create-react-app dhamm-chatbot-frontend
cd dhamm-chatbot-frontend
```

#### 2. Install Required Dependencies

```bash
npm install axios react-markdown
```

#### 3. Configure API Connection

Create an API service file at `src/services/api.js`:

```javascript
import axios from 'axios';

// Development API URL
const DEV_API_URL = 'http://localhost:8000';

// Production API URL - replace with your Azure deployed backend URL
const PROD_API_URL = 'https://your-azure-app-name.azurewebsites.net';

// Use production URL if in production environment, otherwise use development URL
const BASE_URL = process.env.NODE_ENV === 'production' ? PROD_API_URL : DEV_API_URL;

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatService = {
  sendMessage: async (question) => {
    try {
      const response = await api.post('/api/chat', { question });
      return response.data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },
  
  clearHistory: async () => {
    try {
      const response = await api.post('/api/clear');
      return response.data;
    } catch (error) {
      console.error('Error clearing chat history:', error);
      throw error;
    }
  },
  
  toggleChunks: async (showChunks) => {
    try {
      const response = await api.post('/api/toggle-chunks', { show_chunks: showChunks });
      return response.data;
    } catch (error) {
      console.error('Error toggling chunks visibility:', error);
      throw error;
    }
  }
};

export default api;
```

#### 4. Update CORS Settings in Flask Backend

Update the CORS configuration in `app5.py` to allow requests from your React frontend:

```python
# Import CORS
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, static_url_path='')

# Configure CORS for production
if os.environ.get('FLASK_ENV') == 'production':
    # In production, allow only requests from your Azure hosted frontend
    CORS(app, resources={r"/api/*": {"origins": "https://your-frontend-url.azurewebsites.net"}})
else:
    # In development, allow all origins
    CORS(app)
```

### Deploying to Azure

#### 1. Prepare Flask Backend for Azure App Service

Create a new file named `web.config` in your Flask app directory:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="httpPlatformHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified" />
    </handlers>
    <httpPlatform processPath="%home%\site\wwwroot\venv\Scripts\python.exe" arguments="%home%\site\wwwroot\app.py" stdoutLogEnabled="true" stdoutLogFile="%home%\LogFiles\stdout" startupTimeLimit="60">
      <environmentVariables>
        <environmentVariable name="PYTHONPATH" value="%home%\site\wwwroot" />
      </environmentVariables>
    </httpPlatform>
  </system.webServer>
</configuration>
```

Create an `app.py` file for production entry point:

```python
import os
from app5 import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
```

#### 2. Create a Deployment Script

Create a file named `deploy.sh`:

```bash
#!/bin/bash

# Initialize virtual environment on Azure
python -m venv antenv
source antenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize vector store
python vectordb.py

# Ensure permissions
chmod 755 app.py
```

#### 3. Deploy Flask Backend to Azure App Service

```bash
# Install Azure CLI if you haven't already
# brew install azure-cli (macOS)

# Login to Azure
az login

# Create a resource group (skip if you already have one)
az group create --name dhamm-chatbot-rg --location eastus

# Create an App Service plan (skip if you already have one)
az appservice plan create --name dhamm-chatbot-plan --resource-group dhamm-chatbot-rg --sku B1 --is-linux

# Create a web app
az webapp create --resource-group dhamm-chatbot-rg --plan dhamm-chatbot-plan --name dhamm-chatbot-backend --runtime "PYTHON|3.9"

# Configure environment variables
az webapp config appsettings set --resource-group dhamm-chatbot-rg --name dhamm-chatbot-backend --settings FLASK_ENV=production GROQ_API_KEY=your_groq_api_key GOOGLE_API_KEY=your_google_api_key

# Deploy your application code from local git (or use Azure DevOps for CI/CD)
az webapp deployment source config-local-git --resource-group dhamm-chatbot-rg --name dhamm-chatbot-backend

# Get the deployment URL from the output of the previous command
# Add a git remote and push your code
git remote add azure <deployment-url>
git push azure main
```

#### 4. Deploy React Frontend to Azure Static Web Apps

1. Build your React application:

```bash
npm run build
```

2. Deploy to Azure Static Web Apps:

```bash
# Install Azure Static Web Apps CLI if you haven't already
npm install -g @azure/static-web-apps-cli

# Login to Azure
az login

# Create a Static Web App
az staticwebapp create --name dhamm-chatbot-frontend --resource-group dhamm-chatbot-rg --location eastus --source . --branch main --app-location "/" --output-location "build"

# Follow the prompts to authenticate with GitHub/Azure DevOps for deployment
```

### Azure Configuration and Environment Settings

#### 1. Configure Environment Variables in Azure

Navigate to your App Service in the Azure Portal:

1. Go to Configuration > Application settings
2. Add the following key-value pairs:
   - `FLASK_ENV`: `production`
   - `GROQ_API_KEY`: `your_groq_api_key`
   - `GOOGLE_API_KEY`: `your_google_api_key`

#### 2. Enable CORS in Azure

For your backend App Service:

1. Go to CORS settings
2. Add your frontend URL to the allowed origins

#### 3. Configure Custom Domain (Optional)

1. Purchase a domain from a domain registrar
2. In Azure Portal, go to your App Service > Custom domains
3. Follow the steps to add and verify your domain

#### 4. Set Up Azure Application Insights (Optional)

For monitoring:

```bash
# Install Application Insights extension
az extension add --name application-insights

# Create an Application Insights resource
az monitor app-insights component create --app dhamm-chatbot-insights --location eastus --resource-group dhamm-chatbot-rg

# Get the instrumentation key and add it to your app settings
az monitor app-insights component show --app dhamm-chatbot-insights --resource-group dhamm-chatbot-rg --query instrumentationKey

# Add the key to your web app settings
az webapp config appsettings set --resource-group dhamm-chatbot-rg --name dhamm-chatbot-backend --settings APPINSIGHTS_INSTRUMENTATIONKEY=<your-key>
```

### Azure-Specific Considerations

1. **Vector Database Storage**: 
   - Azure App Service has ephemeral storage. For persistent vector database storage, consider:
     - Azure Files mounted to your App Service
     - Azure Blob Storage with the Azure Storage Python SDK
     - Azure Cosmos DB or Azure Database for PostgreSQL for production-scale apps

2. **Scaling and Performance**:
   - Start with at least a B1 or P1v2 App Service plan for adequate memory
   - Scale up if memory pressure occurs during vector operations
   - Consider using Azure Container Apps for better control over resources

3. **Security Best Practices**:
   - Use Azure Key Vault for storing API keys instead of environment variables
   - Implement authentication using Azure AD or Auth0
   - Set up a private endpoint for secure communication between your resources
   - Enable HTTPS only and TLS 1.2+ requirements

4. **Cost Optimization**:
   - Use consumption plans for low-traffic applications
   - Consider serverless options like Azure Functions for the API
   - Implement auto-scaling rules based on traffic patterns

## Code Maintenance

### Updating LangChain Version

LangChain is actively developed and APIs may change. If you encounter deprecation warnings or errors:

1. Update the package: `pip install --upgrade langchain langchain-core langchain-community`
2. Review the migration guides: https://python.langchain.com/docs/versions/
3. Common changes to watch for:
   - `conversation({"question": text})` → `conversation.invoke({"question": text})`
   - `chain.run()` → `chain.invoke()`
   - Memory implementations may change

### Performance Optimization

To optimize the application for better performance:

1. Use the production Gunicorn server with multiple workers
2. Tune the chunk size and retrieval parameters (K) based on your content
3. Monitor memory usage during vector operations
4. Use batch processing for large transcript files during initialization
5. Set appropriate timeouts for LLM API calls (especially for complex questions)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
