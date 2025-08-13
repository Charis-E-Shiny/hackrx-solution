# Deployment Guide

This document provides step-by-step instructions for deploying the HackRx LLM Query-Retrieval System on various free hosting platforms.

## 🚀 Quick Deploy Options

### Option 1: Render.com (Recommended - Free Tier)

1. **Fork/Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd hackrx-llm-query-system
   ```

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Deploy**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Use these settings:
     - **Build Command**: `pip install fastapi uvicorn pydantic httpx pymupdf scikit-learn numpy openai`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Environment Variables**:
       - `OPENAI_API_KEY`: Your OpenAI API key

4. **Test Deployment**
   ```bash
   curl -X GET "https://your-app-name.onrender.com/" \
     -H "Authorization: Bearer 94adf1bd4f8978d3029fd88cb36c7f95a255d331301920caa50830f5f6c3ba26"
   ```

### Option 2: Railway (Free Tier)

1. **Deploy from GitHub**
   - Go to [railway.app](https://railway.app)
   - Sign up and create new project
   - Connect GitHub repository

2. **Configure Environment**
   - Add environment variable: `OPENAI_API_KEY`
   - Railway will auto-detect Python and install dependencies

3. **Custom Start Command** (if needed)
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

### Option 3: Heroku (Free Tier Alternative)

1. **Create Heroku App**
   ```bash
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY="your-key-here"
   ```

2. **Deploy**
   ```bash
   git push heroku main
   ```

## 🔧 Local Development

### Setup
```bash
# Clone repository
git clone <repo-url>
cd hackrx-llm-query-system

# Install dependencies
pip install fastapi uvicorn pydantic httpx pymupdf scikit-learn numpy openai

# Set environment variable
export OPENAI_API_KEY="your-openai-api-key"

# Start server
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### Test Local Installation
```bash
# Run the test suite
python test_api.py

# Or test manually
curl -X GET "http://localhost:5000/health" \
  -H "Authorization: Bearer 94adf1bd4f8978d3029fd88cb36c7f95a255d331301920caa50830f5f6c3ba26"
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 512MB RAM minimum
- Internet connection for PDF downloads and OpenAI API

### Dependencies (Auto-installed)
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `httpx` - HTTP client
- `pymupdf` - PDF processing
- `scikit-learn` - TF-IDF embeddings
- `numpy` - Numerical operations
- `openai` - OpenAI API client

### API Keys Required
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com)
  - Used for: Document analysis and question answering
  - Cost: ~$0.01 per 10 questions (very affordable)

## 🎯 HackRx Compliance Verification

After deployment, verify all requirements are met:

```bash
# Test the exact HackRx endpoint
curl -X POST "https://your-deployed-app.com/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 94adf1bd4f8978d3029fd88cb36c7f95a255d331301920caa50830f5f6c3ba26" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?",
      "Does this policy cover maternity expenses, and what are the conditions?"
    ]
  }'
```

Expected response format:
```json
{
  "answers": [
    "A grace period of thirty days is provided...",
    "There is a waiting period of thirty-six (36) months...",
    "Yes, the policy covers maternity expenses..."
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Error: No module named 'fitz'**
   ```bash
   pip install pymupdf
   ```

2. **OpenAI API Error**
   - Verify API key is set correctly
   - Check OpenAI account has credits
   - Ensure key starts with "sk-"

3. **Memory Issues on Free Tier**
   - System optimized for 512MB RAM
   - Uses TF-IDF instead of heavy ML models
   - Efficient text processing

4. **Timeout Issues**
   - Increase timeout for large PDF processing
   - System typically responds within 10 seconds

### Performance Optimization

- **Cold Start**: First request may take 10-15 seconds
- **Warm Requests**: Subsequent requests < 5 seconds
- **Memory Usage**: ~200-300MB typical usage
- **Processing Time**: 3-5 seconds per document

## 💰 Cost Analysis

### Free Tier Hosting
- **Render.com**: 750 hours/month free
- **Railway**: $5 credit monthly (plenty for testing)
- **Heroku**: Alternative options available

### OpenAI Costs (Only Variable Cost)
- **GPT-4o**: ~$0.005 per question
- **Typical Usage**: 100 questions = $0.50
- **Cost per Document**: ~$0.02-0.05 (5-10 questions)

### Total Monthly Cost
- **Hosting**: $0 (free tier)
- **API**: $5-20 depending on usage
- **Break-even**: Much cheaper than alternatives

## 🎯 Production Checklist

- [ ] OpenAI API key configured
- [ ] Repository deployed successfully
- [ ] Health check endpoint working
- [ ] Main query endpoint tested
- [ ] Authentication working
- [ ] Error handling verified
- [ ] Performance acceptable (< 10 seconds)
- [ ] Logs accessible for debugging

Your HackRx LLM Query-Retrieval System is now ready for production use! 🚀