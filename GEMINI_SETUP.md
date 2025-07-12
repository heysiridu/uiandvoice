# Gemini API Setup Guide

## Step-by-Step Instructions

### Step 1: Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" at the top of the page
3. Click "New Project"
4. Enter a project name (e.g., "Voice Agent Project")
5. Click "Create"

### Step 2: Enable the Gemini API
1. In your Google Cloud project, go to the [API Library](https://console.cloud.google.com/apis/library)
2. Search for "Gemini API" or "Generative AI API"
3. Click on "Gemini API" from the results
4. Click "Enable"

### Step 3: Create API Credentials
1. Go to [Credentials](https://console.cloud.google.com/apis/credentials) in your Google Cloud Console
2. Click "Create Credentials" → "API Key"
3. Your new API key will be displayed - **copy it immediately** (you won't be able to see it again)
4. Click "Restrict Key" to secure it (recommended)

### Step 4: Configure API Key Restrictions (Optional but Recommended)
1. In the API key settings, under "Application restrictions":
   - Choose "HTTP referrers" or "IP addresses" for web apps
   - Choose "None" for server applications
2. Under "API restrictions":
   - Select "Restrict key"
   - Choose "Gemini API" from the dropdown
3. Click "Save"

### Step 5: Set Up Environment Variables
Create a `.env` file in your project root with the following content:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here

# LiveKit Configuration
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here

# Deepgram Configuration
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

Replace `your_gemini_api_key_here` with the actual API key you copied in Step 3.

### Step 6: Install Dependencies
Your project already has the required dependencies in `requirements.txt`. Run:

```bash
pip install -r requirements.txt
```

### Step 7: Test the Configuration
Your code already includes Gemini integration. The key parts are:

1. **Import and Configuration** (lines 11 and 32 in main.py):
```python
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
```

2. **Usage in generate_gemini_response function** (around line 274):
```python
async def generate_gemini_response(conversation: ConversationState, user_input: str) -> str:
    # Your Gemini integration code here
```

### Step 8: Verify Setup
1. Make sure your `.env` file is in the project root
2. Run your application: `python main.py`
3. Check that the Gemini client initializes without errors

## Troubleshooting

### Common Issues:
1. **"API key not found"**: Make sure your `.env` file is in the project root and contains the correct API key
2. **"API not enabled"**: Go back to Step 2 and ensure the Gemini API is enabled
3. **"Quota exceeded"**: Check your Google Cloud billing and quotas
4. **"Invalid API key"**: Double-check that you copied the entire API key correctly

### Testing the API Key:
You can test your API key with this simple script:

```python
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, how are you?")
    print(response.text)
else:
    print("GEMINI_API_KEY not found in environment variables")
```

## Security Best Practices

1. **Never commit your `.env` file** to version control
2. **Use environment variables** in production
3. **Restrict your API key** to specific IPs or applications
4. **Monitor usage** in Google Cloud Console
5. **Set up billing alerts** to avoid unexpected charges

## Next Steps

Once your Gemini API is configured:
1. Test your voice agent application
2. Customize the conversation flow in the `QUESTIONS` list
3. Adjust the Gemini prompt in `generate_gemini_response` function
4. Deploy your application with proper environment variable management 