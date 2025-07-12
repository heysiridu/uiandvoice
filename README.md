# Voice Agent with Twilio, LiveKit, Deepgram, Gemini, and ElevenLabs

A conversational voice agent that handles incoming phone calls, asks users questions, and provides intelligent responses using state-of-the-art AI services.

## Architecture Overview

```
Phone → Twilio → LiveKit → Deepgram (STT) → Gemini (LLM) → ElevenLabs (TTS) → User
```

## Features

- **Phone Integration**: Accept incoming calls via Twilio
- **Real-time Audio Streaming**: LiveKit handles WebRTC audio streams
- **Speech Recognition**: Deepgram converts speech to text in real-time
- **AI Responses**: Google Gemini generates contextual responses
- **Natural Voice**: ElevenLabs provides high-quality text-to-speech
- **Conversation Management**: Tracks conversation state and user responses

## Prerequisites

- Python 3.9+
- Accounts and API keys for:
  - [Twilio](https://www.twilio.com/)
  - [LiveKit](https://livekit.io/)
  - [Deepgram](https://deepgram.com/)
  - [Google Gemini](https://ai.google.dev/)
  - [ElevenLabs](https://elevenlabs.io/)
- ngrok (for local development)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd voice-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the environment example file:
```bash
cp .env.example .env
```

5. Configure your `.env` file with your API keys and settings.

## Configuration

### Twilio Setup

1. Log into your Twilio Console
2. Buy a phone number (if you don't have one)
3. Configure the phone number's voice webhook:
   - **When a call comes in**: `https://your-domain.com/twilio/voice`
   - **Call status changes**: `https://your-domain.com/twilio/status`
   - Method: POST for both

### LiveKit Setup

1. Set up a LiveKit server (local or cloud)
2. Get your API key and secret
3. Update the `LIVEKIT_URL` in your `.env` file

### Local Development with ngrok

For local testing, use ngrok to expose your local server:

```bash
# In one terminal, start the app:
python main.py

# In another terminal, start ngrok:
ngrok http 8000
```

Update your Twilio webhooks with the ngrok URL.

## Running the Application

1. Start the FastAPI server:
```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. The API will be available at `http://localhost:8000`
3. API documentation is available at `http://localhost:8000/docs`

## Usage

1. Call your Twilio phone number
2. The agent will:
   - Greet you
   - Ask you 3 questions:
     - What's your name?
     - What brings you here today?
     - How can I best assist you?
   - Provide conversational responses to each answer
   - Thank you and end the call

## API Endpoints

- `GET /` - Health check endpoint
- `POST /twilio/voice` - Handles incoming Twilio calls
- `POST /twilio/status` - Handles call status updates

## Project Structure

```
voice-agent/
├── main.py              # Main FastAPI application
├── .env.example         # Example environment configuration
├── .env                 # Your actual configuration (not in git)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Customization

### Changing Questions

Edit the `QUESTIONS` list in `main.py`:

```python
QUESTIONS = [
    "Your first question?",
    "Your second question?",
    "Your third question?"
]
```

### Modifying Greetings

Update the `GREETING` and `CLOSING` variables in `main.py`.

### Adjusting Voice Settings

Modify the ElevenLabs voice settings in the `send_tts_response` function.

## Error Handling

The application includes basic error handling:
- Falls back to predefined responses if Gemini fails
- Asks users to repeat if transcription fails
- Gracefully handles service failures

## Development Tips

1. **Testing**: Use the Twilio CLI or dashboard to test calls
2. **Debugging**: Check logs for detailed information about each call
3. **Monitoring**: Use the health check endpoint to monitor uptime
4. **Scaling**: Consider using a process manager like Gunicorn for production

## Future Enhancements

- Integration with CrewAI for agent-to-agent communication
- Advanced conversation interruption handling
- Call recording and analytics
- Multi-language support
- Custom voice training
- Webhook security with Twilio signatures

## Troubleshooting

### Common Issues

1. **"Service not available" error**
   - Check all API keys are correctly set in `.env`
   - Verify all services are accessible

2. **Audio quality issues**
   - Check LiveKit server configuration
   - Verify network bandwidth

3. **Transcription errors**
   - Ensure Deepgram language model matches user language
   - Check audio sample rate settings

## License

[Your License Here]

## Support

For issues or questions, please [create an issue](your-repo-issues-url).