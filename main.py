import os
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
from livekit import api, rtc
from deepgram import DeepgramClient as Deepgram
import google.generativeai as genai
import elevenlabs
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Voice Agent API")

# Configuration from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice

# Initialize API clients
deepgram_client = Deepgram(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
genai.configure(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
# Set API key for elevenlabs
if ELEVENLABS_API_KEY:
    elevenlabs.set_api_key(ELEVENLABS_API_KEY)
elevenlabs_client = ELEVENLABS_API_KEY is not None


class ConversationState(BaseModel):
    """Track the state of a conversation"""
    call_sid: str
    room_name: str
    current_question: int = 0
    responses: List[str] = []
    start_time: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True


# Store active conversations
active_conversations: Dict[str, ConversationState] = {}

# Define the questions to ask
QUESTIONS = [
    "What's your name?",
    "What brings you here today?",
    "How can I best assist you?"
]

# Initial greeting
GREETING = "Hello! Welcome to our voice assistant. I'd like to ask you a few questions to better understand how I can help you today."

# Closing message
CLOSING = "Thank you for your time! I've noted all your responses and we'll be in touch soon. Have a great day!"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "Voice Agent is running", "timestamp": datetime.now().isoformat()}


@app.post("/twilio/voice")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio voice calls"""
    try:
        # Parse form data from Twilio
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        from_number = form_data.get("From", "")
        
        logger.info(f"Incoming call from {from_number} with SID: {call_sid}")
        
        # Create a TwiML response with speech recognition
        response = VoiceResponse()
        
        # Add a greeting
        response.say("Hello! Welcome to our voice assistant. I'm here to help you today.")
        
        # Add a pause
        response.pause(length=1)
        
        # Ask for name with speech recognition
        gather = response.gather(
            input='speech',
            timeout=10,
            speech_timeout='auto',
            action='/twilio/name_response',
            method='POST'
        )
        gather.say("What's your name?")
        
        # If no response, redirect to ask again
        response.redirect('/twilio/voice')
        
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        response = VoiceResponse()
        response.say("Sorry, we're experiencing technical difficulties. Please try again later.")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")


@app.post("/twilio/name_response")
async def handle_name_response(request: Request):
    """Handle the name response from the user"""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        speech_result = form_data.get("SpeechResult", "")
        confidence = form_data.get("Confidence", "")
        
        logger.info(f"Name response for call {call_sid}: {speech_result} (confidence: {confidence})")
        
        # Save the name to a file
        if speech_result:
            # Clean up the speech result (remove trailing punctuation)
            cleaned_result = speech_result.strip().rstrip('.!?')
            await save_response_to_file(call_sid, "name", cleaned_result)
        
        # Create response for the second question
        response = VoiceResponse()
        
        # Ask for reason with speech recognition
        gather = response.gather(
            input='speech',
            timeout=10,
            speech_timeout='auto',
            action='/twilio/reason_response',
            method='POST'
        )
        gather.say("Why are you calling today?")
        
        # If no response, ask again
        response.say("I didn't hear your response. Let me ask again.")
        response.redirect('/twilio/name_response')
        
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error handling name response: {e}")
        response = VoiceResponse()
        response.say("I'm sorry, I didn't catch that. Let me ask again.")
        response.redirect('/twilio/voice')
        return Response(content=str(response), media_type="application/xml")


@app.post("/twilio/reason_response")
async def handle_reason_response(request: Request):
    """Handle the reason response from the user"""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        speech_result = form_data.get("SpeechResult", "")
        confidence = form_data.get("Confidence", "")
        
        logger.info(f"Reason response for call {call_sid}: {speech_result} (confidence: {confidence})")
        
        # Save the reason to a file
        if speech_result:
            # Clean up the speech result (remove trailing punctuation)
            cleaned_result = speech_result.strip().rstrip('.!?')
            await save_response_to_file(call_sid, "reason", cleaned_result)
        
        # Create final response
        response = VoiceResponse()
        response.say("Thank you for calling! Have a great day.")
        response.hangup()
        
        # Create a summary of the call
        await create_call_summary(call_sid)
        
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error handling reason response: {e}")
        response = VoiceResponse()
        response.say("I'm sorry, I didn't catch that. Let me ask again.")
        response.redirect('/twilio/reason_response')
        return Response(content=str(response), media_type="application/xml")


@app.post("/twilio/status")
async def handle_call_status(request: Request):
    """Handle Twilio call status callbacks"""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        
        logger.info(f"Call {call_sid} status: {call_status}")
        
        # Clean up conversation state when call ends
        if call_status in ["completed", "failed", "busy", "no-answer"]:
            if call_sid in active_conversations:
                del active_conversations[call_sid]
                logger.info(f"Cleaned up conversation state for {call_sid}")
                
    except Exception as e:
        logger.error(f"Error handling call status: {e}")
    
    return PlainTextResponse("OK")


async def save_response_to_file(call_sid: str, question_type: str, response_text: str):
    """Save user responses to a text file"""
    try:
        # Create responses directory if it doesn't exist
        os.makedirs("responses", exist_ok=True)
        
        # Create filename based on call_sid only (no timestamp)
        filename = f"responses/call_{call_sid}.txt"
        
        # Append the response to file
        with open(filename, "a") as f:
            f.write(f"Call SID: {call_sid}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Question: {question_type}\n")
            f.write(f"Response: {response_text}\n")
            f.write("-" * 50 + "\n")
        
        logger.info(f"Saved {question_type} response to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving response to file: {e}")


async def create_call_summary(call_sid: str):
    """Create a summary file for the completed call"""
    try:
        filename = f"responses/call_{call_sid}.txt"
        
        if os.path.exists(filename):
            # Read the existing file
            with open(filename, "r") as f:
                content = f.read()
            
            # Create a summary file
            summary_filename = f"responses/call_{call_sid}_summary.txt"
            with open(summary_filename, "w") as f:
                f.write("=== CALL SUMMARY ===\n")
                f.write(f"Call SID: {call_sid}\n")
                f.write(f"Call Completed: {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n\n")
                f.write("=== ALL RESPONSES ===\n")
                f.write(content)
                f.write("\n=== END OF CALL ===\n")
            
            logger.info(f"Created call summary: {summary_filename}")
        
    except Exception as e:
        logger.error(f"Error creating call summary: {e}")


async def create_livekit_token(room_name: str, participant_name: str) -> str:
    """Create a LiveKit access token for the participant"""
    try:
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_name(participant_name)
        token.add_grant(api.VideoGrant(
            room_join=True,
            room=room_name
        ))
        return token.to_jwt()
    except Exception as e:
        logger.error(f"Error creating LiveKit token: {e}")
        raise


async def handle_conversation(conversation: ConversationState):
    """Main conversation handler using LiveKit, Deepgram, Gemini, and ElevenLabs"""
    try:
        logger.info(f"Starting conversation handler for {conversation.call_sid}")
        
        # Connect to LiveKit room
        room = rtc.Room()
        
        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"Participant connected: {participant.identity}")
            
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.AUDIO:
                logger.info(f"Audio track subscribed from {participant.identity}")
                # Start processing audio
                asyncio.create_task(process_audio_track(track, conversation))
        
        # Connect to the room
        await room.connect(LIVEKIT_URL, conversation.room_name)
        
        # Wait a moment for connection
        await asyncio.sleep(1)
        
        # Send greeting
        await send_tts_response(conversation.room_name, GREETING)
        
        # Wait a bit before asking first question
        await asyncio.sleep(3)
        
        # Ask first question
        await send_tts_response(conversation.room_name, QUESTIONS[0])
        
    except Exception as e:
        logger.error(f"Error in conversation handler: {e}")


async def process_audio_track(track: rtc.Track, conversation: ConversationState):
    """Process incoming audio track with Deepgram STT"""
    try:
        if not deepgram_client:
            logger.error("Deepgram client not initialized")
            return
            
        # Set up Deepgram live transcription
        options = {
            "punctuate": True,
            "interim_results": False,
            "language": "en-US",
            "model": "general",
            "encoding": "linear16",
            "sample_rate": 16000
        }
        
        # Create Deepgram live connection
        deepgram_live = await deepgram_client.transcription.live(options)
        
        # Handle transcription results
        @deepgram_live.on("transcript")
        async def on_transcript(result):
            transcript = result.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
            if transcript:
                logger.info(f"Transcript: {transcript}")
                await handle_user_response(conversation, transcript)
        
        # Start transcription
        await deepgram_live.start()
        
        # Process audio frames
        async for frame in track:
            if isinstance(frame, rtc.AudioFrame):
                # Convert audio frame to bytes and send to Deepgram
                audio_data = frame.data.tobytes()
                await deepgram_live.send(audio_data)
                
    except Exception as e:
        logger.error(f"Error processing audio track: {e}")


async def handle_user_response(conversation: ConversationState, transcript: str):
    """Handle user's response and generate next action"""
    try:
        # Store the response
        conversation.responses.append(transcript)
        
        # Generate response using Gemini
        gemini_response = await generate_gemini_response(conversation, transcript)
        
        # Send TTS response
        await send_tts_response(conversation.room_name, gemini_response)
        
        # Move to next question or end conversation
        conversation.current_question += 1
        
        if conversation.current_question < len(QUESTIONS):
            # Ask next question after a pause
            await asyncio.sleep(2)
            await send_tts_response(conversation.room_name, QUESTIONS[conversation.current_question])
        else:
            # End conversation
            await asyncio.sleep(2)
            await send_tts_response(conversation.room_name, CLOSING)
            # The call will be ended by Twilio when the user hangs up
            
    except Exception as e:
        logger.error(f"Error handling user response: {e}")
        # Send error message
        await send_tts_response(conversation.room_name, "I'm sorry, I didn't catch that. Could you please repeat?")


async def generate_gemini_response(conversation: ConversationState, user_input: str) -> str:
    """Generate a response using Google Gemini"""
    try:
        if not GEMINI_API_KEY:
            # Fallback response if Gemini is not configured
            return f"Thank you for sharing that with me. I've noted your response."
        
        # Create the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Build conversation context
        context = f"""You are a helpful voice assistant having a conversation with a user. 
        Current question number: {conversation.current_question + 1} of {len(QUESTIONS)}
        Current question: {QUESTIONS[conversation.current_question]}
        User's response: {user_input}
        
        Previous responses: {conversation.responses[:-1] if len(conversation.responses) > 1 else 'None'}
        
        Generate a brief, conversational response (1-2 sentences) acknowledging their answer.
        Be warm, professional, and encouraging."""
        
        # Generate response
        response = await model.generate_content_async(context)
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        # Fallback response
        return "Thank you for that information. Let me continue with the next question."


async def send_tts_response(room_name: str, text: str):
    """Convert text to speech using ElevenLabs and send to LiveKit room"""
    try:
        if not ELEVENLABS_API_KEY:
            logger.error("ElevenLabs API key not configured")
            return
            
        logger.info(f"Generating TTS for: {text}")
        
        # Generate audio using ElevenLabs
        audio_stream = await elevenlabs.generate(
            text=text,
            voice=ELEVENLABS_VOICE_ID,
            model="eleven_monolingual_v1"
        )
        
        # TODO: Send audio to LiveKit room
        # This requires setting up LiveKit participant and audio track publishing
        # For MVP, this is a placeholder
        logger.info(f"TTS generated for room {room_name}")
        
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Voice Agent starting up...")
    logger.info(f"Twilio configured: {bool(TWILIO_ACCOUNT_SID)}")
    logger.info(f"LiveKit configured: {bool(LIVEKIT_API_KEY)}")
    logger.info(f"Deepgram configured: {bool(DEEPGRAM_API_KEY)}")
    logger.info(f"Gemini configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"ElevenLabs configured: {bool(ELEVENLABS_API_KEY)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Voice Agent shutting down...")
    # Clean up any active conversations
    active_conversations.clear()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)