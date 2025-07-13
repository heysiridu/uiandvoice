pip install pydub numpy
# Also install ffmpeg for MP3 support (see AUDIO_RECORDING.md for details)import os
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import wave
import io
import threading
from pydub import AudioSegment
import numpy as np

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
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "9BWtsMINqrJLrRacOk9x")  # Default voice

# Audio recording configuration
DEFAULT_AUDIO_FORMAT = os.getenv("DEFAULT_AUDIO_FORMAT", "mp3")
MP3_BITRATE = os.getenv("MP3_BITRATE", "128k")
RECORD_CONVERSATIONS = os.getenv("RECORD_CONVERSATIONS", "true").lower() == "true"
RECORDINGS_DIRECTORY = os.getenv("RECORDINGS_DIRECTORY", "responses")

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
    audio_recording: bool = True
    audio_buffer: List[bytes] = []
    recording_lock: Optional[threading.Lock] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self.recording_lock = threading.Lock()
        self.audio_buffer = []


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
        
        # Initialize conversation state for recording
        if call_sid not in active_conversations:
            active_conversations[call_sid] = ConversationState(
                call_sid=call_sid,
                room_name=f"room_{call_sid}",
                start_time=datetime.now()
            )
            logger.info(f"Started conversation recording for call {call_sid}")
        
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
        
        # Finalize audio recording if conversation exists
        await finalize_conversation_recording(call_sid, "mp3")
        
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
            # Finalize audio recording before cleanup
            await finalize_conversation_recording(call_sid, "mp3")
            
            if call_sid in active_conversations:
                del active_conversations[call_sid]
                logger.info(f"Cleaned up conversation state for {call_sid}")
                
    except Exception as e:
        logger.error(f"Error handling call status: {e}")
    
    return PlainTextResponse("OK")


@app.get("/audio/formats")
async def get_supported_audio_formats():
    """Get list of supported audio formats for conversation recording"""
    return {
        "supported_formats": [
            {
                "format": "mp3",
                "description": "MP3 - Compressed, good balance of quality and file size",
                "recommended": True,
                "typical_size": "Small (1-2 MB per minute)"
            },
            {
                "format": "wav", 
                "description": "WAV - Uncompressed, highest quality",
                "recommended": False,
                "typical_size": "Large (10-20 MB per minute)"
            },
            {
                "format": "ogg",
                "description": "OGG - Open source compressed format",
                "recommended": False,
                "typical_size": "Small (1-2 MB per minute)"
            },
            {
                "format": "flac",
                "description": "FLAC - Lossless compression",
                "recommended": False,
                "typical_size": "Medium (5-10 MB per minute)"
            }
        ],
        "default_format": "mp3",
        "note": "File sizes are approximate and depend on audio quality and content"
    }


@app.post("/audio/set-format/{call_sid}")
async def set_audio_format(call_sid: str, format_type: str):
    """Set the audio recording format for a specific call"""
    supported_formats = ["mp3", "wav", "ogg", "flac"]
    
    if format_type.lower() not in supported_formats:
        return {
            "error": f"Unsupported format: {format_type}",
            "supported_formats": supported_formats
        }
    
    if call_sid in active_conversations:
        # Note: This would typically be stored in the conversation state
        # For now, it's just a confirmation endpoint
        return {
            "message": f"Audio format set to {format_type.lower()} for call {call_sid}",
            "format": format_type.lower()
        }
    else:
        return {
            "error": f"Call {call_sid} not found or not active"
        }


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


async def save_conversation_audio(call_sid: str, audio_buffer: List[bytes], 
                                 file_format: str = None, sample_rate: int = 16000):
    """Save the recorded conversation audio to a file"""
    try:
        if not audio_buffer:
            logger.warning(f"No audio data to save for call {call_sid}")
            return
            
        # Use default format if not specified
        if file_format is None:
            file_format = DEFAULT_AUDIO_FORMAT
            
        # Create responses directory if it doesn't exist
        os.makedirs(RECORDINGS_DIRECTORY, exist_ok=True)
        
        # Concatenate all audio chunks
        combined_audio = b''.join(audio_buffer)
        
        if not combined_audio:
            logger.warning(f"Empty audio data for call {call_sid}")
            return
        
        # Convert bytes to audio array (assuming 16-bit PCM)
        audio_array = np.frombuffer(combined_audio, dtype=np.int16)
        
        # Create AudioSegment from raw audio
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit = 2 bytes
            channels=1  # Mono
        )
        
        # Determine file path based on format
        if file_format.lower() == "wav":
            file_path = f"{RECORDINGS_DIRECTORY}/call_{call_sid}_conversation.wav"
            audio_segment.export(file_path, format="wav")
        elif file_format.lower() == "mp3":
            file_path = f"{RECORDINGS_DIRECTORY}/call_{call_sid}_conversation.mp3"
            audio_segment.export(file_path, format="mp3", bitrate=MP3_BITRATE)
        elif file_format.lower() == "ogg":
            file_path = f"{RECORDINGS_DIRECTORY}/call_{call_sid}_conversation.ogg"
            audio_segment.export(file_path, format="ogg")
        elif file_format.lower() == "flac":
            file_path = f"{RECORDINGS_DIRECTORY}/call_{call_sid}_conversation.flac"
            audio_segment.export(file_path, format="flac")
        else:
            # Default to MP3
            file_path = f"{RECORDINGS_DIRECTORY}/call_{call_sid}_conversation.mp3"
            audio_segment.export(file_path, format="mp3", bitrate=MP3_BITRATE)
        
        logger.info(f"Saved conversation audio to {file_path}")
        logger.info(f"Audio duration: {len(audio_segment) / 1000:.2f} seconds")
        
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving conversation audio: {e}")
        return None


async def record_audio_chunk(conversation: ConversationState, audio_data: bytes):
    """Add audio chunk to the conversation recording buffer"""
    try:
        if conversation.audio_recording and conversation.recording_lock:
            with conversation.recording_lock:
                conversation.audio_buffer.append(audio_data)
                
    except Exception as e:
        logger.error(f"Error recording audio chunk: {e}")


async def finalize_conversation_recording(call_sid: str, file_format: str = "mp3"):
    """Finalize and save the conversation recording"""
    try:
        if call_sid in active_conversations:
            conversation = active_conversations[call_sid]
            
            # Save the audio recording
            audio_file_path = await save_conversation_audio(
                call_sid, 
                conversation.audio_buffer, 
                file_format
            )
            
            if audio_file_path:
                # Add audio file info to the summary
                await update_call_summary_with_audio(call_sid, audio_file_path)
            
            # Clear the audio buffer to free memory
            conversation.audio_buffer.clear()
            
    except Exception as e:
        logger.error(f"Error finalizing conversation recording: {e}")


async def update_call_summary_with_audio(call_sid: str, audio_file_path: str):
    """Update the call summary to include audio file information"""
    try:
        summary_filename = f"responses/call_{call_sid}_summary.txt"
        
        if os.path.exists(summary_filename):
            # Read existing summary
            with open(summary_filename, "r") as f:
                content = f.read()
            
            # Add audio file info
            audio_info = f"\n=== AUDIO RECORDING ===\n"
            audio_info += f"Audio File: {os.path.basename(audio_file_path)}\n"
            audio_info += f"Full Path: {audio_file_path}\n"
            audio_info += f"File Size: {os.path.getsize(audio_file_path) / 1024:.2f} KB\n"
            audio_info += f"Created: {datetime.now().isoformat()}\n"
            
            # Insert audio info before the end marker
            updated_content = content.replace(
                "\n=== END OF CALL ===\n",
                audio_info + "\n=== END OF CALL ===\n"
            )
            
            # Write updated summary
            with open(summary_filename, "w") as f:
                f.write(updated_content)
            
            logger.info(f"Updated call summary with audio file info: {summary_filename}")
            
    except Exception as e:
        logger.error(f"Error updating call summary with audio info: {e}")


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
                # Convert audio frame to bytes
                audio_data = frame.data.tobytes()
                
                # Record audio chunk for conversation recording
                await record_audio_chunk(conversation, audio_data)
                
                # Send to Deepgram for transcription
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