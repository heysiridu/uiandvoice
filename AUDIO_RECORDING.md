# Audio Recording Feature

This voice assistant now supports recording entire conversations and saving them as audio files.

## Supported Audio Formats

### Recommended Format: MP3
- **File Extension**: `.mp3`
- **Quality**: Good balance of quality and file size
- **File Size**: ~1-2 MB per minute of conversation
- **Bitrate**: 128kbps (configurable)
- **Best for**: Most use cases, easy sharing and playback

### Other Supported Formats

1. **WAV** (`.wav`)
   - Uncompressed, highest quality
   - Large file sizes (~10-20 MB per minute)
   - Best for: Archival purposes, when file size isn't a concern

2. **OGG** (`.ogg`)
   - Open source compressed format
   - Similar file sizes to MP3
   - Best for: Open source environments

3. **FLAC** (`.flac`)
   - Lossless compression
   - Medium file sizes (~5-10 MB per minute)
   - Best for: When you need lossless quality with some compression

## Configuration

### Environment Variables
Add these to your environment or `.env` file:

```bash
# Audio recording settings
DEFAULT_AUDIO_FORMAT=mp3
MP3_BITRATE=128k
RECORD_CONVERSATIONS=true
RECORDINGS_DIRECTORY=responses
```

### Audio Configuration File
The `audio_config.env` file contains additional settings for audio recording.

## File Organization

Conversation recordings are saved in the `responses/` directory with the following naming convention:

```
responses/
├── call_[CALL_SID]_conversation.mp3  # Audio recording
├── call_[CALL_SID].txt               # Transcript
└── call_[CALL_SID]_summary.txt       # Summary with audio info
```

## Features

1. **Automatic Recording**: All conversations are automatically recorded when `RECORD_CONVERSATIONS=true`
2. **Multiple Formats**: Support for MP3, WAV, OGG, and FLAC formats
3. **Quality Control**: Configurable bitrates and sample rates
4. **Integration**: Audio file information is included in call summaries
5. **API Endpoints**: 
   - `GET /audio/formats` - List supported formats
   - `POST /audio/set-format/{call_sid}` - Set format for specific call

## Memory Management

The system efficiently manages memory by:
- Streaming audio chunks to a buffer during the call
- Processing and saving the complete recording when the call ends
- Clearing audio buffers after saving to free memory

## Requirements

Make sure to install the required packages:

```bash
pip install pydub numpy
```

Note: For MP3 support, you may need to install `ffmpeg`:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg
```

## Usage

The audio recording feature works automatically. When a call is completed, you'll find:

1. A transcript file with the conversation text
2. An audio file with the complete conversation
3. A summary file with metadata about both

Example summary output:
```
=== CALL SUMMARY ===
Call SID: CAxxxxxxxxxxxxx
Call Completed: 2025-07-13T10:30:00

=== ALL RESPONSES ===
[Transcript content]

=== AUDIO RECORDING ===
Audio File: call_CAxxxxxxxxxxxxx_conversation.mp3
Full Path: /path/to/responses/call_CAxxxxxxxxxxxxx_conversation.mp3
File Size: 1.2 MB
Created: 2025-07-13T10:30:00
```
