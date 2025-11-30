# README.md - SpeechToText

## Overview

SpeechToText is a Python application that records audio from a microphone and transcribes it to text using Azure Speech-to-Text services. It also includes Text-to-Speech functionality using Azure OpenAI TTS. The transcribed text is copied to the clipboard and pasted into the active application. It runs in the system tray for easy access.

## Installation

1. **Install Dependencies**:
   Open a terminal and run:
   ```bash
   pip install pynput sounddevice requests pyperclip pystray keyboard soundfile simpleaudio python-dotenv
   ```

2. **Set Azure API Details**:
   Create a [`.env`](.env ) file in the same directory as the script with your Azure Speech-to-Text and TTS credentials. Define **all four** variables (no defaults are assumed):
   ```plaintext
   # Speech-to-Text (STT)
   AZURE_STT_ENDPOINT=your_speech_to_text_endpoint
   AZURE_STT_API_KEY=your_speech_to_text_api_key

   # Text-to-Speech (TTS)
   AZURE_TTS_ENDPOINT=your_text_to_speech_endpoint
   AZURE_TTS_API_KEY=your_text_to_speech_api_key
   # Optional: default TTS speed (0.25–4.0, defaults to 1.0 if unset/invalid)
   AZURE_TTS_SPEED_DEFAULT=1.30
   ```

3. **Provide Sound Files**:
   Add short WAV files for feedback sounds ([`start.wav`](start.wav ), [`stop.wav`](stop.wav ), [`cancel.wav`](cancel.wav )) in the same directory as the script.

4. **Run the App**:
   In the terminal, execute:
   ```bash
   python speechtotext.py
   ```

## Usage

### Speech-to-Text
- The application runs in the system tray.
- Press **Ctrl + Windows key** to start recording.
- Release or press **Ctrl + Windows key** again to stop recording and transcribe the audio.
- Press **Esc** during recording to cancel.
- Right-click the tray icon to select a microphone or exit the application.

### Text-to-Speech
- Copy or highlight text that you want to hear spoken.
- Press **Alt + Windows key** to speak the text from your clipboard.
- Right-click the tray icon and select **TTS Voice** to choose from available voices (alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse).

## Building an Executable

To create a standalone executable, use PyInstaller with the following command:
```bash
pyinstaller --noconfirm --onefile --windowed --icon=speaking.ico --add-data "start.wav;." --add-data "stop.wav;." --add-data "cancel.wav;." --add-data "speaking.ico;." speechtotext.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.