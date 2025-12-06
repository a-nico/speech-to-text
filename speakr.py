import threading
import sounddevice as sd
import requests
import pyperclip
import pystray
from pynput import keyboard
import io
import time
import keyboard as kb
from PIL import Image
import numpy as np
import simpleaudio as sa
import os
from dotenv import load_dotenv
import sys
from typing import Optional, List, Dict, Any, Callable
import ctypes
import traceback

__version__ = "1.3.0"

# Record hotkey: Alt + Windows key
CTRL_KEYS = {keyboard.Key.ctrl_l}
WIN_KEYS = {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r}
ALT_KEYS = {keyboard.Key.alt_l}

RECORD_SECONDS = 60  # Max recording duration in seconds
SAMPLE_RATE = 16000  # Default audio sample rate, will be adjusted if unsupported


class Config:
    """Manages application configuration, paths, and credentials."""
    
    # Available TTS voices
    TTS_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
    
    def __init__(self):
        self.base_path = self._get_base_path()
        self._load_dotenv()
        
        # Speech-to-Text (STT) configuration
        self.azure_stt_endpoint = os.getenv("AZURE_STT_ENDPOINT")
        self.azure_stt_api_key = os.getenv("AZURE_STT_API_KEY")

        # Text-to-Speech (TTS) configuration
        self.azure_tts_endpoint = os.getenv("AZURE_TTS_ENDPOINT")
        self.azure_tts_api_key = os.getenv("AZURE_TTS_API_KEY")
        
        self.sound_files = {
            "start": "start.wav",
            "stop": "stop.wav",
            "cancel": "cancel.wav",
            "send": "send.wav"
        }
        
        self.icon_path = self.get_path("speaking.ico")

        self._print_config_status()

    def _get_base_path(self) -> str:
        """Determine the base path for the application, handling PyInstaller."""
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, the PyInstaller bootloader
            # extends the sys module by a flag frozen=True and sets the app 
            # path into variable _MEIPASS'.
            return sys._MEIPASS
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def _load_dotenv(self) -> None:
        """Load environment variables from a .env file located next to the executable or script."""
        if getattr(sys, 'frozen', False):
            # Path for the executable
            env_dir = os.path.dirname(sys.executable)
        else:
            # Path for the script
            env_dir = os.path.dirname(os.path.abspath(__file__))
        
        env_path = os.path.join(env_dir, '.env')
        print(f'Looking for .env file at: {env_path}')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print('.env file loaded successfully')
        else:
            print(f'.env file not found at: {env_path}')

    def get_path(self, filename: str) -> str:
        """Get the absolute path for a given asset file."""
        return os.path.join(self.base_path, filename)

    def _print_config_status(self) -> None:
        """Print the status of the Azure configuration."""
        if not self.azure_stt_endpoint or not self.azure_stt_api_key:
            print("Azure STT configuration incomplete. Check AZURE_STT_ENDPOINT / AZURE_STT_API_KEY.")
        else:
            print("Azure STT configuration loaded successfully")

        if not self.azure_tts_endpoint or not self.azure_tts_api_key:
            print("Azure TTS configuration incomplete. Check AZURE_TTS_ENDPOINT / AZURE_TTS_API_KEY.")
        else:
            print("Azure TTS configuration loaded successfully")

config = Config()


class TextToSpeechService:
    """Handles text-to-speech conversion using Azure OpenAI TTS API."""
    
    def __init__(self, config: Config):
        self.config = config
        # Default voice (can be overridden via env)
        env_voice = os.getenv("AZURE_TTS_VOICE_DEFAULT", "alloy").lower()
        if env_voice in Config.TTS_VOICES:
            self.current_voice: str = env_voice
        else:
            print(f"Invalid AZURE_TTS_VOICE_DEFAULT '{env_voice}'; falling back to 'alloy'")
            self.current_voice: str = "alloy"
        # Default playback/generation speed (can be overridden via env)
        try:
            env_speed = os.getenv("AZURE_TTS_SPEED_DEFAULT")
            if env_speed is not None:
                value = float(env_speed)
                # Clamp to OpenAI's allowed range 0.25–4.0
                self.speed = max(0.25, min(4.0, value))
            else:
                self.speed = 1.0
        except ValueError:
            print("Invalid AZURE_TTS_SPEED_DEFAULT value; falling back to 1.0")
            self.speed = 1.0
        self._is_playing: bool = False
        self._play_lock = threading.Lock()
        self._current_play_obj: Optional[sa.PlayObject] = None
    
    @property
    def available_voices(self) -> List[str]:
        """Return list of available TTS voices."""
        return Config.TTS_VOICES
    
    def set_voice(self, voice: str) -> None:
        """Set the current TTS voice."""
        if voice in self.available_voices:
            self.current_voice = voice
            print(f"TTS voice set to: {voice}")
        else:
            print(f"Invalid voice '{voice}'. Available voices: {self.available_voices}")
    
    def get_clipboard_text(self, copy_selection: bool = False) -> Optional[str]:
        """Get the current text from clipboard.
        
        Args:
            copy_selection: If True, simulate Ctrl+C first to copy any selected text.
        """
        try:
            if copy_selection:
                # Save current clipboard content
                original_clipboard = pyperclip.paste()
                
                # Simulate Ctrl+C to copy selected text
                kb.press_and_release('ctrl+c')
                
                # Small delay to allow clipboard to update
                time.sleep(0.15)
                
                text = pyperclip.paste()
                
                # If clipboard didn't change (nothing was selected), restore original
                if text == original_clipboard:
                    print("No text was selected, using existing clipboard content.")
                
            else:
                text = pyperclip.paste()
            
            if text and text.strip():
                return text.strip()
            return None
        except Exception as e:
            print(f"Error reading clipboard: {e}")
            return None
    
    def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Send text to Azure OpenAI TTS API and return audio bytes."""
        if not self.config.azure_tts_api_key or not self.config.azure_tts_endpoint:
            print("Azure TTS not configured.")
            show_error_notification("TTS Error", "Azure TTS not configured. Please set AZURE_TTS_ENDPOINT and AZURE_TTS_API_KEY.")
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.azure_tts_api_key}"
            }
            
            payload = {
                "model": "tts-hd",
                "input": text,
                "voice": self.current_voice,
                "speed": self.speed,
            }
            
            print(f"Sending TTS request with voice '{self.current_voice}' at speed {self.speed}...")
            response = requests.post(
                self.config.azure_tts_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            print(f"TTS response received: {len(response.content)} bytes")
            return response.content
            
        except requests.exceptions.Timeout:
            print("TTS request timed out after 30 seconds.")
            play_click("cancel")
            show_error_notification("TTS Error", "Request timed out after 30 seconds. Please try again.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"TTS request failed: {e}")
            play_click("cancel")
            show_error_notification("TTS Error", f"Failed to synthesize speech: {e}")
            return None
    
    def stop_playback(self) -> None:
        """Stop the currently playing TTS audio."""
        with self._play_lock:
            if self._current_play_obj is not None and self._is_playing:
                try:
                    self._current_play_obj.stop()
                    print("TTS playback stopped.")
                except Exception as e:
                    print(f"Error stopping TTS playback: {e}")
                finally:
                    self._current_play_obj = None
                    self._is_playing = False
            else:
                print("No TTS audio is currently playing.")
    
    def play_audio(self, audio_data: bytes) -> None:
        """Play audio data using simpleaudio."""
        with self._play_lock:
            if self._is_playing:
                print("Audio is already playing, skipping...")
                return
            self._is_playing = True
        
        def _play():
            try:
                # The API returns MP3 by default, we need to convert it
                # Try to play as WAV first, if that fails try MP3
                try:
                    # Try playing as WAV
                    audio_io = io.BytesIO(audio_data)
                    import wave
                    with wave.open(audio_io, 'rb') as wf:
                        wave_obj = sa.WaveObject(
                            wf.readframes(wf.getnframes()),
                            wf.getnchannels(),
                            wf.getsampwidth(),
                            wf.getframerate()
                        )
                    play_obj = wave_obj.play()
                    with self._play_lock:
                        self._current_play_obj = play_obj
                    play_obj.wait_done()
                except Exception:
                    # If WAV fails, the audio is likely MP3
                    # Use soundfile to convert
                    import soundfile as sf
                    audio_io = io.BytesIO(audio_data)
                    data, samplerate = sf.read(audio_io)
                    
                    # Convert to int16 for simpleaudio
                    if data.dtype != np.int16:
                        # Normalize and convert to int16
                        if data.max() <= 1.0 and data.min() >= -1.0:
                            data = (data * 32767).astype(np.int16)
                        else:
                            data = data.astype(np.int16)
                    
                    # Handle mono/stereo
                    if len(data.shape) == 1:
                        num_channels = 1
                    else:
                        num_channels = data.shape[1]
                    
                    wave_obj = sa.WaveObject(
                        data.tobytes(),
                        num_channels,
                        2,  # bytes per sample for int16
                        samplerate
                    )
                    play_obj = wave_obj.play()
                    with self._play_lock:
                        self._current_play_obj = play_obj
                    play_obj.wait_done()
                    
            except Exception as e:
                print(f"Error playing TTS audio: {e}")
                print(traceback.format_exc())
                show_error_notification("TTS Playback Error", f"Failed to play audio: {e}")
            finally:
                with self._play_lock:
                    self._is_playing = False
                    self._current_play_obj = None
        
        threading.Thread(target=_play, daemon=True).start()
    
    def speak_clipboard(self, copy_selection: bool = True) -> None:
        """Get text from clipboard and speak it.
        
        Args:
            copy_selection: If True, simulate Ctrl+C first to copy any selected text.
        """
        text = self.get_clipboard_text(copy_selection=copy_selection)
        if not text:
            print("No text in clipboard to speak.")
            show_error_notification("TTS Error", "No text found in clipboard. Please select or copy some text first.")
            return
        
        print(f"Speaking clipboard text: {text[:50]}..." if len(text) > 50 else f"Speaking: {text}")
        
        audio_data = self.synthesize_speech(text)
        if audio_data:
            self.play_audio(audio_data)


# Global TTS service instance (initialized after config)
tts_service: Optional[TextToSpeechService] = None

# Cache for preloaded WaveObjects
SOUND_WAVES = {}

def load_sounds() -> None:
    """Preload sound files into memory for feedback sounds."""
    for kind, filename in config.sound_files.items():
        path = config.get_path(filename)
        if not os.path.exists(path):
            print(f"Sound file not found at startup: {path}")
            continue
        try:
            SOUND_WAVES[kind] = sa.WaveObject.from_wave_file(path)
            print(f"Preloaded sound '{kind}': {path}")
        except Exception as e:
            print(f"Error preloading sound '{kind}' from {path}: {e}")

def play_click(kind: str = "start") -> None:
    wave_obj = SOUND_WAVES.get(kind) or SOUND_WAVES.get("start")
    if not wave_obj:
        print(f"No preloaded sound available for '{kind}'.")
        return
    def _play():
        try:
            wave_obj.play()
        except Exception as e:
            print(f"Error playing preloaded sound '{kind}': {e}")
    threading.Thread(target=_play, daemon=True).start()


def show_error_notification(title: str, message: str) -> None:
    """Show an error message to the user via Windows message box."""
    def _show():
        try:
            # Use Windows MessageBox API
            MB_OK = 0x0
            MB_ICONERROR = 0x10
            MB_SYSTEMMODAL = 0x1000
            ctypes.windll.user32.MessageBoxW(
                0, 
                message, 
                title, 
                MB_OK | MB_ICONERROR | MB_SYSTEMMODAL
            )
        except Exception as e:
            print(f"Failed to show error dialog: {e}")
    
    # Show message box in a separate thread to not block the main thread
    threading.Thread(target=_show, daemon=True).start()


def safe_execute(func: Callable, error_context: str, *args, **kwargs) -> Any:
    """Execute a function safely, catching and logging any exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = f"{error_context}: {e}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        show_error_notification("Speech-to-Text Error", error_msg)
        return None


class Recorder:
    """Handles audio recording from the selected microphone device."""
    def __init__(self) -> None:
        self.recording: bool = False
        self.audio: List[np.ndarray] = []
        self.lock: threading.Lock = threading.Lock()
        self.device_id: Optional[int] = None
        self.wasapi_devices: List[Dict[str, Any]] = []
        self.sample_rate: int = SAMPLE_RATE
        self.error_occurred: bool = False  # Track if an error occurred during recording
        self.on_error_callback: Optional[Callable[[], None]] = None  # Callback for error recovery
        self._initialize_device()

    def _initialize_device(self) -> None:
        """Find and select an appropriate audio input device."""
        self._find_wasapi_devices()
        if self.wasapi_devices:
            self._select_preferred_device()
        else:
            print("No devices found using Windows WASAPI API. Falling back to default device.")
            self._fallback_to_default_device()
        
        if self.device_id is not None:
            self._set_supported_sample_rate()

    def _find_wasapi_devices(self) -> None:
        """Find all available WASAPI input devices."""
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("No input devices (microphones) found.")
            return

        print("Available input devices (microphones) using recommended Windows WASAPI API:")
        for i, dev in enumerate(input_devices):
            try:
                host_api = sd.query_hostapis(dev['hostapi'])
                if host_api['name'] == 'Windows WASAPI':
                    self.wasapi_devices.append(dev)
                    print(f"  {i}: {dev['name']}")
            except sd.PortAudioError:
                continue

    def _select_preferred_device(self) -> None:
        """Select the best available WASAPI device."""
        # Priority order: Samson, USB Sound Card, then first available
        samson_device = next((d for d in self.wasapi_devices if 'samson' in d['name'].lower()), None)
        if samson_device:
            self.device_id = samson_device['index']
            print(f"Using Samson input device: {samson_device['name']} (ID: {self.device_id})")
            return
        
        usb_sound_card = next((d for d in self.wasapi_devices if 'usb sound card' in d['name'].lower()), None)
        if usb_sound_card:
            self.device_id = usb_sound_card['index']
            print(f"Using USB Sound Card input device: {usb_sound_card['name']} (ID: {self.device_id})")
            return
        
        self.device_id = self.wasapi_devices[0]['index']
        print(f"Using first WASAPI device: {self.wasapi_devices[0]['name']} (ID: {self.device_id})")

    def _fallback_to_default_device(self) -> None:
        """Fall back to the system's default input device."""
        try:
            device_index = sd.default.device[0]
            if device_index != -1:
                self.device_id = device_index
                device_info = sd.query_devices(device_index)
                print(f"Using default input device: {device_info['name']} (ID: {self.device_id})")
            else:
                print("No default input device found.")
        except Exception as e:
            print(f"Could not determine default device: {e}")

    def set_device(self, device_id: int) -> None:
        """Update the current recording device and adjust sample rate."""
        device_info = next((d for d in self.wasapi_devices if d['index'] == device_id), None)
        if device_info:
            self.device_id = device_id
            print(f"Selected device: {device_info['name']} (ID: {device_id})")
            self._set_supported_sample_rate()
        else:
            print(f"Device ID {device_id} not found in WASAPI devices.")

    def _set_supported_sample_rate(self) -> None:
        """Set a sample rate supported by the current device."""
        if self.device_id is None:
            print("No device selected, cannot set sample rate.")
            return
            
        device_info = sd.query_devices(self.device_id)

        # Common sample rates to test
        common_rates = [44100, 48000, 16000, 8000]
        for rate in common_rates:
            try:
                sd.check_input_settings(device=self.device_id, samplerate=rate, channels=1, dtype='int16')
                self.sample_rate = rate
                print(f"Set sample rate to {rate} Hz for device ID {self.device_id}")
                return
            except sd.PortAudioError:
                continue
        
        # If none of the common rates work, fall back to the device's default sample rate if available
        default_rate = device_info.get('default_samplerate', SAMPLE_RATE)
        try:
            sd.check_input_settings(device=self.device_id, samplerate=int(default_rate), channels=1, dtype='int16')
            self.sample_rate = int(default_rate)
            print(f"Set sample rate to default {default_rate} Hz for device ID {self.device_id}")
        except (sd.PortAudioError, ValueError):
            self.sample_rate = SAMPLE_RATE
            print(f"No supported sample rate found. Falling back to default {SAMPLE_RATE} Hz, may not work.")

    def start(self) -> bool:
        """Start recording audio in a separate thread. Returns True if started successfully."""
        if self.device_id is None:
            print("Cannot start recording: No input device is selected.")
            show_error_notification("Recording Error", "No microphone selected. Please select a microphone from the tray menu.")
            return False
        
        self.recording = True
        self.audio = []
        self.error_occurred = False
        threading.Thread(target=self._record, daemon=True).start()
        return True

    def stop(self) -> None:
        """Stop the current recording."""
        self.recording = False

    def cancel(self) -> None:
        """Stop recording and discard any captured audio."""
        self.stop()
        with self.lock:
            self.audio = []
        self.error_occurred = False

    def refresh_devices(self) -> None:
        """Refresh the list of available audio devices and reselect if needed."""
        print("Refreshing audio devices...")

        # Re-initialize the sounddevice library to detect hardware changes
        sd._terminate()
        sd._initialize()

        previous_device_id = self.device_id
        self.wasapi_devices = []
        self.device_id = None
        
        self._find_wasapi_devices()
        if self.wasapi_devices:
            # Try to keep the same device if it still exists
            if previous_device_id is not None:
                device_still_exists = any(d['index'] == previous_device_id for d in self.wasapi_devices)
                if device_still_exists:
                    self.device_id = previous_device_id
                    print(f"Restored previous device ID: {previous_device_id}")
                else:
                    print(f"Previous device (ID: {previous_device_id}) no longer available.")
                    self._select_preferred_device()
            else:
                self._select_preferred_device()
        else:
            print("No devices found after refresh. Falling back to default device.")
            self._fallback_to_default_device()
        
        if self.device_id is not None:
            self._set_supported_sample_rate()

    def _record(self) -> None:
        """Internal method to handle the recording loop."""
        # Callback to collect audio chunks
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Recording status: {status}")
            if self.recording:
                with self.lock:
                    self.audio.append(np.array(indata, dtype='int16'))
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', callback=callback, device=self.device_id):
                start_time = time.time()
                while self.recording and (time.time() - start_time < RECORD_SECONDS):
                    time.sleep(0.1)
        except sd.PortAudioError as e:
            error_msg = f"Audio device error: {e}\nThe microphone may have been disconnected."
            print(f"Error recording audio: {e}")
            print("This may indicate that the microphone was disconnected.")
            self.recording = False
            self.error_occurred = True
            show_error_notification("Microphone Error", error_msg)
            # Trigger error callback to reset state
            if self.on_error_callback:
                self.on_error_callback()
        except Exception as e:
            error_msg = f"Recording failed: {e}"
            print(f"Unexpected error during recording: {e}")
            print(traceback.format_exc())
            self.recording = False
            self.error_occurred = True
            show_error_notification("Recording Error", error_msg)
            # Trigger error callback to reset state
            if self.on_error_callback:
                self.on_error_callback()

    def get_wav_bytes(self) -> Optional[io.BytesIO]:
        """Return the recorded audio as WAV bytes in a BytesIO buffer."""
        import soundfile as sf
        try:
            with self.lock:
                if not self.audio:
                    return None
                audio_np = np.concatenate(self.audio, axis=0)
            buf = io.BytesIO()
            sf.write(buf, audio_np, self.sample_rate, format='WAV', subtype='PCM_16')
            buf.seek(0)
            return buf
        except Exception as e:
            print(f"Error creating WAV bytes: {e}")
            show_error_notification("Audio Processing Error", f"Failed to process recorded audio: {e}")
            return None


# Global flag for echo mode
ECHO_MODE = False

def transcribe_audio(wav_bytes: io.BytesIO) -> str:
    """Send audio data to Azure Speech-to-Text service and return transcribed text.
       When echo mode is enabled, play back the audio instead of sending it."""
    global ECHO_MODE
    try:
        if ECHO_MODE:
            print("ECHO MODE: Playing back recorded audio instead of sending to API.")
            wav_bytes.seek(0)
            try:
                import wave
                import simpleaudio as sa
                with wave.open(wav_bytes, 'rb') as wf:
                    wave_obj = sa.WaveObject(
                        wf.readframes(wf.getnframes()),
                        wf.getnchannels(),
                        wf.getsampwidth(),
                        wf.getframerate()
                    )
                play_obj = wave_obj.play()
                play_obj.wait_done()
            except Exception as e:
                print(f"Error during playback: {e}")
            return "[Echo mode: Audio played back locally]"
        else:
            if not config.azure_stt_api_key or not config.azure_stt_endpoint:
                print("Azure STT not configured.")
                show_error_notification(
                    "STT Error",
                    "Azure STT not configured. Please set AZURE_STT_ENDPOINT and AZURE_STT_API_KEY.",
                )
                return ""

            headers = {"api-key": config.azure_stt_api_key}
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            response = requests.post(config.azure_stt_endpoint, headers=headers, files=files)
            response.raise_for_status()
            return response.json().get("text", "")
    except Exception as e:
        print(f"Transcription/playback failed: {e}")
        return ""

def copy_and_paste(text: str) -> bool:
    """Copy the given text to clipboard and simulate Ctrl+V to paste it. Returns True on success."""
    try:
        pyperclip.copy(text)
        kb.press_and_release('ctrl+v')
        return True
    except Exception as e:
        print(f"Error copying/pasting text: {e}")
        show_error_notification("Paste Error", f"Failed to paste text: {e}\n\nThe text has been copied to your clipboard.")
        return False

def create_icon() -> Image.Image:
    """Load tray icon from speaking.ico; fallback to default."""
    size = (64, 64)
    try:
        img = Image.open(config.icon_path)
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Icon load error: {e} — using default.")
        return Image.new("RGB", size, (240, 255, 0))

def create_tray_menu(
    recorder: "Recorder",
    icon: pystray.Icon,
    on_refresh_mics: Callable[[pystray.Icon], None],
    on_exit: Callable[[pystray.Icon], None]
) -> pystray.Menu:
    """Create the tray icon menu with microphone selection, TTS voice/speed selection, echo mode, and other options."""
    def toggle_echo_mode(icon: pystray.Icon, item: pystray.MenuItem) -> None:
        global ECHO_MODE
        ECHO_MODE = not ECHO_MODE
        print(f"Echo mode {'enabled' if ECHO_MODE else 'disabled'}.")

    def echo_mode_checked(item: pystray.MenuItem) -> bool:
        return ECHO_MODE

    return pystray.Menu(
        pystray.MenuItem("Microphones", pystray.Menu(lambda: create_mic_menu(recorder, icon))),
        pystray.MenuItem("TTS Voice", pystray.Menu(lambda: create_voice_menu(icon))),
        pystray.MenuItem("TTS Speed", pystray.Menu(lambda: create_speed_menu(icon))),
        pystray.MenuItem("Refresh mics", on_refresh_mics),
        pystray.MenuItem("Echo mode", toggle_echo_mode, checked=echo_mode_checked),
        pystray.MenuItem("Exit", on_exit)
    )

def create_mic_menu(recorder, icon) -> List[pystray.MenuItem]:
    """Create a submenu for selecting microphones."""
    mic_items: List[pystray.MenuItem] = []
    for dev in recorder.wasapi_devices:
        # Capture the device index for the handler and checked state
        device_index: int = dev['index']

        def make_handler(d_id: int) -> Callable[[], None]:
            def handler() -> None:
                recorder.set_device(d_id)
                icon.update_menu()  # Update the menu to show the new checkmark
            return handler

        def make_checker(d_id: int) -> Callable[[pystray.MenuItem], bool]:
            return lambda item: recorder.device_id == d_id

        item = pystray.MenuItem(
            dev['name'],
            make_handler(device_index),
            checked=make_checker(device_index),
            radio=True
        )
        mic_items.append(item)
    return mic_items


def create_voice_menu(icon) -> List[pystray.MenuItem]:
    """Create a submenu for selecting TTS voices."""
    global tts_service
    voice_items: List[pystray.MenuItem] = []
    
    if tts_service is None:
        return [pystray.MenuItem("TTS not initialized", lambda: None, enabled=False)]
    
    for voice in tts_service.available_voices:
        def make_handler(v: str) -> Callable[[], None]:
            def handler() -> None:
                tts_service.set_voice(v)
                icon.update_menu()
            return handler
        
        def make_checker(v: str) -> Callable[[pystray.MenuItem], bool]:
            return lambda item, voice=v: tts_service.current_voice == voice
        
        item = pystray.MenuItem(
            voice.capitalize(),
            make_handler(voice),
            checked=make_checker(voice),
            radio=True
        )
        voice_items.append(item)
    
    return voice_items


def create_speed_menu(icon) -> List[pystray.MenuItem]:
    """Create a submenu for selecting TTS playback/generation speed."""
    global tts_service

    speed_items: List[pystray.MenuItem] = []

    if tts_service is None:
        return [pystray.MenuItem("TTS not initialized", lambda: None, enabled=False)]

    # Preset speeds (as requested): 1.0, 1.15, 1.30, 1.45, 1.6
    preset_speeds = [1.0, 1.15, 1.30, 1.45, 1.6]

    for speed in preset_speeds:
        label = f"{speed:.2f}x".rstrip("0").rstrip(".") + "x"

        def make_handler(s: float) -> Callable[[], None]:
            def handler() -> None:
                tts_service.speed = s
                print(f"TTS speed set to {s}")
                icon.update_menu()
            return handler

        def make_checker(s: float) -> Callable[[pystray.MenuItem], bool]:
            return lambda item, value=s: abs(tts_service.speed - value) < 1e-6

        item = pystray.MenuItem(
            label,
            make_handler(speed),
            checked=make_checker(speed),
            radio=True,
        )
        speed_items.append(item)

    return speed_items

def main() -> None:
    global tts_service
    
    load_sounds()  # Preload sounds once at startup

    recorder = Recorder()
    tts_service = TextToSpeechService(config)

    # --- Hotkey Management ---
    
    # Use a lock to safely modify state from different threads
    state_lock = threading.Lock()
    pressed_keys = set()
    # Track the order of key presses to enforce correct sequence
    # Windows key must be pressed SECOND (after Ctrl or Alt)
    key_press_order: List[keyboard.Key] = []
    hotkey_combo = {keyboard.Key.alt, keyboard.Key.cmd}  # Alt + Win for speech-to-text
    tts_hotkey_combo = {keyboard.Key.ctrl, keyboard.Key.cmd}  # Ctrl + Win for text-to-speech
    combo_activated = False
    tts_combo_activated = False
    record_start_time = 0.0
    MIN_RECORD_DURATION = 1.0

    def reset_state() -> None:
        """Reset all state variables to recover from errors."""
        nonlocal combo_activated, tts_combo_activated
        with state_lock:
            combo_activated = False
            tts_combo_activated = False
            pressed_keys.clear()
            key_press_order.clear()
        if recorder.recording:
            recorder.cancel()
        print("State reset complete - ready for next recording.")

    # Register the error callback with the recorder
    recorder.on_error_callback = reset_state

    def on_press(key: keyboard.Key) -> None:
        nonlocal combo_activated, tts_combo_activated, record_start_time
        
        try:
            # Allow ESC to cancel an active recording without transcribing, or stop TTS playback
            if key == keyboard.Key.esc:
                if recorder.recording or combo_activated:
                    safe_execute(play_click, "Playing cancel sound", "cancel")
                    print("Recording canceled.")
                    recorder.cancel()
                    with state_lock:
                        combo_activated = False
                        pressed_keys.clear()
                        key_press_order.clear()
                    return
                elif tts_service and tts_service._is_playing:
                    tts_service.stop_playback()
                    return

            # Normalize left/right modifier keys
            normalized_key = key
            if key in CTRL_KEYS:
                normalized_key = keyboard.Key.ctrl
            elif key in WIN_KEYS:
                normalized_key = keyboard.Key.cmd
            elif key in ALT_KEYS:
                normalized_key = keyboard.Key.alt

            with state_lock:
                if normalized_key in (keyboard.Key.ctrl, keyboard.Key.cmd, keyboard.Key.alt):
                    # Only add to order if not already pressed (avoid key repeat)
                    if normalized_key not in pressed_keys:
                        pressed_keys.add(normalized_key)
                        key_press_order.append(normalized_key)
                
                # Helper function to check if Windows key was pressed second
                def is_windows_key_second(required_first_key: keyboard.Key) -> bool:
                    """Check if the key order is: required_first_key, then Windows key."""
                    if len(key_press_order) < 2:
                        return False
                    # Find positions of the keys in the press order
                    try:
                        first_key_pos = key_press_order.index(required_first_key)
                        win_key_pos = key_press_order.index(keyboard.Key.cmd)
                        # Windows key must come after the first key
                        return win_key_pos > first_key_pos
                    except ValueError:
                        return False
                
                # Check for TTS hotkey (Ctrl + Win) - must check before recording hotkey
                # Windows key must be pressed AFTER Ctrl
                if (tts_hotkey_combo.issubset(pressed_keys) and 
                    not tts_combo_activated and 
                    not combo_activated and
                    is_windows_key_second(keyboard.Key.ctrl)):
                    # Don't activate TTS if Alt is also pressed (to avoid conflict with Alt+Win)
                    if keyboard.Key.alt not in pressed_keys:
                        tts_combo_activated = True
                        # No start sound for TTS - only play send sound on release
                        print("TTS hotkey activated - will speak clipboard text on release...")
                        return
                
                # Check for recording hotkey (Alt + Win)
                # Windows key must be pressed AFTER Alt
                if (hotkey_combo.issubset(pressed_keys) and 
                    not recorder.recording and 
                    not combo_activated and 
                    not tts_combo_activated and
                    is_windows_key_second(keyboard.Key.alt)):
                    combo_activated = True
                    safe_execute(play_click, "Playing start sound", "start")
                    print("Recording started... (release to transcribe)")
                    record_start_time = time.time()
                    if not recorder.start():
                        # Failed to start recording, reset state
                        combo_activated = False
                        pressed_keys.clear()
                        key_press_order.clear()
        except Exception as e:
            print(f"Error in on_press handler: {e}")
            print(traceback.format_exc())
            reset_state()

    def on_release(key: keyboard.Key) -> None:
        nonlocal combo_activated, tts_combo_activated
        
        try:
            # Normalize left/right modifier keys
            normalized_key = key
            if key in CTRL_KEYS:
                normalized_key = keyboard.Key.ctrl
            elif key in WIN_KEYS:
                normalized_key = keyboard.Key.cmd
            elif key in ALT_KEYS:
                normalized_key = keyboard.Key.alt
            
            with state_lock:
                # Check if we were in a TTS session (Ctrl + Win combo was activated)
                was_tts_combo_active = tts_combo_activated
                # Check if we were in a recording session (combo was activated)
                was_combo_active = combo_activated
                is_combo_key = normalized_key in hotkey_combo
                is_tts_combo_key = normalized_key in tts_hotkey_combo
            
            # Handle TTS hotkey release (Ctrl + Win)
            if is_tts_combo_key and was_tts_combo_active:
                safe_execute(play_click, "Playing send sound", "send")
                print("TTS hotkey released - speaking clipboard text...")
                
                # Trigger TTS in a separate thread to not block the listener
                def do_tts():
                    try:
                        tts_service.speak_clipboard()
                    except Exception as e:
                        print(f"TTS error: {e}")
                        print(traceback.format_exc())
                
                threading.Thread(target=do_tts, daemon=True).start()
                
                with state_lock:
                    pressed_keys.clear()
                    key_press_order.clear()
                    tts_combo_activated = False
                return
            
            # Handle recording hotkey release (Alt + Win)
            if is_combo_key and was_combo_active:
                recorder.stop()
                duration: float = time.time() - record_start_time

                # Check if an error occurred during recording
                if recorder.error_occurred:
                    print("Recording had an error, skipping transcription.")
                    with state_lock:
                        pressed_keys.clear()
                        combo_activated = False
                    return

                # Check for minimum recording duration
                if duration < MIN_RECORD_DURATION:
                    safe_execute(play_click, "Playing cancel sound", "cancel")
                    print(f"Recording too short ({duration:.2f}s). Canceled.")
                else:
                    safe_execute(play_click, "Playing stop sound", "stop")
                    print("Recording stopped.")
                    
                    wav_bytes = recorder.get_wav_bytes()
                    if wav_bytes:
                        print("Transcribing...")
                        try:
                            text = transcribe_audio(wav_bytes)
                            if text:
                                print("Transcribed:", text)
                                copy_and_paste(text)
                            else:
                                print("Transcription returned no text.")
                        except Exception as e:
                            print(f"Error during transcription or pasting: {e}")
                            show_error_notification("Transcription Error", f"Failed to transcribe audio: {e}")

                # Reset state for the next recording
                with state_lock:
                    pressed_keys.clear()
                    key_press_order.clear()
                    combo_activated = False
            
            elif normalized_key in (keyboard.Key.ctrl, keyboard.Key.cmd, keyboard.Key.alt):
                with state_lock:
                    pressed_keys.discard(normalized_key)
                    # Remove from order list as well
                    if normalized_key in key_press_order:
                        key_press_order.remove(normalized_key)
                
        except Exception as e:
            print(f"Error in on_release handler: {e}")
            print(traceback.format_exc())
            reset_state()

    # Start global hotkey listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Run tray icon with a context menu for exit and microphone selection
    def on_exit(icon: pystray.Icon) -> None:
        listener.stop()  # Stop the listener thread
        icon.stop()

    def on_refresh_mics(icon: pystray.Icon) -> None:
        """Refresh the microphone list and rebuild the menu."""
        recorder.refresh_devices()
        # Rebuild the entire menu so the microphone list updates correctly
        icon.menu = create_tray_menu(recorder, icon, on_refresh_mics, on_exit)
        icon.update_menu()
        print("Microphone menu refreshed.")

    # The icon needs to be created before the menu so we can pass it to the menu creation function
    icon = pystray.Icon("Speakr", create_icon(), "Speakr")

    icon.menu = create_tray_menu(recorder, icon, on_refresh_mics, on_exit)
    icon.run()

if __name__ == "__main__":
    main()


