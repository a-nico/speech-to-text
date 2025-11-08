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

__version__ = "1.1.0"

# Record hotkey: Ctrl + Windows key
CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
WIN_KEYS = {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r}

RECORD_SECONDS = 60  # Max recording duration in seconds
SAMPLE_RATE = 16000  # Default audio sample rate, will be adjusted if unsupported


class Config:
    """Manages application configuration, paths, and credentials."""
    def __init__(self):
        self.base_path = self._get_base_path()
        self._load_dotenv()
        
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        
        self.sound_files = {
            "start": "start.wav",
            "stop": "stop.wav",
            "cancel": "cancel.wav"
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
        if not self.azure_endpoint or not self.azure_api_key:
            print('Azure Speech configuration incomplete. Check environment variables.')
        else:
            print('Azure Speech configuration loaded successfully')

config = Config()

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


class Recorder:
    """Handles audio recording from the selected microphone device."""
    def __init__(self) -> None:
        self.recording: bool = False
        self.audio: List[np.ndarray] = []
        self.lock: threading.Lock = threading.Lock()
        self.device_id: Optional[int] = None
        self.wasapi_devices: List[Dict[str, Any]] = []
        self.sample_rate: int = SAMPLE_RATE
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
        samson_device = next((d for d in self.wasapi_devices if 'Samson' in d['name']), None)
        if samson_device:
            self.device_id = samson_device['index']
            print(f"Using Samson input device: {samson_device['name']} (ID: {self.device_id})")
        else:
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

    def start(self) -> None:
        """Start recording audio in a separate thread."""
        if self.device_id is None:
            print("Cannot start recording: No input device is selected.")
            return
        self.recording = True
        self.audio = []
        threading.Thread(target=self._record).start()  # Start recording in a new thread

    def stop(self) -> None:
        """Stop the current recording."""
        self.recording = False

    def cancel(self) -> None:
        """Stop recording and discard any captured audio."""
        self.stop()
        with self.lock:
            self.audio = []

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
        def callback(indata, frames, time, status):
            if self.recording:
                with self.lock:
                    # Ensure indata is a numpy array of int16
                    self.audio.append(np.array(indata, dtype='int16'))
        # Open microphone stream and record for RECORD_SECONDS or until stopped
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', callback=callback, device=self.device_id):
                start_time = time.time()
                while self.recording and (time.time() - start_time < RECORD_SECONDS):
                    time.sleep(0.1)
        except sd.PortAudioError as e:
            print(f"Error recording audio: {e}")
            print("This may indicate that the microphone was disconnected.")
            self.recording = False
        except Exception as e:
            print(f"Unexpected error during recording: {e}")
            self.recording = False

    def get_wav_bytes(self) -> Optional[io.BytesIO]:
        """Return the recorded audio as WAV bytes in a BytesIO buffer."""
        import soundfile as sf
        with self.lock:
            if not self.audio:
                return None
            # Stack audio chunks into a single numpy array
            audio_np = np.concatenate(self.audio, axis=0)
        buf = io.BytesIO()
        # Write numpy array to buffer as WAV
        sf.write(buf, audio_np, self.sample_rate, format='WAV', subtype='PCM_16')
        buf.seek(0)
        return buf

def transcribe_audio(wav_bytes: io.BytesIO) -> str:
    """Send audio data to Azure Speech-to-Text service and return transcribed text."""
    try:
        headers = {
            "api-key": config.azure_api_key,
        }
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }
        response = requests.post(config.azure_endpoint, headers=headers, files=files)
        response.raise_for_status()
        return response.json().get("text", "")
    except requests.RequestException as e:
        print(f"Transcription failed: {e}")
        return ""

def copy_and_paste(text: str) -> None:
    """Copy the given text to clipboard and simulate Ctrl+V to paste it."""
    pyperclip.copy(text)
    kb.press_and_release('ctrl+v')

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

def create_tray_menu(recorder, icon, on_refresh_mics, on_exit):
    """Create the tray icon menu with microphone selection and other options."""
    return pystray.Menu(
        pystray.MenuItem("Microphones", pystray.Menu(lambda: create_mic_menu(recorder, icon))),
        pystray.MenuItem("Refresh mics", on_refresh_mics),
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

def main() -> None:
    load_sounds()  # Preload sounds once at startup

    recorder = Recorder()

    # --- Hotkey Management ---
    
    # Using a set to track currently pressed modifier keys
    pressed_keys = set()
    
    # The combination we are looking for
    hotkey_combo = {keyboard.Key.ctrl, keyboard.Key.cmd}
    
    # State variables for hotkey logic
    combo_activated = False
    record_start_time = 0.0
    MIN_RECORD_DURATION = 1.0  # Minimum recording duration in seconds

    def on_press(key: keyboard.Key) -> None:
        nonlocal combo_activated, record_start_time
        
        # Allow ESC to cancel an active recording without transcribing
        if key == keyboard.Key.esc and recorder.recording:
            play_click("cancel")
            print("Recording canceled.")
            recorder.cancel()
            combo_activated = False # Reset combo flag
            pressed_keys.clear() # Clear all pressed keys
            return

        # Normalize left/right modifier keys
        if key in CTRL_KEYS:
            key = keyboard.Key.ctrl
        elif key in WIN_KEYS:
            key = keyboard.Key.cmd

        if key in hotkey_combo:
            pressed_keys.add(key)
        
        # If the full combo is pressed and we are not already recording, start.
        if hotkey_combo.issubset(pressed_keys) and not recorder.recording:
            # This check prevents re-triggering if keys are held down.
            if not combo_activated:
                combo_activated = True
                play_click("start")
                print("Recording started... (release to transcribe)")
                record_start_time = time.time()
                recorder.start()

    def on_release(key: keyboard.Key) -> None:
        nonlocal combo_activated
        
        # Normalize left/right modifier keys
        if key in CTRL_KEYS:
            key = keyboard.Key.ctrl
        elif key in WIN_KEYS:
            key = keyboard.Key.cmd
            
        # If a combo key is released and we were recording, stop and process.
        if key in hotkey_combo and recorder.recording:
            recorder.stop()
            duration: float = time.time() - record_start_time

            # Check for minimum recording duration
            if duration < MIN_RECORD_DURATION:
                play_click("cancel")
                print(f"Recording too short ({duration:.2f}s). Canceled.")
            else:
                play_click("stop")
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
                        print("Error during transcription or pasting:", e)

            # Reset state for the next recording.
            # Crucially, clear all keys to prevent the bug where one key remains "pressed".
            pressed_keys.clear()
            combo_activated = False
        
        # Also remove the key from the set if it's released outside of a recording action
        elif key in pressed_keys:
            pressed_keys.remove(key)

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
    icon = pystray.Icon("SpeechToText", create_icon(), "SpeechToText")

    icon.menu = create_tray_menu(recorder, icon, on_refresh_mics, on_exit)
    icon.run()

if __name__ == "__main__":
    main()


