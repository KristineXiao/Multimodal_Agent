"""
Speech utilities for multimodal Boston Guide Agent
Handles Speech-to-Text (STT) and Text-to-Speech (TTS) functionality
"""

import os
import time
import tempfile
import warnings
import threading
from typing import Optional, Tuple
import pyaudio
import wave
import whisper
import pyttsx3
import keyboard
import numpy as np
from gtts import gTTS
import pygame

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class SpeechManager:
    def __init__(self, whisper_model_size: str = "base"):
        """
        Initialize Speech Manager with STT and TTS capabilities
        
        Args:
            whisper_model_size: Size of Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.whisper_model = None
        self.whisper_model_size = whisper_model_size
        self.tts_engine = None
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.recording = False
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Check gTTS availability
        self.gtts_available = self._check_gtts_availability()
        
        self._load_whisper_model()
        self._setup_tts_engine()
    
    def _check_gtts_availability(self):
        """Check if gTTS is available"""
        try:
            from gtts import gTTS
            return True
        except ImportError:
            print("⚠️ gTTS not available - will use local TTS only")
            return False
    
    def _load_whisper_model(self):
        """Load Whisper model for speech recognition"""
        try:
            print(f"🎤 Loading Whisper {self.whisper_model_size} model...")
            self.whisper_model = whisper.load_model(self.whisper_model_size)
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            self.whisper_model = None
    
    def _setup_tts_engine(self):
        """Setup pyttsx3 TTS engine"""
        try:
            # Create a fresh engine instance
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
            
            self.tts_engine = pyttsx3.init(driverName='sapi5')  # Explicitly use SAPI5 on Windows
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer English voices
                for voice in voices:
                    if 'english' in voice.name.lower() and ('zira' in voice.name.lower() or 'david' in voice.name.lower()):
                        self.tts_engine.setProperty('voice', voice.id)
                        print(f"🎤 Using voice: {voice.name}")
                        break
                else:
                    # Use first English voice available
                    for voice in voices:
                        if 'english' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            print(f"🎤 Using voice: {voice.name}")
                            break
                    else:
                        # Fallback to first voice
                        self.tts_engine.setProperty('voice', voices[0].id)
                        print(f"🎤 Using voice: {voices[0].name}")
            
            # Set speech rate and volume - more conservative settings
            self.tts_engine.setProperty('rate', 160)  # Slightly slower
            self.tts_engine.setProperty('volume', 0.8)  # Slightly quieter
            
            print("✅ TTS engine initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing TTS engine: {e}")
            self.tts_engine = None
    
    def record_audio(self, max_duration: int = 10) -> Optional[str]:
        """
        Record audio from microphone and save to temporary file
        
        Args:
            max_duration: Maximum recording duration in seconds
            
        Returns:
            Path to recorded audio file or None if failed
        """
        if not self.whisper_model:
            print("❌ Whisper model not available")
            return None
        
        try:
            # Initialize PyAudio
            audio = pyaudio.PyAudio()
            
            # Check if microphone is available
            try:
                # Test microphone access first
                info = audio.get_host_api_info_by_index(0)
                print(f"🎤 Using audio device: {info.get('name', 'Default')}")
                
                stream = audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    input_device_index=None,  # Use default device
                    frames_per_buffer=self.chunk
                )
            except Exception as e:
                print(f"❌ Could not access microphone: {e}")
                print("💡 Make sure microphone permissions are granted and device is not in use")
                audio.terminate()
                return None
            
            print("🎤 Recording... Press SPACE to stop, or wait for automatic stop")
            print(f"   (Maximum duration: {max_duration} seconds)")
            
            frames = []
            self.recording = True
            start_time = time.time()
            
            # Record in a separate thread to allow for key detection
            def record_loop():
                while self.recording and (time.time() - start_time) < max_duration:
                    try:
                        data = stream.read(self.chunk, exception_on_overflow=False)
                        frames.append(data)
                    except Exception as e:
                        print(f"⚠️ Recording error: {e}")
                        break
            
            # Start recording thread
            record_thread = threading.Thread(target=record_loop)
            record_thread.start()
            
            # Monitor for spacebar press
            while self.recording and (time.time() - start_time) < max_duration:
                if keyboard.is_pressed('space'):
                    print("🛑 Stopping recording...")
                    self.recording = False
                    break
                time.sleep(0.1)
            
            # Ensure recording stops
            self.recording = False
            record_thread.join(timeout=1)
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            if not frames:
                print("❌ No audio recorded")
                return None
            
            # Save recorded audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_filename = temp_file.name
            temp_file.close()  # Close the file handle so we can write to it
            
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            duration = time.time() - start_time
            print(f"✅ Recorded {duration:.1f} seconds of audio")
            
            return temp_filename
            
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return None
    
    def speech_to_text(self, audio_file: str) -> Optional[str]:
        """
        Convert audio file to text using Whisper with numpy array approach (Windows fix)
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.whisper_model:
            print("❌ Whisper model not available")
            return None
        
        try:
            print("🔄 Transcribing audio...")
            
            # Check if audio file exists and is readable
            if not os.path.exists(audio_file):
                print(f"❌ Audio file not found: {audio_file}")
                return None
            
            if os.path.getsize(audio_file) == 0:
                print("❌ Audio file is empty")
                return None
            
            # Use librosa to load audio as numpy array (Windows Whisper file path fix)
            try:
                import librosa
                print("📥 Loading audio with librosa...")
                audio_data, sample_rate = librosa.load(audio_file, sr=16000)
                print(f"✅ Audio loaded: {len(audio_data)} samples, {len(audio_data)/sample_rate:.1f}s")
                
                # Transcribe using numpy array instead of file path
                result = self.whisper_model.transcribe(
                    audio_data,  # Pass numpy array directly
                    language="en",
                    fp16=False,
                    verbose=False
                )
                
            except ImportError:
                print("⚠️ librosa not available, trying file path method...")
                # Fallback to file path (may fail on Windows)
                result = self.whisper_model.transcribe(
                    audio_file,
                    language="en",
                    fp16=False,
                    verbose=False
                )
            
            text = result["text"].strip()
            
            if text and len(text) > 0:
                print(f"✅ Transcription: '{text}'")
                return text
            else:
                print("❌ No speech detected (empty transcription)")
                return None
                
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            except:
                pass
    
    def text_to_speech_local(self, text: str) -> bool:
        """
        Convert text to speech using local pyttsx3 engine with better reliability
        
        Args:
            text: Text to convert to speech
            
        Returns:
            True if successful, False otherwise
        """
        if not self.tts_engine:
            print("❌ TTS engine not available")
            return False
        
        if not text or not text.strip():
            return False
        
        try:
            print("🔊 Speaking...")
            
            # Create a fresh engine for each speech to avoid stuck states
            import pyttsx3
            temp_engine = pyttsx3.init(driverName='sapi5')
            
            # Copy settings from main engine
            voices = temp_engine.getProperty('voices')
            if voices:
                # Use same voice as main engine
                current_voice = self.tts_engine.getProperty('voice')
                temp_engine.setProperty('voice', current_voice)
            
            temp_engine.setProperty('rate', 160)
            temp_engine.setProperty('volume', 0.8)
            
            # Speak the text
            temp_engine.say(text.strip())
            temp_engine.runAndWait()
            
            # Clean up temp engine
            temp_engine.stop()
            del temp_engine
            
            return True
            
        except Exception as e:
            print(f"❌ Error with local TTS: {e}")
            
            # Fallback: try with main engine one more time
            try:
                print("🔄 Retrying with main engine...")
                self.tts_engine.stop()
                time.sleep(0.1)
                self.tts_engine.say(text.strip())
                self.tts_engine.runAndWait()
                return True
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return False
    
    def text_to_speech_gtts(self, text: str, lang: str = "en") -> bool:
        """
        Convert text to speech using Google TTS (gTTS) - improved for consecutive calls
        Requires internet connection but provides better quality and reliability
        
        Args:
            text: Text to convert to speech
            lang: Language code (default: "en")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to temporary file with unique name
            import uuid
            temp_file_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
            tts.save(temp_file_path)
            
            # Ensure file is written
            time.sleep(0.1)
            
            # Stop any currently playing audio
            pygame.mixer.music.stop()
            time.sleep(0.1)
            
            # Load and play audio
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish with timeout
            timeout = 120  # Increased to 2 minutes for long introductions
            start_time = time.time()
            while pygame.mixer.music.get_busy() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            # Check if we timed out
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                print(f"⚠️ TTS timed out after {elapsed_time:.1f} seconds")
            
            # Ensure audio fully completes
            time.sleep(0.5)
            
            # Clean up
            try:
                os.unlink(temp_file_path)
            except:
                pass  # File might be locked, ignore cleanup errors
            
            return True
            
        except Exception as e:
            print(f"❌ Error with Google TTS: {e}")
            return False
    
    def text_to_speech(self, text: str, use_gtts: bool = True) -> bool:
        """
        Convert text to speech with working method
        
        Args:
            text: Text to convert to speech
            use_gtts: Try Google TTS first (requires internet)
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # Clean up text for better speech
        clean_text = text.replace("**", "").replace("*", "").replace("#", "")
        clean_text = clean_text.replace("🔊", "").replace("🎤", "").replace("✅", "").replace("❌", "")
        clean_text = clean_text.replace("🌟", "").replace("📍", "").replace("👋", "")
        clean_text = clean_text.strip()
        
        if not clean_text:
            return False
        
        # Switch to gTTS for better reliability and quality
        if use_gtts and self.gtts_available:
            return self.text_to_speech_gtts(clean_text)
        else:
            # Fallback to local TTS if gTTS fails
            return self._speak_with_fresh_engine(clean_text)
    
    def _speak_with_fresh_engine(self, text: str) -> bool:
        """Use main engine with proper synchronization and very long delays"""
        try:
            import time
            
            if not self.tts_engine:
                print("❌ No TTS engine available")
                return False
            
            # Clear any pending speech
            try:
                self.tts_engine.stop()
            except:
                pass
            
            # Wait a moment for engine to be ready
            time.sleep(0.3)
            
            # Queue the text
            self.tts_engine.say(text)
            
            # Run and wait - this should block until complete
            self.tts_engine.runAndWait()
            
            # Ensure Windows audio completes
            time.sleep(2.0)
            
            return True
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return False
    
    def get_voice_input(self, prompt: str, max_duration: int = 10) -> Optional[str]:
        """
        Get voice input from user with audio recording and transcription
        
        Args:
            prompt: Text prompt to display to user
            max_duration: Maximum recording duration in seconds
            
        Returns:
            Transcribed text or None if failed
        """
        print(f"\n{prompt}")
        print("🎤 Speak now...")
        
        # Record audio
        audio_file = self.record_audio(max_duration)
        if not audio_file:
            return None
        
        # Convert to text
        text = self.speech_to_text(audio_file)
        return text
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.tts_engine:
                self.tts_engine.stop()
            pygame.mixer.quit()
        except:
            pass

# Convenience functions for easy usage
def create_speech_manager(model_size: str = "base") -> SpeechManager:
    """Create and return a SpeechManager instance"""
    return SpeechManager(model_size)

def voice_to_text(prompt: str = "Speak now:", max_duration: int = 10) -> Optional[str]:
    """Quick voice input function"""
    manager = create_speech_manager()
    try:
        return manager.get_voice_input(prompt, max_duration)
    finally:
        manager.cleanup()

def text_to_voice(text: str, use_gtts: bool = True) -> bool:
    """Quick text-to-speech function"""
    manager = create_speech_manager()
    try:
        return manager.text_to_speech(text, use_gtts)
    finally:
        manager.cleanup()