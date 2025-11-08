import os
import time
import tempfile
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

_HAS_COQUI = None
_HAS_GTTS = None
_HAS_PYDUB = None

logger = logging.getLogger("TextToSpeech")

SILENT_WAV = Path("data/silent.wav")
SR = 22050

def _ensure_silent_wav(path: Path = SILENT_WAV, sr: int = SR, duration_s: float = 0.5) -> str:
    """Ensure silent WAV file exists"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        samples = np.zeros(int(sr * duration_s), dtype="float32")
        sf.write(str(path), samples, sr)
    return str(path)

_ensure_silent_wav()

def _validate_wav_file(file_path: str, min_duration: float = 0.1) -> bool:
    """Validate that a WAV file exists, is readable, and has minimum duration"""
    try:
        if not os.path.exists(file_path):
            return False
        
        # Check file size first
        if os.path.getsize(file_path) < 500:
            return False
        
        with sf.SoundFile(file_path) as f:
            duration = f.frames / f.samplerate
            return duration >= min_duration and f.frames > 0
    except Exception as e:
        logger.debug(f"WAV validation failed for {file_path}: {e}")
        return False

class TextToSpeech:
    """Enhanced TTS with Coqui (offline) and gTTS (online) fallback"""

    def __init__(self):
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

        self.tts_model = None

        global _HAS_COQUI
        if _HAS_COQUI is None:
            try:
                from TTS.api import TTS
                _HAS_COQUI = True
            except Exception:
                _HAS_COQUI = False

        # Try to load Coqui TTS
        if _HAS_COQUI:
            try:
                logger.info("Loading Coqui TTS (CPU mode)...")
                from TTS.api import TTS
                self.tts_model = TTS("tts_models/en/vctk/vits", gpu=False)
                logger.info("✓ Coqui TTS loaded successfully")
            except TypeError:
                try:
                    from TTS.api import TTS
                    self.tts_model = TTS("tts_models/en/vctk/vits")
                    logger.info("✓ Coqui TTS loaded (fallback mode)")
                except Exception as e2:
                    logger.warning(f"Coqui TTS fallback failed: {e2}")
                    self.tts_model = None
            except Exception as e:
                logger.warning(f"Coqui TTS load failed: {e}")
                self.tts_model = None
        else:
            logger.info("Coqui TTS not installed; will use gTTS")

    def synthesize_speech(self, text: str, output_path: str, online: bool = True, lang: str = "en") -> str:
        """Synthesize speech with robust error handling and validation
        
        Args:
            text: Text to synthesize
            output_path: Output WAV file path
            online: Whether to use online TTS (gTTS)
            lang: Language code (en, ta, etc.)
            
        Returns:
            Path to generated WAV file (guaranteed to be valid or silent fallback)
        """
        if not text or not text.strip():
            logger.debug("Empty text, returning silent audio")
            return _ensure_silent_wav()

        text = text.strip()
        
        # Truncate very long text
        if len(text) > 1000:
            text = text[:1000] + "..."
            logger.info("Text truncated to 1000 characters")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Synthesizing speech: {len(text)} chars, lang={lang}")

        # For Tamil, use gTTS (better quality)
        if lang == "ta":
            logger.info("Tamil language detected, using gTTS")
            return self._synthesize_with_gtts(text, str(out_path), lang="ta")

        # For English, try Coqui first (offline), then gTTS
        if self.tts_model and lang == "en":
            logger.info("Attempting Coqui TTS synthesis...")
            result = self._synthesize_with_coqui(text, str(out_path))
            if result:
                return result
            logger.info("Coqui TTS failed, falling back to gTTS")

        # Fall back to gTTS
        return self._synthesize_with_gtts(text, str(out_path), lang=lang)
    
    def _synthesize_with_coqui(self, text: str, output_path: str) -> Optional[str]:
        """Synthesize with Coqui TTS (offline)"""
        if not self.tts_model:
            return None
        
        try:
            # Determine speaker
            speaker = None
            try:
                if hasattr(self.tts_model, "model_name") and "vctk" in getattr(self.tts_model, "model_name", "").lower():
                    speaker = "p227"  # Female voice
            except Exception:
                pass

            # Generate audio
            logger.debug(f"Generating with Coqui, speaker={speaker}")
            self.tts_model.tts_to_file(text=text, speaker=speaker, file_path=output_path)

            # Wait for file with timeout
            max_wait = 12.0
            wait_interval = 0.3
            elapsed = 0.0
            
            while elapsed < max_wait:
                if os.path.exists(output_path):
                    # Give a bit more time for file to be fully written
                    time.sleep(0.5)
                    
                    if _validate_wav_file(output_path, min_duration=0.2):
                        file_size = os.path.getsize(output_path)
                        logger.info(f"✓ Coqui TTS generated audio: {file_size} bytes")
                        return output_path
                
                time.sleep(wait_interval)
                elapsed += wait_interval

            logger.warning(f"Coqui TTS file validation failed after {max_wait}s")
            
            # Clean up invalid file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

        except Exception as e:
            logger.warning(f"Coqui TTS error: {e}")
        
        return None
    
    def _synthesize_with_gtts(self, text: str, output_path: str, lang: str = "en") -> str:
        """Synthesize with gTTS (online) with robust error handling"""
        global _HAS_GTTS, _HAS_PYDUB
        
        if _HAS_GTTS is None:
            try:
                from gtts import gTTS
                _HAS_GTTS = True
            except Exception:
                _HAS_GTTS = False
                logger.warning("gTTS not available, install: pip install gtts")

        if _HAS_PYDUB is None:
            try:
                from pydub import AudioSegment
                _HAS_PYDUB = True
            except Exception:
                _HAS_PYDUB = False
                logger.warning("pydub not available, install: pip install pydub")

        if not _HAS_GTTS:
            logger.warning("gTTS unavailable, returning silent audio")
            return _ensure_silent_wav()

        tmp_mp3 = None
        try:
            # Create temp MP3 file
            tmp_fd, tmp_mp3 = tempfile.mkstemp(suffix=".mp3")
            os.close(tmp_fd)

            from gtts import gTTS
            
            logger.info(f"Generating gTTS audio: lang={lang}")
            
            # Generate MP3 with gTTS
            gtts_obj = gTTS(text=text, lang=lang, slow=False)
            gtts_obj.save(tmp_mp3)
            
            # Verify MP3 was created
            if not os.path.exists(tmp_mp3) or os.path.getsize(tmp_mp3) < 100:
                raise Exception(f"gTTS did not generate valid MP3 for lang={lang}")
            
            logger.info(f"✓ gTTS generated MP3: {os.path.getsize(tmp_mp3)} bytes")

            # Convert MP3 to WAV
            if _HAS_PYDUB:
                try:
                    from pydub import AudioSegment
                    
                    logger.debug("Converting MP3 to WAV with pydub")
                    audio = AudioSegment.from_file(tmp_mp3, format="mp3")
                    
                    if len(audio) > 0:
                        audio.export(output_path, format="wav")
                        time.sleep(0.5)  # Give filesystem time to flush
                        
                        if _validate_wav_file(output_path):
                            file_size = os.path.getsize(output_path)
                            logger.info(f"✓ Successfully converted to WAV: {file_size} bytes")
                            return output_path
                    else:
                        raise Exception("Empty audio from gTTS")
                        
                except Exception as conv_e:
                    logger.warning(f"pydub conversion failed: {conv_e}")
            else:
                # Try ffmpeg conversion as fallback
                try:
                    import subprocess
                    logger.debug("Attempting ffmpeg conversion")
                    
                    result = subprocess.run(
                        ['ffmpeg', '-i', tmp_mp3, '-acodec', 'pcm_s16le', '-ar', '22050', output_path],
                        capture_output=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0 and _validate_wav_file(output_path):
                        logger.info(f"✓ FFmpeg converted to WAV successfully")
                        return output_path
                        
                except Exception as ffmpeg_e:
                    logger.warning(f"ffmpeg conversion failed: {ffmpeg_e}")

        except Exception as e:
            logger.error(f"gTTS synthesis failed for lang={lang}: {e}")
            
        finally:
            # Clean up temp MP3
            if tmp_mp3 and os.path.exists(tmp_mp3):
                try:
                    os.remove(tmp_mp3)
                except:
                    pass

        logger.warning(f"All TTS methods failed for lang={lang}, returning silent audio")
        return _ensure_silent_wav()