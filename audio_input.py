import io
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import logging

logger = logging.getLogger("AudioTranscriber")

_HAS_FASTER_WHISPER = None
_HAS_TRANSFORMERS_WHISPER = None

class AudioTranscriber:
    """Audio transcription with Faster Whisper (preferred) and Transformers fallback"""
    
    def __init__(self, model_name="base"):
        """Initialize audio transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        logger.info("Initializing AudioTranscriber...")
        self.device = "cpu"
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.mode = None

        global _HAS_FASTER_WHISPER, _HAS_TRANSFORMERS_WHISPER

        # Try faster-whisper first (more efficient)
        if _HAS_FASTER_WHISPER is None:
            try:
                from faster_whisper import WhisperModel
                _HAS_FASTER_WHISPER = True
            except Exception:
                _HAS_FASTER_WHISPER = False

        if _HAS_FASTER_WHISPER:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading faster-whisper model: {model_name}")
                self.model = WhisperModel(model_name, device=self.device, compute_type="int8")
                self.mode = "faster-whisper"
                logger.info("✓ faster-whisper initialized successfully")
                return
            except Exception as e:
                logger.warning(f"faster-whisper initialization failed: {e}")
                _HAS_FASTER_WHISPER = False

        # Fall back to transformers Whisper
        if _HAS_TRANSFORMERS_WHISPER is None:
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                import torch
                _HAS_TRANSFORMERS_WHISPER = True
            except Exception:
                _HAS_TRANSFORMERS_WHISPER = False

        if _HAS_TRANSFORMERS_WHISPER:
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                import torch
                
                logger.info(f"Loading transformers Whisper model: {model_name}")
                model_id = f"openai/whisper-{model_name}"
                
                self.processor = WhisperProcessor.from_pretrained(model_id)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
                
                # Disable forced decoder IDs for better transcription
                try:
                    self.model.config.forced_decoder_ids = None
                except Exception:
                    pass
                
                self.mode = "transformers-whisper"
                logger.info("✓ transformers Whisper initialized successfully")
                return
                
            except Exception as e:
                logger.error(f"transformers Whisper initialization failed: {e}")
                _HAS_TRANSFORMERS_WHISPER = False

        # If both failed
        raise RuntimeError(
            "No Whisper backend available. Install either:\n"
            "  - faster-whisper: pip install faster-whisper\n"
            "  - transformers: pip install transformers torch"
        )

    def _audio_bytes_to_np(self, audio_bytes: bytes) -> tuple:
        """Convert audio bytes to numpy array at 16kHz (Whisper requirement)
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Use pydub to handle various audio formats
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Convert to mono, 16kHz (Whisper requirement)
            seg = seg.set_channels(1).set_frame_rate(16000)
            
            # Export to WAV in memory
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            buf.seek(0)
            
            # Read with soundfile
            data, sr = sf.read(buf)
            
            # Ensure float32 format
            data = np.asarray(data, dtype=np.float32)
            
            return data, sr
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            raise

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Transcribed text string
        """
        if not audio_bytes:
            logger.warning("Empty audio bytes received")
            return ""

        try:
            logger.info(f"Transcribing audio: {len(audio_bytes)} bytes")
            
            # Convert to numpy array
            audio_np, sr = self._audio_bytes_to_np(audio_bytes)
            
            logger.debug(f"Audio converted: {audio_np.shape}, sr={sr}Hz")

            if self.mode == "faster-whisper":
                # faster-whisper transcription
                buf = io.BytesIO()
                sf.write(buf, audio_np, sr, format="WAV")
                buf.seek(0)
                
                segments, info = self.model.transcribe(
                    buf, 
                    beam_size=5,
                    language="en",  # Can be changed to "ta" for Tamil
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Combine all segments
                text = " ".join([seg.text for seg in segments])
                text = text.strip()
                
                logger.info(f"✓ Transcription complete: {len(text)} chars")
                return text

            elif self.mode == "transformers-whisper":
                # transformers Whisper transcription
                import torch
                
                # Prepare input
                inputs = self.processor(
                    audio_np, 
                    sampling_rate=sr, 
                    return_tensors="pt"
                )
                
                input_features = inputs.input_features.to(self.device)
                
                # Generate transcription
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=448,
                        num_beams=5,
                        language="en"
                    )
                
                # Decode
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )
                
                text = transcription[0].strip() if transcription else ""
                
                logger.info(f"✓ Transcription complete: {len(text)} chars")
                return text

            else:
                logger.error("No valid transcription mode")
                return ""

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio from file path
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            return self.transcribe_audio(audio_bytes)
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return ""