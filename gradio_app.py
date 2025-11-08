import os
import time
import io
import tempfile
import gradio as gr
from datetime import datetime
import threading
import shutil
import json
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import socket
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import queue
from functools import lru_cache, wraps
from contextlib import contextmanager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env file loading")

import numpy as np
import soundfile as sf
from pathlib import Path

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Centralized configuration management"""
    DIAGNOSIS_TIMEOUT = 15.0  # Increased for better LLM response
    RAG_TIMEOUT = 8.0  # Increased for better retrieval
    LLM_TIMEOUT = 12.0  # Increased for better generation
    TTS_TIMEOUT = 15.0  # Increased significantly for reliable audio
    MAX_WORKERS = 4
    MAX_CONVERSATION_TURNS = 10
    MAX_SYMPTOMS_HISTORY = 50
    TEMP_DIR = os.path.join(os.getcwd(), "temp_outputs")
    SILENT_WAV_PATH = "data/silent.wav"
    SAMPLE_RATE = 22050
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_AUDIO_LENGTH = 300  # 5 minutes
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs("data", exist_ok=True)

Config.ensure_directories()

# ============================================================================
# UTILITY FUNCTIONS & DECORATORS
# ============================================================================

def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

def safe_execute(default_return=None, log_error=True):
    """Decorator for safe function execution with error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator

@contextmanager
def timeout_context(seconds: float):
    """Context manager for timeout operations"""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False)

def sanitize_input(text: str, max_length: int = 5000) -> str:
    """Sanitize and validate user input"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Input truncated to {max_length} characters")
    
    # Remove potentially harmful patterns
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE)
    
    return text

# ============================================================================
# LAZY MODULE LOADING WITH CACHING
# ============================================================================

class ModuleLoader:
    """Singleton for lazy module loading with caching"""
    _instance = None
    _modules = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_module(self, module_name: str, class_name: str, *args, **kwargs):
        """Get or create module instance"""
        with self._lock:
            if module_name not in self._modules:
                try:
                    logger.info(f"Loading {class_name}...")
                    module = __import__(f'modules.{module_name}', fromlist=[class_name])
                    cls = getattr(module, class_name)
                    self._modules[module_name] = cls(*args, **kwargs)
                    logger.info(f"✓ {class_name} loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load {class_name}: {e}")
                    self._modules[module_name] = None
            return self._modules[module_name]
    
    def reset_module(self, module_name: str):
        """Reset a specific module"""
        with self._lock:
            if module_name in self._modules:
                del self._modules[module_name]

loader = ModuleLoader()

# Module getter functions
def get_llm_agent():
    return loader.get_module('llm_agent', 'LLMAgent')

def get_medical_agent():
    llm = get_llm_agent()
    rag = get_rag_retriever()
    parser = get_symptom_parser()
    if llm and rag and parser:
        return loader.get_module('medical_agent', 'MedicalAgent', llm, rag, parser)
    return None

def get_symptom_parser():
    return loader.get_module('symptom_parser', 'SymptomParser')

def get_rag_retriever():
    return loader.get_module('rag_retriever', 'RAGRetriever')

def get_tts():
    return loader.get_module('tts_output', 'TextToSpeech')

def get_hospital_locator():
    return loader.get_module('hospital_locator', 'HospitalLocator')

def get_metrics_tracker():
    return loader.get_module('metrics_tracker', 'MetricsTracker')

def get_audio_transcriber():
    return loader.get_module('audio_input', 'AudioTranscriber')

# ============================================================================
# ENHANCED DATA STRUCTURES
# ============================================================================

@dataclass
class SymptomEntry:
    """Individual symptom entry with metadata"""
    symptom: str
    timestamp: datetime
    severity: str = "unknown"
    context: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "symptom": self.symptom,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "context": self.context
        }

@dataclass
class ConversationTurn:
    """Single conversation turn with rich metadata"""
    user_input: str
    bot_response: str
    timestamp: datetime
    severity: str
    symptoms_mentioned: List[str]
    action_taken: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "user": self.user_input,
            "assistant": self.bot_response,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "symptoms": self.symptoms_mentioned,
            "action": self.action_taken
        }

@dataclass
class ConversationState:
    """Enhanced conversation state with comprehensive tracking"""
    session_id: str = field(default_factory=lambda: f"session_{int(time.time())}_{os.getpid()}")
    symptoms_history: List[SymptomEntry] = field(default_factory=list)
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    raw_inputs: List[str] = field(default_factory=list)
    current_severity: str = "unknown"
    hospital_recommended: bool = False
    last_assessment: str = ""
    session_start: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_symptom(self, symptom: str, severity: str = "unknown", context: str = ""):
        """Add a symptom with metadata"""
        entry = SymptomEntry(
            symptom=symptom,
            timestamp=datetime.now(),
            severity=severity,
            context=context
        )
        self.symptoms_history.append(entry)
        
        # Keep only recent symptoms
        if len(self.symptoms_history) > Config.MAX_SYMPTOMS_HISTORY:
            self.symptoms_history = self.symptoms_history[-Config.MAX_SYMPTOMS_HISTORY:]
    
    def add_turn(self, user_input: str, bot_response: str, 
                 severity: str, symptoms: List[str], action: str = ""):
        """Add a conversation turn"""
        turn = ConversationTurn(
            user_input=user_input,
            bot_response=bot_response,
            timestamp=datetime.now(),
            severity=severity,
            symptoms_mentioned=symptoms,
            action_taken=action
        )
        self.conversation_turns.append(turn)
        self.current_severity = severity
        self.last_assessment = bot_response
        
        # Keep only recent turns
        if len(self.conversation_turns) > Config.MAX_CONVERSATION_TURNS:
            self.conversation_turns = self.conversation_turns[-Config.MAX_CONVERSATION_TURNS:]
    
    def get_context_string(self, max_turns: int = 5) -> str:
        """Get formatted conversation history"""
        if not self.conversation_turns:
            return ""
        
        recent_turns = self.conversation_turns[-max_turns:]
        context_parts = ["Previous conversation:"]
        
        for turn in recent_turns:
            context_parts.append(f"\nPatient: {turn.user_input}")
            context_parts.append(f"Assistant: {turn.bot_response[:300]}...")
            context_parts.append(f"Severity: {turn.severity}")
        
        return "\n".join(context_parts)
    
    def get_all_symptoms_string(self) -> str:
        """Get unique symptoms as formatted string"""
        if not self.symptoms_history:
            return "No symptoms reported yet"
        
        unique_symptoms = list(dict.fromkeys(
            entry.symptom for entry in self.symptoms_history
        ))
        return ", ".join(unique_symptoms)
    
    def get_symptom_timeline(self) -> str:
        """Get symptoms with timeline"""
        if not self.symptoms_history:
            return "No symptoms recorded"
        
        timeline = []
        for entry in self.symptoms_history[-10:]:  # Last 10 symptoms
            time_str = entry.timestamp.strftime("%H:%M")
            timeline.append(f"• {entry.symptom} (at {time_str})")
        
        return "\n".join(timeline)
    
    def needs_hospital_recommendation(self) -> bool:
        """Check if hospital recommendation is needed"""
        high_severity_keywords = ["high", "emergency", "severe", "urgent", "immediate"]
        return any(keyword in self.current_severity.lower() 
                  for keyword in high_severity_keywords)
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        elapsed = (datetime.now() - self.session_start).total_seconds()
        return elapsed > Config.SESSION_TIMEOUT
    
    def to_dict(self) -> Dict:
        """Serialize state to dictionary"""
        return {
            "session_id": self.session_id,
            "symptoms": [s.to_dict() for s in self.symptoms_history],
            "turns": [t.to_dict() for t in self.conversation_turns],
            "current_severity": self.current_severity,
            "hospital_recommended": self.hospital_recommended,
            "session_start": self.session_start.isoformat(),
            "metadata": self.metadata
        }

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """Manage conversation sessions with automatic cleanup"""
    _instance = None
    _sessions: Dict[str, ConversationState] = {}
    _lock = threading.Lock()
    _cleanup_thread = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._start_cleanup_thread()
        return cls._instance
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_expired_sessions,
                daemon=True
            )
            self._cleanup_thread.start()
    
    def _cleanup_expired_sessions(self):
        """Periodically clean up expired sessions"""
        while True:
            time.sleep(300)  # Check every 5 minutes
            with self._lock:
                expired = [
                    sid for sid, state in self._sessions.items()
                    if state.is_expired()
                ]
                for sid in expired:
                    logger.info(f"Cleaning up expired session: {sid}")
                    del self._sessions[sid]
    
    def get_or_create(self, session_id: Optional[str] = None) -> ConversationState:
        """Get existing session or create new one"""
        with self._lock:
            if not session_id or session_id not in self._sessions:
                state = ConversationState()
                self._sessions[state.session_id] = state
                logger.info(f"Created new session: {state.session_id}")
                
                # Track in metrics
                metrics = get_metrics_tracker()
                if metrics:
                    metrics.start_session(state.session_id)
                
                # Reset LLM conversation
                llm = get_llm_agent()
                if llm:
                    llm.reset_conversation()
                
                return state
            
            return self._sessions[session_id]
    
    def get(self, session_id: str) -> Optional[ConversationState]:
        """Get session by ID"""
        with self._lock:
            return self._sessions.get(session_id)
    
    def delete(self, session_id: str):
        """Delete a session"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
    
    def count(self) -> int:
        """Get active session count"""
        with self._lock:
            return len(self._sessions)

session_manager = SessionManager()

# ============================================================================
# AUDIO UTILITIES
# ============================================================================

@safe_execute(default_return=None)
def ensure_silent_wav() -> str:
    """Generate or return path to silent wav file"""
    silent_path = Config.SILENT_WAV_PATH
    
    if os.path.exists(silent_path):
        return silent_path
    
    import wave
    
    with wave.open(silent_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(Config.SAMPLE_RATE)
        silence = np.zeros(Config.SAMPLE_RATE, dtype=np.int16)
        wav_file.writeframes(silence.tobytes())
    
    logger.info(f"Created silent wav file at {silent_path}")
    return silent_path

# ============================================================================
# MEDICAL KNOWLEDGE BASE
# ============================================================================

class MedicalKnowledgeBase:
    """Enhanced medical knowledge with structured information"""
    
    SEVERITY_KEYWORDS = {
        "emergency": ["chest pain", "difficulty breathing", "shortness of breath",
                     "severe bleeding", "loss of consciousness", "stroke", "seizure",
                     "severe head injury", "poisoning", "severe allergic reaction"],
        "high": ["high fever", "persistent vomiting", "severe dizziness",
                "rapid heartbeat", "severe abdominal pain", "severe pain",
                "confusion", "severe dehydration"],
        "moderate": ["fever", "headache", "vomiting", "dizziness", "cough",
                    "body aches", "fatigue", "moderate pain"],
        "mild": ["mild headache", "mild fever", "minor ache", "mild discomfort"]
    }
    
    SYMPTOM_ADVICE = {
        "chest pain": {
            "urgency": "emergency",
            "advice": """⚠️ **URGENT - CHEST PAIN:**
• Call 108 immediately or go to nearest ER
• Do NOT ignore chest pain - it can indicate serious conditions
• If accompanied by shortness of breath, sweating, or nausea, this is an emergency
• Sit or lie down until help arrives
• Chew aspirin only if instructed by medical professionals""",
            "warning_signs": ["Radiating pain to arm/jaw", "Sweating", "Nausea", "Shortness of breath"]
        },
        
        "difficulty breathing": {
            "urgency": "emergency",
            "advice": """⚠️ **URGENT - DIFFICULTY BREATHING:**
• Seek immediate emergency care
• Sit upright to ease breathing
• Stay calm and breathe slowly and deeply
• Can indicate asthma, pneumonia, or cardiac issues
• Do not delay - go to hospital immediately""",
            "warning_signs": ["Blue lips/fingers", "Confusion", "Severe wheezing"]
        },
        
        "fever": {
            "urgency": "moderate",
            "advice": """**For Fever:**
• Monitor temperature regularly (concern if >101°F/38.3°C)
• Stay well hydrated - drink water, herbal tea, warm broth
• Get adequate rest to help fight infection
• Use fever-reducing medication as directed
• Dress lightly and use light bedding
• See doctor if fever lasts >3 days or is very high (>104°F/40°C)""",
            "warning_signs": ["Fever >103°F", "Severe headache with fever", "Stiff neck", "Rash"]
        },
        
        "headache": {
            "urgency": "mild",
            "advice": """**For Headache:**
• Rest in quiet, dark room
• Apply cold compress to forehead or heat to neck
• Stay well hydrated
• Identify and avoid triggers (bright lights, stress, certain foods)
• Most headaches improve with rest
• See doctor if severe, persistent, or with fever/vision changes""",
            "warning_signs": ["Sudden severe headache", "With fever", "Vision changes", "Confusion"]
        },
        
        "cough": {
            "urgency": "mild",
            "advice": """**For Cough:**
• Stay hydrated - warm fluids like tea with honey
• Use honey to soothe throat (NOT for children <1 year)
• Avoid irritants like smoke and strong perfumes
• Use humidifier to add moisture
• Cover mouth when coughing
• See doctor if cough lasts >3 weeks or has blood""",
            "warning_signs": ["Coughing blood", "Severe chest pain", "High fever", "Breathing difficulty"]
        }
    }
    
    @classmethod
    def get_advice_for_symptom(cls, symptom: str) -> Dict[str, Any]:
        """Get structured advice for a symptom"""
        symptom_lower = symptom.lower()
        
        for key, data in cls.SYMPTOM_ADVICE.items():
            if key in symptom_lower:
                return data
        
        # Default advice for unknown symptoms
        return {
            "urgency": "mild",
            "advice": f"""**For {symptom.title()}:**
• Monitor the symptom closely
• Rest and stay hydrated
• Note any changes or new symptoms
• Seek medical help if symptoms worsen or persist""",
            "warning_signs": ["Worsening condition", "High fever", "Severe pain"]
        }
    
    @classmethod
    def assess_severity(cls, symptoms: List[str]) -> Dict[str, str]:
        """Enhanced severity assessment"""
        symptoms_lower = [s.lower() for s in symptoms]
        
        # Check for emergency symptoms
        for symptom in symptoms_lower:
            for emergency_keyword in cls.SEVERITY_KEYWORDS["emergency"]:
                if emergency_keyword in symptom:
                    return {
                        "level": "Emergency",
                        "description": "⚠️ Your symptoms require immediate medical attention.",
                        "immediate_action": "Call 108 for ambulance immediately",
                        "warning_signs": "• Worsening pain or breathing\n• Loss of consciousness\n• Severe bleeding",
                        "consultation_advice": "Go to nearest emergency room immediately",
                        "emoji": "🚨"
                    }
        
        # Check for high severity
        high_count = sum(
            1 for symptom in symptoms_lower
            for high_keyword in cls.SEVERITY_KEYWORDS["high"]
            if high_keyword in symptom
        )
        
        if high_count >= 1 or len(symptoms) >= 5:
            return {
                "level": "High - Needs attention",
                "description": "Your symptoms are concerning and need prompt evaluation.",
                "immediate_action": "Contact your doctor immediately or visit urgent care",
                "warning_signs": "• Symptoms worsening\n• New severe symptoms\n• High fever (>103°F)",
                "consultation_advice": "See a doctor within the next few hours",
                "emoji": "⚠️"
            }
        
        # Check for moderate severity
        if len(symptoms) >= 2 or any(
            any(mod_kw in symptom for mod_kw in cls.SEVERITY_KEYWORDS["moderate"])
            for symptom in symptoms_lower
        ):
            return {
                "level": "Moderate - Should see a doctor",
                "description": "Your symptoms should be evaluated by a healthcare provider soon.",
                "immediate_action": "Schedule doctor's appointment within 24-48 hours",
                "warning_signs": "• Symptoms persisting >3 days\n• Fever >102°F\n• Increasing pain",
                "consultation_advice": "Book an appointment with your doctor",
                "emoji": "⚕️"
            }
        
        # Mild severity
        return {
            "level": "Mild - Monitor symptoms",
            "description": "Your symptoms appear mild but should be monitored.",
            "immediate_action": "Rest and monitor your symptoms",
            "warning_signs": "• Symptoms lasting >1 week\n• New symptoms appearing\n• Worsening condition",
            "consultation_advice": "See a doctor if symptoms persist or worsen",
            "emoji": "ℹ️"
        }

# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

@timer
@safe_execute(default_return=[])
def retrieve_medical_context(symptoms: List[str]) -> List[str]:
    """Retrieve medical context with timeout and caching"""
    if not symptoms:
        return []
    
    retriever = get_rag_retriever()
    if not retriever or not retriever.is_ready():
        logger.warning("RAG retriever not available")
        return []
    
    try:
        with timeout_context(Config.RAG_TIMEOUT) as executor:
            future = executor.submit(retriever.search_by_symptoms, symptoms, k=8)
            results = future.result(timeout=Config.RAG_TIMEOUT)
            
            if results:
                logger.info(f"Retrieved {len(results)} medical documents")
                return results
            
            return []
    
    except FutureTimeoutError:
        logger.warning(f"RAG retrieval timeout after {Config.RAG_TIMEOUT}s")
        return []
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []

@timer
def generate_diagnosis(symptoms: List[str], retrieved_context: List[str],
                       state: ConversationState, user_input: str) -> str:
    """Generate diagnosis with enhanced context - FIXED to prefer API responses"""
    agent = get_llm_agent()
    if not agent:
        logger.warning("LLM agent not available, using fallback")
        return generate_fallback_diagnosis(symptoms, state, user_input, retrieved_context)
    
    try:
        # Build comprehensive context
        conversation_history = state.get_context_string()
        all_symptoms = state.get_all_symptoms_string()
        symptom_timeline = state.get_symptom_timeline()
        
        # Format medical context
        context_str = ""
        if retrieved_context:
            context_str = "\n\nRelevant medical information:\n" + "\n".join([
                f"• {doc[:250]}..." for doc in retrieved_context[:4]
            ])
        
        patient_info = f"""All symptoms: {all_symptoms}
Current concern: {user_input}
Symptom timeline:
{symptom_timeline}{context_str}"""
        
        # Generate with timeout
        with timeout_context(Config.LLM_TIMEOUT) as executor:
            future = executor.submit(
                agent.generate_diagnosis,
                symptoms=symptoms,
                retrieved_context=retrieved_context,
                patient_info=patient_info,
                conversation_history=conversation_history,
                user_query=user_input
            )
            
            diagnosis = future.result(timeout=Config.LLM_TIMEOUT)
            
            # FIXED: Accept API responses more liberally
            if not diagnosis or len(diagnosis.strip()) < 50:
                logger.warning("LLM returned insufficient response, using fallback")
                return generate_fallback_diagnosis(symptoms, state, user_input, retrieved_context)
            
            # Track success
            metrics = get_metrics_tracker()
            if metrics:
                metrics.track_llm_generation(state.session_id, True, Config.LLM_TIMEOUT)
            
            logger.info(f"✓ LLM diagnosis generated: {len(diagnosis)} characters")
            return diagnosis
    
    except FutureTimeoutError:
        logger.warning(f"LLM diagnosis timeout after {Config.LLM_TIMEOUT}s")
        return generate_fallback_diagnosis(symptoms, state, user_input, retrieved_context)
    
    except Exception as e:
        logger.error(f"Diagnosis generation error: {e}", exc_info=True)
        
        # Track failure
        metrics = get_metrics_tracker()
        if metrics:
            metrics.track_llm_generation(state.session_id, False, 0)
        
        return generate_fallback_diagnosis(symptoms, state, user_input, retrieved_context)

def generate_fallback_diagnosis(symptoms: List[str], state: ConversationState,
                               user_input: str, retrieved_context: List[str] = None) -> str:
    """Enhanced fallback diagnosis using medical knowledge base"""
    all_symptoms = state.get_all_symptoms_string()
    severity = MedicalKnowledgeBase.assess_severity(symptoms)
    
    # Build context-aware assessment
    context_section = ""
    if retrieved_context:
        context_section = "\n\n**📚 Medical Context:**\n"
        for i, doc in enumerate(retrieved_context[:2], 1):
            summary = doc[:180] + "..." if len(doc) > 180 else doc
            context_section += f"{i}. {summary}\n"
    
    # Get specific advice for each symptom
    advice_sections = []
    for symptom in symptoms[:3]:  # Limit to top 3 symptoms
        advice_data = MedicalKnowledgeBase.get_advice_for_symptom(symptom)
        advice_sections.append(advice_data["advice"])
    
    advice_text = "\n\n".join(advice_sections) if advice_sections else ""
    
    response = f"""**{severity['emoji']} Medical Assessment**

**Your symptoms:** {all_symptoms}

**Current concern:** {user_input}

**Assessment:**
{advice_text}
{context_section}

**Severity Level:** {severity['level']}

{severity['description']}

**Recommended Actions:**
1. {severity['immediate_action']}
2. {severity['consultation_advice']}
3. Monitor your symptoms and note any changes

**⚠️ Warning Signs to Watch For:**
{severity['warning_signs']}

**Important:** This is preliminary information only. Please consult a healthcare provider for proper medical evaluation."""

    # Add hospital recommendations if needed
    if severity['level'] in ["High - Needs attention", "Emergency"]:
        hospital_info = get_hospital_recommendations(severity['level'])
        response = f"{response}\n\n{hospital_info}"
        state.hospital_recommended = True

    return response

@timer
@safe_execute(default_return="")
def generate_followup_questions(state: ConversationState, diagnosis: str,
                               user_input: str) -> str:
    """Generate contextual follow-up questions"""
    agent = get_llm_agent()
    
    if agent:
        try:
            questions = agent.generate_followup(
                symptoms=list(dict.fromkeys(e.symptom for e in state.symptoms_history)),
                conversation_history=state.get_context_string(),
                last_assessment=diagnosis[:500],
                user_query=user_input
            )
            
            if questions and len(questions) > 50 and questions.count('?') >= 3:
                return questions
        
        except Exception as e:
            logger.error(f"Follow-up generation error: {e}")
    
    # Fallback questions
    return generate_fallback_questions(state, user_input)

def generate_fallback_questions(state: ConversationState, recent_query: str = "") -> str:
    """Generate intelligent fallback questions"""
    questions = ["**❓ Follow-up Questions:**\n"]
    
    # Extract discussed topics
    discussed = set()
    for turn in state.conversation_turns:
        discussed.update(turn.user_input.lower().split())
    
    # Context-aware questions
    context_questions = []
    
    if not any(w in discussed for w in ['days', 'hours', 'weeks', 'started', 'began', 'when']):
        context_questions.append("When exactly did these symptoms start, and how have they changed?")
    
    if not any(w in discussed for w in ['worse', 'better', 'improving', 'worsening']):
        context_questions.append("Are your symptoms getting better, worse, or staying the same?")
    
    if not any(w in discussed for w in ['trigger', 'cause', 'after', 'during']):
        context_questions.append("Have you noticed anything that triggers or relieves your symptoms?")
    
    if not any(w in discussed for w in ['medication', 'medical', 'condition', 'history', 'taking']):
        context_questions.append("Do you have any medical conditions or take medications regularly?")
    
    if not any(w in discussed for w in ['pain', 'scale', 'rate', 'level']):
        context_questions.append("On a scale of 1-10, how would you rate your discomfort?")
    
    # Select best questions
    selected = context_questions[:4] if context_questions else [
        "Can you describe how your symptoms have progressed?",
        "Are there activities that make your symptoms better or worse?",
        "Have you tried any treatments for these symptoms?"
    ]
    
    questions.extend([f"{i+1}. {q}" for i, q in enumerate(selected)])
    return "\n".join(questions)

@safe_execute(default_return=None)
def synthesize_speech(text: str) -> Optional[str]:
    """Generate TTS audio with English diagnosis and Tamil summary"""
    tts = get_tts()
    if not tts:
        logger.warning("TTS not available, returning silent audio")
        return ensure_silent_wav()
    
    try:
        # Prepare English text for audio
        audio_text_en = prepare_text_for_audio(text)
        
        if not audio_text_en or len(audio_text_en.strip()) < 10:
            logger.warning("Text too short for TTS, returning silent audio")
            return ensure_silent_wav()
        
        # Generate Tamil summary
        tamil_summary = generate_tamil_summary(text)
        logger.info(f"Tamil summary: {tamil_summary}")
        
        # Create file paths
        timestamp = int(time.time())
        pid = os.getpid()
        en_audio_path = os.path.join(Config.TEMP_DIR, f"response_en_{timestamp}_{pid}.wav")
        ta_audio_path = os.path.join(Config.TEMP_DIR, f"response_ta_{timestamp}_{pid}.wav")
        final_audio_path = os.path.join(Config.TEMP_DIR, f"response_{timestamp}_{pid}.wav")
        
        logger.info(f"Generating English TTS: {len(audio_text_en)} characters")
        
        # Generate English audio
        with timeout_context(Config.TTS_TIMEOUT) as executor:
            future_en = executor.submit(tts.synthesize_speech, audio_text_en, en_audio_path, online=True, lang="en")
            result_en = future_en.result(timeout=Config.TTS_TIMEOUT)
            
            if not result_en or not os.path.exists(result_en) or os.path.getsize(result_en) < 1000:
                logger.warning("English TTS generation failed")
                return ensure_silent_wav()
        
        logger.info(f"Generating Tamil summary TTS: {len(tamil_summary)} characters")
        
        # Generate Tamil audio
        with timeout_context(Config.TTS_TIMEOUT) as executor:
            future_ta = executor.submit(tts.synthesize_speech, tamil_summary, ta_audio_path, online=True, lang="ta")
            result_ta = future_ta.result(timeout=Config.TTS_TIMEOUT)
            
            if not result_ta or not os.path.exists(result_ta) or os.path.getsize(result_ta) < 1000:
                logger.warning("Tamil TTS generation failed, using English only")
                return result_en
        
        # Combine English + Tamil audio files
        logger.info("Combining English and Tamil audio...")
        try:
            from pydub import AudioSegment
            
            audio_en = AudioSegment.from_wav(result_en)
            audio_ta = AudioSegment.from_wav(result_ta)
            
            # Add 500ms silence between English and Tamil
            silence = AudioSegment.silent(duration=500)
            
            combined = audio_en + silence + audio_ta
            combined.export(final_audio_path, format="wav")
            
            if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 1000:
                logger.info(f"✓ Combined audio generated: {final_audio_path} ({os.path.getsize(final_audio_path)} bytes)")
                
                # Clean up temporary files
                try:
                    os.remove(result_en)
                    os.remove(result_ta)
                except:
                    pass
                
                return final_audio_path
            else:
                logger.warning("Combined audio file invalid, using English only")
                return result_en
                
        except Exception as e:
            logger.warning(f"Audio combination failed: {e}, using English only")
            return result_en
    
    except FutureTimeoutError:
        logger.warning(f"TTS timeout after {Config.TTS_TIMEOUT}s")
    except Exception as e:
        logger.error(f"TTS generation error: {e}", exc_info=True)
    
    return ensure_silent_wav()

def prepare_text_for_audio(text: str) -> str:
    """Clean text for TTS output"""
    # Remove questions section
    for marker in ["Follow-up questions", "**Follow-up", "❓", "\n1.", "\n2."]:
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos].strip()
            break
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[🚨⚠️📚❓⚕️ℹ️�💊]', '', text)
    
    # Limit length
    if len(text) > 800:
        text = text[:800] + "..."
    
    return text.strip()

def generate_tamil_summary(diagnosis: str) -> str:
    """Generate a Tamil summary of the diagnosis (used for audio)"""
    # Extract key information from diagnosis
    diagnosis_lower = diagnosis.lower()
    
    # Determine condition type
    condition = ""
    if "appendicitis" in diagnosis_lower or "abdominal pain" in diagnosis_lower:
        condition = "வயிற்று வலி மற்றும் சாத்தியமான அப்பென்டிசிஸ்"
    elif "chest pain" in diagnosis_lower or "heart" in diagnosis_lower:
        condition = "மார்பு வலி"
    elif "fever" in diagnosis_lower and "cough" in diagnosis_lower:
        condition = "காய்ச்சல் மற்றும் இருமல்"
    elif "fever" in diagnosis_lower:
        condition = "காய்ச்சல்"
    elif "cough" in diagnosis_lower:
        condition = "இருமல்"
    elif "headache" in diagnosis_lower:
        condition = "தலைவலி"
    elif "leg pain" in diagnosis_lower:
        condition = "கால் வலி"
    else:
        condition = "உங்கள் அறிகுறிகள்"
    
    # Determine severity level
    severity = ""
    action = ""
    if any(w in diagnosis_lower for w in ["emergency", "immediate", "urgent", "call 108", "call 911", "er now"]):
        severity = "மிக அவசர நிலை"
        action = "உடனடியாக 108 அழைக்கவும் அல்லது அவசர சிகிச்சை பிரிவுக்கு செல்லவும்"
    elif any(w in diagnosis_lower for w in ["see a doctor", "consult", "medical attention"]):
        severity = "மருத்துவ கவனிப்பு தேவை"
        action = "விரைவில் மருத்துவரை சந்திக்கவும்"
    elif any(w in diagnosis_lower for w in ["monitor", "rest", "home care"]):
        severity = "கண்காணிப்பு தேவை"
        action = "வீட்டில் ஓய்வு எடுத்து அறிகுறிகளை கண்காணிக்கவும்"
    else:
        severity = "மதிப்பீடு"
        action = "மருத்துவ ஆலோசனை பெறவும்"
    
    # Build Tamil summary
    tamil_summary = f"""சுருக்கம்: {condition} பற்றிய மதிப்பீடு. 
நிலை: {severity}. 
பரிந்துரை: {action}. 
கூடுதல் தகவலுக்கு ஆங்கில விளக்கத்தை படிக்கவும்."""
    
    return tamil_summary

def generate_detailed_tamil_response(diagnosis: str, followup: str) -> str:
    """Generate detailed Tamil translation of diagnosis"""
    diagnosis_lower = diagnosis.lower()
    
    # Build Tamil response based on condition
    tamil_text = "=== தமிழ் பதில் (Tamil Response) ===\n\n"
    
    # Chest pain - மார்பு வலி
    if "chest pain" in diagnosis_lower:
        tamil_text += """🚨 அவசர மருத்துவ மதிப்பீடு - மார்பு வலி

மார்பு வலிக்கு உடனடி மருத்துவ பரிசோதனை தேவை.

சாத்தியமான காரணங்கள்:

1. இதய தொடர்பான (மிக அவசரம்)
   • மாரடைப்பு
   • இதயத்திற்கு குறைந்த இரத்த ஓட்டம்
   
2. தசை அல்லது எலும்பு தொடர்பான
   • தசை இழுப்பு
   • விலா எலும்பு அழற்சி
   
3. சுவாச தொடர்பான
   • நிமோனியா
   • நுரையீரல் அழற்சி

🚨 உடனடி நடவடிக்கைகள்:

1. கடுமையான வலி இருந்தால் உடனடியாக 108 அழைக்கவும்
2. நீங்களே வாகனம் ஓட்ட வேண்டாம்
3. வசதியான நிலையில் உட்காருங்கள் அல்லது படுத்துக் கொள்ளுங்கள்
4. அஸ்பிரின் மாத்திரை இருந்தால் மெல்லுங்கள் (ஒவ்வாமை இல்லை என்றால்)
5. அமைதியாக இருந்து மெதுவாக சுவாசிக்கவும்

⚠️ இந்த அறிகுறிகள் இருந்தால் உடனடியாக 108 அழைக்கவும்:
• கடுமையான, நசுக்கும் மார்பு வலி
• கை, தாடை அல்லது முதுகுக்கு பரவும் வலி
• மூச்சுத் திணறல்
• அதிக வியர்வை
• குமட்டல் அல்லது வாந்தி
• தலைசுற்றல் அல்லது மயக்கம்

எப்போது உதவி பெறுவது:
இப்போதே - உடனடியாக அருகிலுள்ள மருத்துவமனைக்கு செல்லுங்கள் அல்லது 108 அழைக்கவும்

முக்கிய குறிப்பு:
மார்பு வலி உயிருக்கு ஆபத்தானதாக இருக்கலாம். அவசர மருத்துவ சிகிச்சையை தாமதப்படுத்த வேண்டாம்."""
    
    # Abdominal pain - வயிற்று வலி
    elif "abdominal pain" in diagnosis_lower or "appendicitis" in diagnosis_lower:
        tamil_text += """⚠️ அவசரம் - வயிற்று வலிக்கு உடனடி பரிசோதனை தேவை

கீழ் வலது பக்க கடுமையான வயிற்று வலிக்கு உடனடி மருத்துவ மதிப்பீடு தேவை.

🚨 சாத்தியமான அவசர நிலை - அப்பென்டிசிடிஸ்

அப்பென்டிசிடிஸ் அறிகுறிகள்:

✓ கீழ் வலது வயிற்றில் கடுமையான வலி
✓ குமட்டல் மற்றும் வாந்தி
✓ பசியின்மை
✓ லேசான காய்ச்சல்
✓ தொப்புளைச் சுற்றி தொடங்கி வலது பக்கம் நகரும் வலி
✓ அசைவு, இருமல் அல்லது அழுத்தத்தால் வலி அதிகரிக்கும்

🚨 உடனடி நடவடிக்கைகள்:

1. எதுவும் சாப்பிடவோ குடிக்கவோ வேண்டாம்
2. வலி நிவாரண மருந்து எடுக்க வேண்டாம்
3. வயிற்றில் சூடு வைக்க வேண்டாம்
4. உடனடியாக மருத்துவமனைக்கு செல்லுங்கள் அல்லது 108 அழைக்கவும்
5. நீங்களே வாகனம் ஓட்ட வேண்டாம்

⚠️ கடுமையான எச்சரிக்கை அறிகுறிகள்:
• திடீரென வலி குறைந்து பின்னர் மோசமாகுதல்
• அதிக காய்ச்சல் (102°F / 39°C க்கு மேல்)
• வேகமான இதயத் துடிப்பு
• கடுமையான வயிற்று கடினத்தன்மை
• அதிர்ச்சி அறிகுறிகள் (வெளிர், வியர்வை, பலவீனமான நாடித் துடிப்பு)

எப்போது உதவி பெறுவது:
🚨 இப்போதே அவசர சிகிச்சை பிரிவுக்கு செல்லுங்கள் - இது மருத்துவ அவசரநிலை

நேர அட்டவணை:
அப்பென்டிசிடிஸ் அறிகுறி தொடங்கி 24-72 மணி நேரத்திற்குள் வெடிக்கலாம், இது உயிருக்கு ஆபத்தான நோய்த்தொற்றுக்கு வழிவகுக்கும்.

முக்கிய குறிப்பு:
குமட்டல் மற்றும் பசியின்மையுடன் கூடிய கீழ் வலது வயிற்று வலி மருத்துவ அவசர நிலை. உடனடியாக அருகிலுள்ள மருத்துவமனைக்கு செல்லுங்கள்."""
    
    # Fever and cough - காய்ச்சல் மற்றும் இருமல்
    elif "fever" in diagnosis_lower and "cough" in diagnosis_lower:
        tamil_text += """🤒 காய்ச்சல் மற்றும் இருமல் மதிப்பீடு

உங்கள் அறிகுறிகள்:
• காய்ச்சல்
• இருமல்
• சோர்வு

சாத்தியமான காரணங்கள்:
1. வைரஸ் தொற்று (சளி, காய்ச்சல்)
2. சுவாசக்குழாய் அழற்சி
3. நிமோனியா
4. COVID-19

பரிந்துரைகள்:

வீட்டில் பராமரிப்பு:
• நிறைய தண்ணீர் குடியுங்கள்
• ஓய்வு எடுங்கள்
• பரசிடமால் காய்ச்சலுக்கு எடுக்கலாம்
• சூடான திரவங்கள் குடியுங்கள்

மருத்துவரை பார்க்க வேண்டிய நேரம்:
• 3 நாட்களுக்கு மேல் காய்ச்சல் தொடர்ந்தால்
• மூச்சுத் திணறல் இருந்தால்
• மார்பு வலி ஏற்பட்டால்
• அறிகுறிகள் மோசமாகினால்

⚠️ உடனடியாக மருத்துவமனைக்கு செல்ல வேண்டிய அறிகுறிகள்:
• கடுமையான மூச்சுத் திணறல்
• நீல நிற உதடுகள் அல்லது முகம்
• குழப்பம் அல்லது மயக்கம்
• மார்பு வலி
• அதிக காய்ச்சல் (104°F / 40°C க்கு மேல்)"""
    
    # General response
    else:
        tamil_text += generate_tamil_summary(diagnosis)
        tamil_text += "\n\nவிரிவான ஆங்கில விளக்கத்தை 'English' தாவலில் பார்க்கவும்."
    
    # Add follow-up questions in Tamil if present
    if followup and "Follow-up" in followup:
        tamil_text += "\n\n📋 கூடுதல் கேள்விகள்:\n"
        tamil_text += "மேலும் துல்லியமான மதிப்பீட்டிற்கு, தயவுசெய்து இந்தக் கேள்விகளுக்கு பதிலளிக்கவும்.\n"
        tamil_text += "(ஆங்கிலம் அல்லது தமிழில் பதிலளிக்கலாம்)"
    
    return tamil_text

def get_hospital_recommendations(severity: str) -> str:
    """Get nearby hospital recommendations"""
    try:
        locator = get_hospital_locator()
        if not locator:
            return get_fallback_emergency_info()
        
        hospitals = locator.find_nearby_hospitals(
            emergency_only=(severity == "Emergency"),
            max_results=5
        )
        
        return locator.format_hospital_recommendations(hospitals, severity)
    
    except Exception as e:
        logger.error(f"Hospital locator error: {e}")
        return get_fallback_emergency_info()

def get_fallback_emergency_info() -> str:
    """Fallback emergency information"""
    return """**🚨 Emergency Medical Services:**

**📞 Call 108 immediately for ambulance service**

**Other Emergency Numbers:**
- Emergency: 112
- Medical Helpline: 104
- Women's Helpline: 1091
- Senior Citizens Helpline: 14567

**� What to do:**
1. Stay calm and call for help immediately
2. Don't drive yourself if experiencing severe symptoms
3. Have someone stay with you
4. Keep your phone charged and nearby
5. Gather any medications you're currently taking
6. Note exact symptoms and when they started

**⚠️ Go to ER immediately if you experience:**
- Chest pain or pressure
- Difficulty breathing
- Severe bleeding
- Loss of consciousness
- Stroke symptoms (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 108)"""

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

@timer
def process_patient_input(message: str, audio_input, state: ConversationState) -> Tuple[str, str, str]:
    """Main processing pipeline with enhanced error handling"""
    
    # Sanitize and process input
    input_text = sanitize_input(message or "")
    
    # Process audio if provided
    if audio_input is not None:
        transcriber = get_audio_transcriber()
        if transcriber:
            try:
                logger.info("Processing audio input...")
                
                # Handle different audio input types
                audio_bytes = None
                if hasattr(audio_input, 'read'):
                    audio_bytes = audio_input.read()
                elif isinstance(audio_input, str) and os.path.exists(audio_input):
                    with open(audio_input, 'rb') as f:
                        audio_bytes = f.read()
                elif isinstance(audio_input, bytes):
                    audio_bytes = audio_input
                else:
                    logger.warning(f"Unknown audio input type: {type(audio_input)}")
                
                if audio_bytes:
                    transcribed = transcriber.transcribe_audio(audio_bytes)
                    if transcribed and transcribed.strip():
                        input_text = f"{input_text} {transcribed}".strip() if input_text else transcribed
                        logger.info(f"✓ Audio transcribed: {transcribed[:100]}...")
            
            except Exception as e:
                logger.error(f"Audio transcription error: {e}", exc_info=True)
    
    # Validate input
    if not input_text.strip():
        return (
            "Please describe your symptoms in text or use voice input.",
            "",
            ensure_silent_wav()
        )
    
    logger.info(f"Processing input: {input_text[:100]}...")
    
    # Parse symptoms
    parser = get_symptom_parser()
    symptoms = list(parser.parse_symptoms(input_text)) if parser else []
    
    logger.info(f"Parsed symptoms: {symptoms}")
    
    # Check if this is a follow-up response (no new symptoms but conversation exists)
    input_lower = input_text.lower().strip()
    is_followup_response = (
        len(symptoms) == 0 and 
        len(state.conversation_turns) > 0 and 
        any(word in input_lower for word in [
            "worse", "better", "improving", "worsening", "same", "still", 
            "yes", "no", "started", "hours", "days", "ago", "pain is",
            "getting", "feeling", "now"
        ])
    )
    
    logger.info(f"Follow-up detection: symptoms={len(symptoms)}, has_history={len(state.conversation_turns) > 0}, is_followup={is_followup_response}")
    
    # Handle follow-up responses differently
    if is_followup_response:
        last_diagnosis = state.conversation_turns[-1].bot_response if state.conversation_turns else ""
        
        # Check if it's getting worse (emergency escalation)
        if any(word in input_lower for word in ["worse", "worsening", "increasing", "unbearable", "severe"]):
            if "appendicitis" in last_diagnosis.lower() or "emergency" in last_diagnosis.lower():
                response = """🚨🚨 CRITICAL - SYMPTOMS WORSENING 🚨🚨

You reported that your symptoms are GETTING WORSE. This is a critical sign with potential appendicitis.

⚠️ DO NOT WAIT ANY LONGER ⚠️

IMMEDIATE ACTION REQUIRED:
1. CALL 108 (Ambulance) RIGHT NOW if not already at hospital
2. If someone is with you, have them drive you to ER IMMEDIATELY
3. Do NOT eat, drink, or take medication
4. Lie still and avoid movement if possible

WORSENING SYMPTOMS MAY INDICATE:
• Appendix is about to rupture or has ruptured
• Infection spreading (peritonitis)
• Life-threatening emergency developing

TIME IS CRITICAL - Every minute counts!

This is beyond the scope of a symptom checker. You need emergency medical care NOW."""
                
                state.add_turn(input_text, response, "CRITICAL EMERGENCY", symptoms)
                audio_file = synthesize_speech(response)
                return response, "", audio_file
            else:
                response = f"""Based on your previous symptoms, and now reporting they are getting worse:

⚠️ ESCALATING TO URGENT CARE NEEDED

Since your symptoms are worsening rather than improving, you should:
1. Seek medical attention TODAY
2. Go to urgent care or ER if symptoms are severe
3. Call your doctor immediately
4. Do not wait to see if it improves

Worsening symptoms require professional medical evaluation."""
                
                state.add_turn(input_text, response, "High", symptoms)
                audio_file = synthesize_speech(response)
                return response, "", audio_file
        
        # Improving symptoms
        elif any(word in input_lower for word in ["better", "improving", "improving"]):
            response = """That's good to hear that symptoms are improving.

Continue to:
- Monitor your symptoms
- Rest and stay hydrated
- Follow previous recommendations

However, seek medical care if:
- Symptoms return or worsen
- New symptoms develop
- You have concerns about your condition

Stay vigilant and take care of yourself."""
            
            state.add_turn(input_text, response, "Monitoring", symptoms)
            audio_file = synthesize_speech(response)
            return response, "", audio_file
        
        # General follow-up acknowledgment
        else:
            response = f"""Thank you for the additional information: "{input_text}"

Based on your previous assessment, please continue to follow the recommendations provided.

If you have new or different symptoms, please describe them and I'll provide a new assessment.

For emergency symptoms, seek immediate medical care."""
            
            state.add_turn(input_text, response, "Follow-up", symptoms)
            audio_file = synthesize_speech(response)
            return response, "", audio_file
    
    # Add symptoms to state
    for symptom in symptoms:
        state.add_symptom(symptom, context=input_text)
    
    # Track interaction
    metrics = get_metrics_tracker()
    if metrics:
        metrics.track_interaction(state.session_id, symptoms)
    
    # Retrieve medical context
    retrieved_docs = retrieve_medical_context(symptoms)
    
    # Generate diagnosis
    diagnosis = generate_diagnosis(symptoms, retrieved_docs, state, input_text)
    
    # Extract severity
    severity = extract_severity_from_diagnosis(diagnosis)
    
    # Generate follow-up questions
    followup = generate_followup_questions(state, diagnosis, input_text)
    
    # Combine response
    full_response = f"{diagnosis}\n\n{followup}"
    
    # Add conversation turn
    state.add_turn(input_text, full_response, severity, symptoms)
    
    # Generate audio
    audio_file = synthesize_speech(diagnosis)  # Only diagnosis, not follow-up questions
    
    return diagnosis, followup, audio_file

def extract_severity_from_diagnosis(diagnosis: str) -> str:
    """Extract severity level from diagnosis text"""
    diagnosis_lower = diagnosis.lower()
    
    if any(w in diagnosis_lower for w in ["emergency", "immediate", "urgent", "911", "108"]):
        return "Emergency"
    elif any(w in diagnosis_lower for w in ["high", "severe", "serious"]):
        return "High - Needs attention"
    elif any(w in diagnosis_lower for w in ["moderate", "concerning"]):
        return "Moderate - Should see a doctor"
    else:
        return "Mild - Monitor symptoms"

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def chat_interface(message: str, history: List[Dict], tamil_text: str, audio_input, 
                  session_state: Optional[str]) -> Tuple[List[Dict], str, str, str, str]:
    """
    Main chat interface handler
    Returns: (history, tamil_output, empty_message, audio_file, session_id)
    """
    # Handle empty input
    if not message and not audio_input:
        return history or [], tamil_text or "", "", ensure_silent_wav(), session_state
    
    # Get or create session
    state = session_manager.get_or_create(session_state)
    
    try:
        # Process input
        diagnosis, followup, audio_file = process_patient_input(message, audio_input, state)
        
        # Generate detailed Tamil response
        tamil_response = generate_detailed_tamil_response(diagnosis, followup)
        
        # Format English response
        user_msg = message if message else "[🎤 Voice Input]"
        bot_response = f"{diagnosis}\n\n{followup}"
        
        # Update history
        if history is None:
            history = []
        
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_response})
        
        return history, tamil_response, "", audio_file, state.session_id
    
    except Exception as e:
        logger.error(f"Chat interface error: {e}", exc_info=True)
        
        error_msg = "I apologize, but I encountered an error processing your request. Please try again or rephrase your symptoms."
        
        if history is None:
            history = []
        
        history.append({"role": "assistant", "content": error_msg})
        
        tamil_error = "மன்னிக்கவும், உங்கள் கோரிக்கையை செயலாக்குவதில் பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்."
        
        return history, tamil_error, "", ensure_silent_wav(), state.session_id

def clear_chat() -> Tuple[List, str, str, str, None]:
    """Clear chat and start new session"""
    # Reset LLM
    llm = get_llm_agent()
    if llm:
        llm.reset_conversation()
    
    logger.info("Chat cleared, starting new session")
    return [], "", "", ensure_silent_wav(), None

def export_conversation(session_state: str) -> Optional[str]:
    """Export conversation history as JSON"""
    state = session_manager.get(session_state)
    if not state:
        return None
    
    try:
        export_data = {
            "export_date": datetime.now().isoformat(),
            "session_data": state.to_dict()
        }
        
        export_path = os.path.join(
            Config.TEMP_DIR,
            f"conversation_{state.session_id}.json"
        )
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_path
    
    except Exception as e:
        logger.error(f"Export error: {e}")
        return None

# ============================================================================
# GRADIO UI
# ============================================================================

def create_gradio_interface():
    """Create simple and clean Gradio interface"""
    
    with gr.Blocks(
        title="SympCheck - Medical Symptom Checker",
        theme=gr.themes.Default(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header-box {
            background: #2563eb;
            padding: 20px;
            border-radius: 8px;
            color: white;
            margin-bottom: 20px;
        }
        .emergency-box {
            background: #dc2626;
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin-bottom: 15px;
        }
        .info-box {
            background: #f3f4f6;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        """
    ) as demo:
        
        # Simple Header
        gr.HTML("""
        <div class="header-box">
            <h2 style="margin: 0;">SympCheck Medical Assistant</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Describe your symptoms and get medical guidance</p>
        </div>
        """)
        
        session_state = gr.State(value=None)
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=2):
                # Tabs for English and Tamil
                with gr.Tabs():
                    with gr.Tab("English"):
                        chatbot_en = gr.Chatbot(
                            label="English Conversation",
                            height=450,
                            type="messages",
                            show_label=False
                        )
                    
                    with gr.Tab("தமிழ் (Tamil)"):
                        tamil_output = gr.Textbox(
                            label="Tamil Response",
                            lines=15,
                            max_lines=20,
                            interactive=False,
                            show_label=False
                        )
                
                # Input section
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Describe your symptoms here...",
                        label="Your symptoms",
                        lines=2,
                        scale=4
                    )
                    
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        label="Voice input",
                        scale=1,
                        type="filepath"
                    )
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                
                audio_output = gr.Audio(
                    label="Audio response (English + Tamil)",
                    type="filepath",
                    autoplay=True,
                    interactive=False
                )            # Sidebar
            with gr.Column(scale=1):
                # Emergency alert
                gr.HTML("""
                <div class="emergency-box">
                    <h3 style="margin: 0 0 10px 0;">Emergency</h3>
                    <p style="margin: 0; font-size: 24px; font-weight: bold;">☎️ 108</p>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Call for severe symptoms</p>
                </div>
                """)
                
                # Helplines
                gr.HTML("""
                <div class="info-box">
                    <h4 style="margin: 0 0 10px 0;">Helplines</h4>
                    <p style="margin: 3px 0; font-size: 14px;">Emergency: 112</p>
                    <p style="margin: 3px 0; font-size: 14px;">Medical: 104</p>
                    <p style="margin: 3px 0; font-size: 14px;">Mental Health: 9152987821</p>
                </div>
                """)
                
                # Disclaimer
                gr.HTML("""
                <div class="info-box">
                    <h4 style="margin: 0 0 10px 0;">Disclaimer</h4>
                    <p style="margin: 0; font-size: 13px; line-height: 1.4;">
                    This is for informational purposes only. 
                    Not a substitute for professional medical advice. 
                    Consult a doctor for proper diagnosis.
                    </p>
                </div>
                """)
                
                # Examples
                gr.Examples(
                    examples=[
                        ["I have chest pain and shortness of breath"],
                        ["High fever with severe headache for 3 days"],
                        ["Sharp pain in lower right abdomen with nausea"],
                        ["Persistent cough and fatigue for a week"]
                    ],
                    inputs=msg_input
                )
        
        # Event handlers
        msg_input.submit(
            chat_interface,
            inputs=[msg_input, chatbot_en, tamil_output, audio_input, session_state],
            outputs=[chatbot_en, tamil_output, msg_input, audio_output, session_state]
        )
        
        send_btn.click(
            chat_interface,
            inputs=[msg_input, chatbot_en, tamil_output, audio_input, session_state],
            outputs=[chatbot_en, tamil_output, msg_input, audio_output, session_state]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot_en, tamil_output, msg_input, audio_output, session_state]
        )
    
    return demo

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def find_available_port(start_port: int = 7860, max_attempts: int = 20) -> int:
    """Find an available port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start_port

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("� AI Medical Symptom Checker - Enhanced Version")
    logger.info("=" * 60)
    
    # Create interface
    demo = create_gradio_interface()
    
    # Get port
    port = int(os.getenv("GRADIO_SERVER_PORT", find_available_port()))
    
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"🌐 Access URL: http://localhost:{port}")
    logger.info(f"📊 Active sessions will be managed automatically")
    logger.info(f"💾 Temp directory: {Config.TEMP_DIR}")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)
    
    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_api=False,
        share=False,
        show_error=True,
        favicon_path=None
    )