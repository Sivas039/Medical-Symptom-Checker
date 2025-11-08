# 🏥 SympCheck Plus - AI-Powered Healthcare Assistant

**⚡ Latest Update**: All optimizations implemented! Single entry point, <10 second response time, universal symptom support, and visible AI reasoning.

SympCheck Plus is an intelligent medical assistant that provides comprehensive symptom assessment, contextual conversations, and hospital recommendations. Built with advanced AI and designed for healthcare support.

## ✨ Key Features (Optimized)

- **⚡ Ultra-Fast Diagnosis**: Response within 10 seconds
- **🤖 Universal Symptom Analysis**: Handles ANY symptom type dynamically
- **🧠 Agentic AI (ReAct)**: Shows reasoning process step-by-step
- **🏥 Interactive Hospital Locator**: Real-time map with nearby hospitals
- **🎙️ Voice Input Support**: Audio transcription with Whisper
- **🔊 Text-to-Speech**: Audio responses for accessibility  
- **📊 Medical Knowledge Base**: RAG-powered information retrieval
- **⚡ Lazy Loading**: Fast startup, deferred AI model loading
- **🛡️ Timeout Protection**: No hanging requests, graceful degradation

## 🚀 Quick Start (NEW)

```bash
# Single command to run the app:
python gradio_app.py

# Opens at: http://localhost:7860
```

**That's it!** No need for `run_app.py`, `run.py`, or `quick_start.py` anymore.

## 📊 Project Structure (Reorganized)

```
SympCheck/
├── gradio_app.py              ← ONLY entry point
├── requirements.txt
├── .env                       ← Set API keys here
│
├── modules/                   ← AI Components
│   ├── llm_agent.py          ✓ Streaming + timeouts
│   ├── medical_agent.py      ✓ ReAct reasoning engine
│   ├── rag_retriever.py      ✓ Medical knowledge base
│   ├── symptom_parser.py     ✓ Symptom extraction
│   ├── hospital_locator.py   ✓ Map generation
│   └── tts_output.py         ✓ Audio synthesis
│
├── prompts/                   ← AI Templates
│   ├── diagnosis_prompt.txt   (enhanced for any symptom)
│   └── followup_prompt.txt
│
├── data/                      ← Medical Data
├── metrics/                   ← Analytics
├── docs/                      ← Documentation
├── tests/                     ← Unit tests
├── legacy/                    ← Archived launchers
│   ├── run_app.py
│   ├── run.py
│   ├── quick_start.py
│   └── start_app.py
│
└── README.md                  (You are here)
```

## 🎯 Performance Improvements

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Response Time | 15-20s | 7-9s | ✅ |
| Symptom Types | ~30 hardcoded | Unlimited | ✅ |
| AI Reasoning | Hidden | Visible | ✅ |
| Hospital Display | Text | Interactive Map | ✅ |
| Startup Time | Slow | Fast | ✅ |
- Optional: Google Maps API key for hospital locator

### Installation Steps

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd SympCheck-AI-Healthcare
```

2. **Create virtual environment:**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Download spaCy model (recommended):**
```bash
python -m spacy download en_core_web_sm
```

6. **Run the application:**
```bash
python gradio_app.py
```

7. **Access the interface:**
   - Open your browser to `http://localhost:7860`
   - Start describing your symptoms!

## 📋 Environment Configuration

Create a `.env` file with the following variables:

```env
# LLM Configuration (Optional - for advanced features)
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=openai/gpt-4o-mini

# Google Maps API (Optional - for hospital locator)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# Logging Level
LLM_AGENT_LOGLEVEL=INFO
```

## 🔧 Advanced Setup

### Medical Database Setup

To use the RAG (Retrieval-Augmented Generation) features:

1. **Download medical dataset:**
   - Get MedQuAD dataset from: https://github.com/abachaa/MedQuAD
   - Place CSV file at `data/medquad.csv`

2. **Build the database:**
```bash
python build_database.py
```

This creates FAISS indices for fast medical information retrieval.

### Audio Features Setup

For voice input/output features:

1. **Install audio dependencies:**
```bash
# On Ubuntu/Debian:
sudo apt-get install ffmpeg

# On macOS:
brew install ffmpeg

# On Windows:
# Download ffmpeg and add to PATH
```

2. **Test audio functionality:**
```bash
python -c "from modules.audio_input import AudioTranscriber; AudioTranscriber()"
```

## 🎯 Usage Examples

### Basic Symptom Input
```
"I have a headache and fever for 2 days"
```

### Detailed Consultation
```
"I've been experiencing chest pain that gets worse when I breathe deeply. 
It started yesterday evening and I also feel short of breath."
```

### Follow-up Conversation
```
User: "I have stomach pain"
Assistant: [Provides assessment and questions]
User: "The pain is getting worse and I feel nauseous"
Assistant: [Updates assessment based on new information]
```

## 🏥 Hospital Locator

The system can automatically recommend nearby hospitals based on:
- Symptom severity
- Your location (IP-based or GPS)
- Emergency services availability
- Distance and travel time

Emergency numbers are provided for immediate situations.

## 🧠 AI Components

### Symptom Parser
- Uses spaCy NLP for advanced symptom extraction
- Regex fallback for reliability
- Handles negations and synonyms

### LLM Agent
- Supports NVIDIA API, OpenAI-compatible endpoints
- Conversation memory and context awareness
- Local fallback models available

### RAG Retriever
- FAISS vector database for medical knowledge
- Sentence transformer embeddings
- Fast similarity search

### Audio Processing
- Whisper for speech-to-text
- Multiple TTS options (Coqui, gTTS)
- Audio format conversion

## 📊 System Monitoring

The application includes built-in metrics tracking:
- Session statistics
- LLM call success rates
- Symptom detection accuracy
- User interaction patterns

View metrics through the admin interface (if enabled).

## 🔧 Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
pip install black flake8
black .
flake8 .
```

### Adding New Features
1. Create module in `modules/`
2. Add tests in `tests/`
3. Update documentation
4. Submit pull request

## ⚠️ Important Disclaimers

**MEDICAL DISCLAIMER**: This application is for informational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

**EMERGENCY SITUATIONS**: In case of medical emergencies, always call your local emergency services immediately (108 in India, 911 in US, etc.).

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the troubleshooting section

## 🙏 Acknowledgments

- MedQuAD dataset for medical knowledge
- OpenAI Whisper for audio processing
- Hugging Face for transformer models
- Google Maps for hospital location services
- Gradio for the web interface

---

**Built with ❤️ for healthcare accessibility**
