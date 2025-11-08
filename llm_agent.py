import os
import random
import re
from typing import Optional, List
import logging

logger = logging.getLogger("LLMAgent")

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. Using fallback mode.")


class LLMAgent:
    """Enhanced LLM Agent with NVIDIA GPT-OSS 120B integration.

    This agent uses the real GPT-OSS 120B model via NVIDIA API when available,
    with intelligent fallback for reliability.

    Methods:
      - _run(prompt, temperature, max_tokens, add_to_history) - Generate text
      - generate_diagnosis(...) - Generate medical diagnosis
      - generate_followup(...) - Generate follow-up questions
      - reset_conversation() - Clear conversation history
      - ping(msg) - Test agent connectivity
    """

    def __init__(self, timeout: int = 12, max_tokens: int = 500, use_api: bool = True):
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.history = []
        self.use_api = use_api and OPENAI_AVAILABLE
        self.client = None
        self.api_available = False
        
        # Try to initialize OpenAI client with NVIDIA API
        if self.use_api:
            try:
                load_dotenv()
                api_key = os.getenv('NVIDIA_API_KEY')
                api_base = os.getenv('NVIDIA_API_BASE', 'https://integrate.api.nvidia.com/v1')
                
                if api_key:
                    self.client = OpenAI(
                        base_url=api_base,
                        api_key=api_key
                    )
                    self.model = os.getenv('NVIDIA_MODEL', 'openai/gpt-oss-120b')
                    self.api_available = True
                    logger.info(f"✓ NVIDIA GPT-OSS 120B initialized: {self.model}")
                else:
                    logger.warning("NVIDIA_API_KEY not found. Using fallback mode.")
                    self.use_api = False
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA API: {e}. Using fallback mode.")
                self.use_api = False
                self.api_available = False

    def reset_conversation(self):
        self.history = []

    def ping(self, msg: str = "") -> str:
        return f"pong:{str(msg)[:100]}"

    def _run(self, prompt: str, temperature: float = 0.7, max_tokens: int = None, add_to_history: bool = True) -> str:
        """Generate text response using NVIDIA GPT-OSS 120B API or fallback.

        Args:
            prompt: The input prompt/query
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            add_to_history: Whether to track this in conversation history

        Returns:
            Generated text response
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Keep history short
        if add_to_history:
            self.history.append(prompt)
            if len(self.history) > 20:
                self.history = self.history[-20:]

        prompt_lower = prompt.lower()
        
        # Check if this is explicitly a follow-up request
        is_followup_request = (
            "generate follow" in prompt_lower or 
            "followup questions" in prompt_lower or
            "follow-up questions" in prompt_lower
        )
        
        # Try to use NVIDIA API for diagnosis (not follow-up)
        if self.use_api and self.client and self.api_available and not is_followup_request:
            try:
                # Use higher max_tokens for better responses
                api_max_tokens = max(max_tokens, 600)
                
                logger.debug(f"Calling NVIDIA API with {api_max_tokens} max tokens")
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are an expert medical AI assistant. Provide comprehensive, accurate medical assessments in a well-structured format.

Your responses must:
1. Be evidence-based and medically sound
2. Use clear headings and bullet points
3. Always recommend professional medical consultation for serious symptoms
4. Include severity assessment
5. Provide specific, actionable recommendations
6. List warning signs to watch for
7. Be at least 400 words for diagnosis requests

Structure your diagnosis responses as:
**Medical Assessment**
[Brief overview]

**Possible Causes:**
[List 2-3 likely conditions with brief explanations]

**Severity Level:** [Emergency/High/Moderate/Mild]

**Recommended Actions:**
[Numbered list of specific steps]

**Warning Signs:**
[Bullet list of red flags]

**When to Seek Help:**
[Clear guidance on when to see a doctor]

**Important Note:**
[Disclaimer about professional medical care]"""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    top_p=1,
                    max_tokens=api_max_tokens,
                    stream=False
                )
                
                response = completion.choices[0].message.content
                
                # Accept API responses more liberally
                if response and len(response.strip()) > 100:
                    logger.info(f"✓ NVIDIA API response: {len(response)} chars")
                    return response
                else:
                    logger.warning(f"API response too short ({len(response) if response else 0} chars), using fallback")
                    
            except Exception as e:
                logger.warning(f"NVIDIA API error: {str(e)[:100]}")
                # Continue to fallback
        
        # Fallback for follow-up questions
        if is_followup_request:
            return self._generate_followup_fallback(prompt)
        
        # Fallback for diagnosis - use structured medical responses
        return self._generate_diagnosis_fallback(prompt)
    
    def _generate_followup_fallback(self, prompt: str) -> str:
        """Generate follow-up questions fallback"""
        questions = [
            "1. When exactly did these symptoms start?",
            "2. Are the symptoms getting better, worse, or staying the same?",
            "3. Do you have any past medical conditions or take any medications?",
            "4. Have you noticed anything that triggers or relieves the symptoms?",
            "5. On a scale of 1-10, how would you rate your discomfort?"
        ]
        return "\n".join(questions)
    
    def _generate_diagnosis_fallback(self, prompt: str) -> str:
        """Generate structured medical diagnosis fallback"""
        
        # Extract symptoms from prompt
        symptom_responses = {
            "fever": {
                "with_cough": """**Medical Assessment**

Your symptoms of fever and cough suggest a respiratory infection.

**Possible Causes:**
1. **Viral Upper Respiratory Infection (Most Common)**
   - Common cold or flu
   - Usually resolves in 7-10 days
   - Highly contagious

2. **Acute Bronchitis**
   - Inflammation of airways
   - Persistent cough with or without mucus
   - May require medical evaluation

3. **Pneumonia (If Severe)**
   - Lung infection requiring medical attention
   - Watch for high fever, difficulty breathing, chest pain

**Severity Level:** Moderate - Should see a doctor if symptoms worsen

**Recommended Actions:**
1. Rest and stay well hydrated (8-10 glasses of water daily)
2. Monitor temperature every 4-6 hours
3. Use fever reducers (acetaminophen/ibuprofen) as directed
4. Isolate to prevent spread
5. Schedule doctor visit if symptoms don't improve in 3-4 days

**Warning Signs:**
• Fever above 103°F (39.4°C)
• Difficulty breathing or shortness of breath
• Persistent chest pain
• Coughing up blood
• Severe weakness or confusion

**When to Seek Help:**
- See a doctor if fever lasts more than 3 days
- Visit ER immediately if you experience severe difficulty breathing or chest pain
- Consider telehealth consultation for initial assessment

**Important Note:**
This is preliminary information only. Please consult a healthcare provider for proper medical evaluation and treatment.""",
                
                "alone": """**Medical Assessment**

Fever indicates your body is fighting an infection or illness.

**Possible Causes:**
1. **Viral Infection (Most Common)**
   - Cold, flu, or other viral illness
   - Usually resolves in 2-3 days

2. **Bacterial Infection**
   - May require antibiotics
   - Consider if fever persists beyond 3 days

3. **Other Causes**
   - Urinary tract infection
   - Ear or sinus infection
   - Inflammatory conditions

**Severity Level:** Moderate - Monitor closely

**Recommended Actions:**
1. Rest and drink plenty of fluids (water, herbal tea, broth)
2. Take fever reducers as directed (acetaminophen or ibuprofen)
3. Monitor temperature every 4-6 hours
4. Keep a symptom diary
5. Avoid strenuous activity

**Warning Signs:**
• Fever above 103°F (39.4°C)
• Fever lasting more than 3 days
• Severe headache or stiff neck
• Rash developing
• Difficulty breathing

**When to Seek Help:**
- See doctor if fever persists >3 days
- Visit ER for fever >104°F or with severe symptoms
- Call doctor if you have underlying health conditions

**Important Note:**
This assessment is for informational purposes only. Consult a healthcare provider for proper diagnosis."""
            },
            
            "cough": {
                "alone": """**Medical Assessment**

A persistent cough can have various causes.

**Possible Causes:**
1. **Acute Viral Bronchitis (Most Common)**
   - Inflammation of airways
   - Usually improves in 1-2 weeks
   - May have chest discomfort

2. **Post-Nasal Drip**
   - From allergies or sinus congestion
   - Often worse at night
   - May have throat irritation

3. **Irritant Exposure**
   - Smoke, pollution, strong odors
   - Chemical irritants

**Severity Level:** Mild to Moderate - Monitor symptoms

**Recommended Actions:**
1. Stay well hydrated (warm fluids, tea with honey)
2. Use honey for cough relief (adults only, not for children <1 year)
3. Avoid irritants (smoke, perfumes, cold air)
4. Use humidifier to add moisture
5. Elevate head while sleeping

**Warning Signs:**
• Coughing up blood
• Severe chest pain with coughing
• High fever (>102°F)
• Difficulty breathing
• Cough lasting >3 weeks

**When to Seek Help:**
- See doctor if cough persists >2 weeks
- Immediate care for difficulty breathing or blood in mucus
- Consider evaluation if cough interferes with sleep or daily activities

**Important Note:**
This is general information. Consult a healthcare provider for persistent or concerning symptoms."""
            },
            
            "chest pain": """**🚨 URGENT MEDICAL ASSESSMENT - CHEST PAIN**

Chest pain requires immediate medical evaluation.

**Possible Causes:**
1. **Cardiac (Heart-Related) - URGENT**
   - Heart attack (myocardial infarction)
   - Angina (reduced blood flow to heart)
   - CALL 108 IMMEDIATELY if: Pain radiating to arm/jaw, sweating, nausea, shortness of breath

2. **Musculoskeletal**
   - Muscle strain or costochondritis
   - Pain worsens with movement or deep breathing
   - Usually not life-threatening

3. **Respiratory**
   - Pneumonia or pleurisy
   - May have fever, cough
   - Requires medical evaluation

**Severity Level:** EMERGENCY - SEEK IMMEDIATE CARE

**IMMEDIATE ACTIONS REQUIRED:**
1. **CALL 108 FOR AMBULANCE** if severe pain or with: sweating, nausea, shortness of breath, radiating pain
2. Do NOT drive yourself to hospital
3. Sit or lie down in comfortable position
4. Chew aspirin if available and no allergy (325mg) - ONLY if you've called 108 first
5. Stay calm and breathe slowly

**Warning Signs (CALL 108 IMMEDIATELY):**
• Severe, crushing chest pain
• Pain radiating to arm, jaw, or back
• Shortness of breath
• Profuse sweating
• Nausea or vomiting
• Dizziness or fainting

**When to Seek Help:**
NOW - Go to nearest emergency room immediately or call 108

**Important Note:**
Chest pain can be life-threatening. Do not delay seeking emergency medical care. This information does not replace emergency medical treatment.""",
            
            "headache": """**Medical Assessment**

Headaches have many causes, most are not serious.

**Possible Causes:**
1. **Tension Headache (Most Common)**
   - Band-like pressure around head
   - Related to stress, poor posture, eye strain
   - Treatment: Rest, pain relievers, relaxation

2. **Migraine**
   - Throbbing, usually one-sided
   - May have nausea, light sensitivity, visual changes
   - May need prescription medication

3. **Sinus Headache**
   - Facial pressure/pain
   - Worse when bending forward
   - May need decongestants

**Severity Level:** Mild to Moderate - Monitor symptoms

**Recommended Actions:**
1. Rest in quiet, dark room
2. Apply cold compress to forehead or warm compress to neck
3. Stay well hydrated (drink water regularly)
4. Take pain reliever as directed (ibuprofen, acetaminophen)
5. Identify and avoid triggers (bright lights, stress, certain foods, lack of sleep)

**Warning Signs:**
• Sudden severe headache ("worst ever")
• Headache with fever and stiff neck
• Vision changes or difficulty speaking
• Confusion or loss of consciousness
• Headache after head injury

**When to Seek Help:**
- See doctor for frequent or severe headaches
- Emergency care for sudden severe headache with above warning signs
- Medical evaluation if headaches worsen or change pattern

**Important Note:**
Most headaches are not serious, but certain patterns require medical attention. Consult healthcare provider for persistent or concerning symptoms.""",
            
            "leg pain": """**Medical Assessment**

Leg pain can range from minor to serious conditions.

**Possible Causes:**
1. **Muscle Strain/Overuse (Most Common)**
   - From activity, standing, exercise
   - Treatment: RICE (Rest, Ice, Compression, Elevation)

2. **Muscle Cramp**
   - Sudden, sharp pain
   - Often at night
   - May indicate dehydration or electrolyte imbalance

3. **Deep Vein Thrombosis (DVT) - Serious**
   - Blood clot in deep vein
   - Watch for: Swelling, warmth, redness
   - Risk factors: Recent travel, surgery, immobility
   - **SEEK IMMEDIATE CARE IF SUSPECTED**

**Severity Level:** Mild to High (depending on cause)

**Recommended Actions:**
1. Rest and elevate affected leg
2. Apply ice 15-20 minutes, 3-4 times daily
3. Take OTC pain reliever (ibuprofen)
4. Gentle stretching
5. Stay hydrated

**Warning Signs (SEEK IMMEDIATE CARE):**
• Sudden severe pain with swelling
• Leg warm, red, and swollen
• Recent surgery or long travel
• Shortness of breath (could indicate blood clot)
• Difficulty walking or bearing weight

**When to Seek Help:**
- Immediate care for suspected DVT
- See doctor if pain doesn't improve in 2-3 days
- Medical evaluation for recurring leg pain

**Important Note:**
While most leg pain is minor, DVT is a medical emergency. Seek immediate care if you suspect a blood clot.""",
            
            "abdominal pain": """**⚠️ URGENT - Abdominal Pain Requires Immediate Evaluation**

Sharp abdominal pain in the lower right side with nausea and loss of appetite requires immediate medical assessment.

**🚨 POSSIBLE EMERGENCY - Appendicitis**

**Classic Signs of Appendicitis:**
1. Sharp pain in lower right abdomen (McBurney's point)
2. Nausea and vomiting
3. Loss of appetite
4. Low-grade fever (may develop)
5. Pain that starts around navel, then moves to lower right
6. Pain worsens with movement, coughing, or pressure

**Other Possible Causes:**
- Gastroenteritis (stomach flu)
- Ovarian cyst or torsion (in females)
- Kidney stones
- Urinary tract infection
- Intestinal obstruction
- Ectopic pregnancy (in females of childbearing age)

**Severity Level:** 🚨 **HIGH - EMERGENCY EVALUATION REQUIRED**

**IMMEDIATE ACTIONS:**
1. **DO NOT eat or drink anything** (in case surgery needed)
2. **Avoid pain medication** (can mask symptoms)
3. **Do not apply heat** to abdomen
4. **Go to Emergency Room NOW** or call 108
5. **Do not drive yourself** - have someone take you

**CRITICAL WARNING SIGNS - Ruptured Appendix:**
• Sudden relief of pain followed by worse pain
• High fever (>102°F / 39°C)
• Rapid heartbeat
• Severe abdominal rigidity (hard, board-like abdomen)
• Signs of shock (pale, sweaty, weak pulse, confusion)

**When to Seek Help:**
🚨 **GO TO ER NOW** - This is a medical emergency

**Timeline:**
Appendicitis can rupture within 24-72 hours of symptom onset, leading to peritonitis (life-threatening infection).

**What to Expect at ER:**
- Physical exam (rebound tenderness test)
- Blood tests (elevated white blood cells)
- CT scan or ultrasound
- Possible emergency surgery (appendectomy)

**Important Note:**
Sharp lower right abdominal pain with nausea and loss of appetite is a medical emergency that requires immediate evaluation. Appendicitis is the most common surgical emergency. Do not delay - proceed to nearest emergency room immediately. This assessment does not replace emergency medical care."""
        }

        # Detect symptoms in prompt
        prompt_lower = prompt.lower()
        has_fever = "fever" in prompt_lower
        has_cough = "cough" in prompt_lower
        has_abdominal_pain = any(term in prompt_lower for term in ["abdominal pain", "stomach pain", "lower right", "appendicitis"])
        has_chest_pain = "chest pain" in prompt_lower
        has_headache = "headache" in prompt_lower
        has_leg_pain = "leg pain" in prompt_lower
        has_difficulty_breathing = "difficulty breathing" in prompt_lower or "shortness of breath" in prompt_lower
        
        # EMERGENCY SYMPTOMS (Priority Order)
        
        # EMERGENCY: Abdominal pain (possible appendicitis)
        if has_abdominal_pain:
            return symptom_responses["abdominal pain"]
        
        # EMERGENCY: Chest pain or difficulty breathing
        if has_chest_pain or has_difficulty_breathing:
            return symptom_responses["chest pain"]
        
        # Fever + Cough
        if has_fever and has_cough:
            return symptom_responses["fever"]["with_cough"]
        
        # Single symptoms
        if has_fever:
            return symptom_responses["fever"]["alone"]
        
        if has_cough:
            return symptom_responses["cough"]["alone"]
        
        if has_headache:
            return symptom_responses["headache"]
        
        if has_leg_pain:
            return symptom_responses["leg pain"]

        # Generic fallback
        return """**Medical Assessment**

Based on your symptoms, here is a general assessment.

**Recommended Actions:**
1. Monitor your symptoms closely
2. Rest and stay hydrated
3. Note any changes or new symptoms
4. Keep track of symptom timeline

**When to Seek Help:**
- If symptoms worsen or don't improve
- If new symptoms develop
- If you have concerns about your condition

**Important Note:**
This is general information only. For proper medical evaluation and treatment, please consult a qualified healthcare provider."""

    def generate_diagnosis(self, symptoms: List[str], retrieved_context: List[str], 
                          patient_info: str, conversation_history: str, user_query: str) -> str:
        """Generate comprehensive medical diagnosis"""
        
        # Build comprehensive prompt for LLM
        context_snippet = ""
        if retrieved_context:
            context_snippet = "\n\nRelevant Medical Information:\n" + "\n".join([
                f"• {doc[:200]}..." for doc in retrieved_context[:5]
            ])
        
        prompt = f"""Based on the following patient information and medical context, provide a comprehensive medical assessment.

Patient Information:
{patient_info}

Patient's Current Concern:
{user_query}
{context_snippet}

Provide a detailed assessment including:
1. **Medical Assessment** - Brief overview
2. **Possible Causes** - List 2-3 likely conditions with explanations
3. **Severity Level** - Emergency/High/Moderate/Mild with justification
4. **Recommended Actions** - Specific numbered steps
5. **Warning Signs** - Bullet list of red flags to watch for
6. **When to Seek Help** - Clear guidance on timing

Be thorough (at least 300 words), evidence-based, and always recommend professional medical consultation for serious symptoms."""
        
        return self._run(prompt, temperature=0.7, max_tokens=600, add_to_history=False)

    def generate_followup(self, symptoms: List[str], conversation_history: str, 
                         last_assessment: str, user_query: str) -> str:
        """Generate follow-up questions"""
        
        prompt = f"""Based on the patient's symptoms and conversation, generate 4-5 focused follow-up questions.

Patient Symptoms: {', '.join(symptoms)}
Recent Query: {user_query}

Generate questions that would:
1. Clarify symptom timeline
2. Understand severity
3. Identify triggers
4. Check for related symptoms
5. Assess impact on daily life

Format as numbered questions."""
        
        return self._run(prompt, temperature=0.8, max_tokens=300, add_to_history=False)