"""
Agentic AI Medical Diagnosis System
Uses ReAct (Reasoning + Acting) pattern for autonomous medical assessment
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger("MedicalAgent")

class ActionType(Enum):
    """Types of actions the agent can take"""
    PARSE_SYMPTOMS = "parse_symptoms"
    RETRIEVE_MEDICAL_INFO = "retrieve_medical_info"
    ASSESS_SEVERITY = "assess_severity"
    GENERATE_DIAGNOSIS = "generate_diagnosis"
    RECOMMEND_ACTION = "recommend_action"
    ASK_FOLLOWUP = "ask_followup"
    THINK = "think"
    FINAL_ANSWER = "final_answer"


@dataclass
class Action:
    """Represents an action the agent takes"""
    action_type: ActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    reasoning: str = ""


@dataclass
class AgentThought:
    """Represents the agent's reasoning process"""
    step: int
    observation: str
    thought: str
    action: Action
    result: Optional[Any] = None


class MedicalAgent:
    """
    Agentic AI system for medical diagnosis using ReAct (Reasoning + Acting) pattern.
    
    The agent autonomously:
    1. Reasons about the patient's symptoms
    2. Plans diagnostic steps
    3. Retrieves relevant medical information
    4. Assesses severity
    5. Generates evidence-based recommendations
    """
    
    def __init__(self, llm_agent, rag_retriever, symptom_parser):
        """
        Initialize the medical agent with required components.
        
        Args:
            llm_agent: LLMAgent instance for generating text
            rag_retriever: RAGRetriever instance for medical knowledge
            symptom_parser: SymptomParser instance for extracting symptoms
        """
        self.llm_agent = llm_agent
        self.rag_retriever = rag_retriever
        self.symptom_parser = symptom_parser
        
        self.max_steps = 10
        self.thoughts: List[AgentThought] = []
        self.conversation_history: List[Dict[str, Any]] = []
        
    def _parse_action_from_llm(self, text: str) -> Tuple[ActionType, Dict[str, Any]]:
        """Parse LLM output to extract action and parameters"""
        
        action_patterns = {
            ActionType.PARSE_SYMPTOMS: r"parse[_\s]?symptoms?",
            ActionType.RETRIEVE_MEDICAL_INFO: r"retrieve|search|look[_\s]?up",
            ActionType.ASSESS_SEVERITY: r"assess|evaluate[_\s]?severity",
            ActionType.GENERATE_DIAGNOSIS: r"generate|create|provide[_\s]?diagnosis",
            ActionType.RECOMMEND_ACTION: r"recommend|suggest[_\s]?action",
            ActionType.ASK_FOLLOWUP: r"ask|generate[_\s]?followup",
            ActionType.FINAL_ANSWER: r"final[_\s]?answer|conclude",
        }
        
        action_type = ActionType.THINK  # default
        for atype, pattern in action_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                action_type = atype
                break
        
        parameters = {}
        if "symptoms" in text.lower():
            symptoms_match = re.search(r"symptoms?:?\s*([^.]+)", text, re.IGNORECASE)
            if symptoms_match:
                parameters["symptoms"] = symptoms_match.group(1).strip()
        
        if "severity" in text.lower():
            severity_match = re.search(r"severity:?\s*(\w+)", text, re.IGNORECASE)
            if severity_match:
                parameters["severity"] = severity_match.group(1).strip()
        
        return action_type, parameters
    
    def _execute_action(self, action: Action, patient_input: str, context: Dict[str, Any]) -> Any:
        """Execute the given action and return the result"""
        
        logger.info(f"Executing action: {action.action_type.value}")
        
        if action.action_type == ActionType.PARSE_SYMPTOMS:
            # Extract symptoms from patient input
            symptoms = self.symptom_parser.parse_symptoms(patient_input)
            action.result = list(symptoms) if symptoms else []
            action.reasoning = f"Identified {len(action.result)} symptoms from patient description"
            return action.result
        
        elif action.action_type == ActionType.RETRIEVE_MEDICAL_INFO:
            # Retrieve medical information related to symptoms
            symptoms = context.get("symptoms", [])
            query = " ".join(symptoms) if symptoms else patient_input
            docs = self.rag_retriever.retrieve(query, k=5)
            action.result = docs
            action.reasoning = f"Retrieved {len(docs)} medical documents related to: {query}"
            return docs
        
        elif action.action_type == ActionType.ASSESS_SEVERITY:
            # Assess severity based on symptoms
            symptoms = context.get("symptoms", [])
            severity = self._assess_severity_internal(symptoms)
            action.result = severity
            action.reasoning = f"Assessed severity as: {severity['level']}"
            return severity
        
        elif action.action_type == ActionType.GENERATE_DIAGNOSIS:
            # Generate diagnosis using LLM
            symptoms = context.get("symptoms", [])
            medical_info = context.get("medical_info", [])
            
            prompt = self._build_diagnosis_prompt(
                patient_input, symptoms, medical_info, context
            )
            
            diagnosis = self.llm_agent._run(
                prompt,
                temperature=0.7,
                max_tokens=500,
                add_to_history=False
            )
            action.result = diagnosis
            action.reasoning = "Generated diagnosis based on symptoms and medical literature"
            return diagnosis
        
        elif action.action_type == ActionType.RECOMMEND_ACTION:
            # Generate recommendations
            severity = context.get("severity", {})
            symptoms = context.get("symptoms", [])
            
            recommendations = self._generate_recommendations(severity, symptoms)
            action.result = recommendations
            action.reasoning = "Generated recommendations based on severity and symptoms"
            return recommendations
        
        elif action.action_type == ActionType.ASK_FOLLOWUP:
            # Generate follow-up questions
            symptoms = context.get("symptoms", [])
            diagnosis = context.get("diagnosis", "")
            
            followups = self._generate_followups(symptoms, diagnosis, patient_input)
            action.result = followups
            action.reasoning = "Generated follow-up questions for deeper assessment"
            return followups
        
        elif action.action_type == ActionType.FINAL_ANSWER:
            # Prepare final answer
            diagnosis = context.get("diagnosis", "Unable to determine")
            recommendations = context.get("recommendations", [])
            followups = context.get("followups", [])
            
            final_answer = self._format_final_answer(
                diagnosis, recommendations, followups, context
            )
            action.result = final_answer
            action.reasoning = "Compiled final comprehensive medical assessment"
            return final_answer
        
        else:  # THINK
            # Let LLM reason about next steps
            reasoning_prompt = self._build_reasoning_prompt(patient_input, context)
            thought = self.llm_agent._run(
                reasoning_prompt,
                temperature=0.8,
                max_tokens=300,
                add_to_history=False
            )
            action.result = thought
            action.reasoning = thought
            return thought
    
    def _build_reasoning_prompt(self, patient_input: str, context: Dict[str, Any]) -> str:
        """Build prompt for agent reasoning"""
        symptoms = context.get("symptoms", [])
        medical_info = context.get("medical_info", [])
        
        prompt = f"""You are a medical diagnosis AI. Reason about the following:

Patient's statement: {patient_input}

Identified symptoms: {', '.join(symptoms) if symptoms else 'None yet'}

Available medical context: {len(medical_info)} documents retrieved

What should be the next diagnostic step? Consider:
1. What information is missing?
2. What additional symptoms might be relevant?
3. What severity assessment is needed?
4. Should we ask follow-up questions first?

Provide your reasoning in 2-3 sentences."""
        
        return prompt
    
    def _build_diagnosis_prompt(
        self, 
        patient_input: str, 
        symptoms: List[str], 
        medical_info: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Build comprehensive diagnosis prompt"""
        severity = context.get("severity", {})
        severity_level = severity.get("level", "Unknown")
        
        medical_context = "\n".join(medical_info[:3]) if medical_info else "No medical context available"
        
        prompt = f"""As a medical AI assistant, provide a structured diagnosis assessment:

Patient Description: {patient_input}

Identified Symptoms: {', '.join(symptoms) if symptoms else 'None'}

Severity Level: {severity_level}

Medical Literature Context:
{medical_context}

Provide a comprehensive assessment including:
1. Likely conditions (with confidence levels)
2. Differential diagnoses
3. Red flags to watch for
4. Recommended next steps
5. When to seek emergency care

Be evidence-based and conservative in diagnosis."""
        
        return prompt
    
    def _assess_severity_internal(self, symptoms: List[str]) -> Dict[str, Any]:
        """Internal severity assessment based on symptoms"""
        
        emergency_symptoms = {
            "chest pain", "difficulty breathing", "severe headache",
            "loss of consciousness", "severe bleeding", "severe pain"
        }
        
        high_symptoms = {
            "high fever", "persistent vomiting", "severe dizziness",
            "rapid heartbeat", "severe abdominal pain"
        }
        
        symptoms_lower = [s.lower() for s in symptoms]
        
        has_emergency = any(
            any(es in symptom for symptom in symptoms_lower)
            for es in emergency_symptoms
        )
        has_high = any(
            any(hs in symptom for symptom in symptoms_lower)
            for hs in high_symptoms
        )
        
        if has_emergency:
            return {
                "level": "Emergency",
                "score": 5,
                "recommendation": "Seek immediate medical attention (911/108)",
                "priority": "critical"
            }
        elif has_high or len(symptoms) >= 4:
            return {
                "level": "High",
                "score": 4,
                "recommendation": "See doctor urgently (within hours)",
                "priority": "high"
            }
        elif len(symptoms) >= 2:
            return {
                "level": "Moderate",
                "score": 3,
                "recommendation": "Schedule doctor visit (24-48 hours)",
                "priority": "medium"
            }
        else:
            return {
                "level": "Mild",
                "score": 2,
                "recommendation": "Monitor symptoms, rest",
                "priority": "low"
            }
    
    def _generate_recommendations(self, severity: Dict[str, Any], symptoms: List[str]) -> List[str]:
        """Generate medical recommendations"""
        recommendations = []
        
        severity_level = severity.get("level", "Unknown")
        
        if severity_level == "Emergency":
            recommendations.append("🚨 Call emergency services (911/108) immediately")
            recommendations.append("Do not drive yourself; wait for ambulance")
        elif severity_level == "High":
            recommendations.append("Contact your doctor immediately")
            recommendations.append("Visit urgent care or emergency room")
        elif severity_level == "Moderate":
            recommendations.append("Schedule a doctor's appointment within 24-48 hours")
            recommendations.append("Consider telehealth consultation if urgent")
        else:
            recommendations.append("Monitor your symptoms closely")
            recommendations.append("Rest and stay hydrated")
        
        # Symptom-specific recommendations
        symptoms_lower = [s.lower() for s in symptoms]
        
        if any("fever" in s for s in symptoms_lower):
            recommendations.append("Take temperature regularly; use fever reducers as directed")
        
        if any("cough" in s for s in symptoms_lower):
            recommendations.append("Avoid irritants; use honey for throat comfort")
        
        if any("headache" in s for s in symptoms_lower):
            recommendations.append("Rest in quiet, dark room; stay hydrated")
        
        recommendations.append("Keep all medications nearby and organized")
        recommendations.append("Seek emergency care if symptoms worsen suddenly")
        
        return recommendations
    
    def _generate_followups(self, symptoms: List[str], diagnosis: str, patient_input: str) -> str:
        """Generate follow-up questions"""
        
        questions = ["**Important Follow-up Questions:**"]
        
        if not any(word in patient_input.lower() for word in ["started", "began", "started", "days"]):
            questions.append("1. When exactly did these symptoms start?")
        
        if not any(word in patient_input.lower() for word in ["worse", "better", "improving", "worsening"]):
            questions.append("2. Are symptoms getting worse, better, or staying the same?")
        
        if not any(word in patient_input.lower() for word in ["trigger", "cause", "relief"]):
            questions.append("3. What makes symptoms better or worse?")
        
        if not any(word in patient_input.lower() for word in ["medication", "medical", "history", "condition"]):
            questions.append("4. Do you have any medical conditions or take medications?")
        
        if len(questions) < 4:
            questions.append(f"{len(questions)}. Is there anything else affecting your symptoms?")
        
        return "\n".join(questions)
    
    def _format_final_answer(
        self,
        diagnosis: str,
        recommendations: List[str],
        followups: str,
        context: Dict[str, Any]
    ) -> str:
        """Format the final comprehensive answer"""
        
        severity = context.get("severity", {})
        severity_level = severity.get("level", "Unknown")
        
        answer = f"""
**Medical Assessment**

**Severity Level:** {severity_level}

**Diagnosis and Assessment:**
{diagnosis}

**Recommended Actions:**
{chr(10).join(f"• {rec}" for rec in recommendations)}

{followups}

**Disclaimer:** This assessment is for informational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.
"""
        return answer.strip()
    
    def diagnose(self, patient_input: str) -> Tuple[str, List[AgentThought]]:
        """
        Main agent function: Autonomously diagnose based on patient input
        
        Returns:
            Tuple of (final_diagnosis, thought_process)
        """
        
        logger.info(f"Starting medical diagnosis for: {patient_input[:100]}...")
        
        self.thoughts = []
        context: Dict[str, Any] = {
            "patient_input": patient_input,
            "symptoms": [],
            "medical_info": [],
            "severity": {},
            "diagnosis": "",
            "recommendations": [],
            "followups": ""
        }
        
        step = 1
        
        # Step 1: Parse symptoms
        action = Action(
            action_type=ActionType.PARSE_SYMPTOMS,
            description="Extract and identify symptoms from patient description"
        )
        symptoms = self._execute_action(action, patient_input, context)
        context["symptoms"] = symptoms
        
        thought = AgentThought(
            step=step,
            observation=f"Found {len(symptoms)} symptoms: {', '.join(symptoms)}",
            thought="Symptoms identified. Need medical information to assess.",
            action=action,
            result=symptoms
        )
        self.thoughts.append(thought)
        step += 1
        
        # Step 2: Retrieve medical information
        action = Action(
            action_type=ActionType.RETRIEVE_MEDICAL_INFO,
            description="Look up medical information related to identified symptoms"
        )
        medical_info = self._execute_action(action, patient_input, context)
        context["medical_info"] = medical_info
        
        thought = AgentThought(
            step=step,
            observation=f"Retrieved {len(medical_info)} medical documents",
            thought="Medical context acquired. Now assess severity.",
            action=action,
            result=medical_info
        )
        self.thoughts.append(thought)
        step += 1
        
        # Step 3: Assess severity
        action = Action(
            action_type=ActionType.ASSESS_SEVERITY,
            description="Evaluate symptom severity to determine urgency"
        )
        severity = self._execute_action(action, patient_input, context)
        context["severity"] = severity
        
        thought = AgentThought(
            step=step,
            observation=f"Severity: {severity['level']} (Priority: {severity['priority']})",
            thought=f"Severity assessed as {severity['level']}. Generate diagnosis.",
            action=action,
            result=severity
        )
        self.thoughts.append(thought)
        step += 1
        
        # Step 4: Generate diagnosis
        action = Action(
            action_type=ActionType.GENERATE_DIAGNOSIS,
            description="Generate diagnosis based on symptoms and medical literature"
        )
        diagnosis = self._execute_action(action, patient_input, context)
        context["diagnosis"] = diagnosis
        
        thought = AgentThought(
            step=step,
            observation=f"Diagnosis generated ({len(diagnosis)} characters)",
            thought="Diagnosis complete. Generate recommendations.",
            action=action,
            result=diagnosis[:200] + "..." if len(diagnosis) > 200 else diagnosis
        )
        self.thoughts.append(thought)
        step += 1
        
        # Step 5: Generate recommendations
        action = Action(
            action_type=ActionType.RECOMMEND_ACTION,
            description="Create actionable recommendations for patient"
        )
        recommendations = self._execute_action(action, patient_input, context)
        context["recommendations"] = recommendations
        
        thought = AgentThought(
            step=step,
            observation=f"Generated {len(recommendations)} recommendations",
            thought="Recommendations complete. Generate follow-up questions.",
            action=action,
            result=recommendations
        )
        self.thoughts.append(thought)
        step += 1
        
        # Step 6: Generate follow-up questions
        action = Action(
            action_type=ActionType.ASK_FOLLOWUP,
            description="Generate relevant follow-up questions for deeper assessment"
        )
        followups = self._execute_action(action, patient_input, context)
        context["followups"] = followups
        
        thought = AgentThought(
            step=step,
            observation="Follow-up questions generated",
            thought="All diagnostic steps complete. Prepare final answer.",
            action=action,
            result=followups[:200] + "..." if len(followups) > 200 else followups
        )
        self.thoughts.append(thought)
        step += 1
        
        # Step 7: Final answer
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            description="Compile final comprehensive medical assessment"
        )
        final_answer = self._execute_action(action, patient_input, context)
        
        thought = AgentThought(
            step=step,
            observation="Final assessment compiled",
            thought="Medical diagnosis process complete.",
            action=action,
            result=final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
        )
        self.thoughts.append(thought)
        
        logger.info(f"Diagnosis complete in {step} steps")
        
        return final_answer, self.thoughts
    
    def format_thoughts_for_ui(self) -> str:
        """Format agent thoughts as readable markdown for Gradio UI display"""
        if not self.thoughts:
            return ""
        
        md_output = "## 🧠 Agent Reasoning Process (ReAct Pattern)\n\n"
        
        for thought in self.thoughts:
            md_output += f"### Step {thought.step}: {thought.action.action_type.value}\n"
            md_output += f"**Action**: {thought.action.description}\n"
            md_output += f"**Observation**: {thought.observation}\n"
            md_output += f"**Thought**: {thought.thought}\n"
            
            if thought.result:
                if isinstance(thought.result, list):
                    md_output += f"**Result**: {len(thought.result)} items\n"
                elif isinstance(thought.result, dict):
                    md_output += f"**Result**: {json.dumps(thought.result, indent=2)}\n"
                else:
                    result_str = str(thought.result)
                    if len(result_str) > 300:
                        md_output += f"**Result**: {result_str[:300]}...\n"
                    else:
                        md_output += f"**Result**: {result_str}\n"
            
            md_output += "\n"
        
        return md_output
