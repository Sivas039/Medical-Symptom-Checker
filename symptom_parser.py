
import re
from typing import Set

class SymptomParser:
    """
    Symptom parser with optional spaCy acceleration and a robust regex fallback.
    If spaCy cannot be imported or causes heavy imports, it gracefully falls back to regex-only.
    """

    COMMON_SINGLE = [
        "fever", "cough", "headache", "chills", "fatigue", "nausea", "rash",
        "dizziness", "vomit", "vomiting"
    ]

    COMMON_MULTI = [
        "sore throat", "shortness of breath", "chest pain", "body aches",
        "runny nose", "stuffy nose", "muscle pain", "joint pain", "high fever",
        "difficulty breathing", "abdominal pain", "stomach pain", "diarrhea",
        "leg pain", "neck pain", "back pain", "arm pain", "foot pain", "knee pain",
        "shoulder pain", "head pain", "eye pain", "ear pain", "throat pain"
    ]

    FALLBACK_PATTERN = re.compile(
        r"\b(" + "|".join(re.escape(s) for s in (COMMON_SINGLE + COMMON_MULTI)) + r")\b",
        re.IGNORECASE
    )

    def __init__(self, model: str = "en_core_web_sm", enable_spacy: bool = True):
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None

        if enable_spacy:
            try:
                import spacy 
                from spacy.matcher import Matcher, PhraseMatcher  
                if not spacy.util.is_package(model):
                    try:
                        spacy.cli.download(model)
                    except Exception:
                        pass

                self.nlp = spacy.load(model, disable=["parser", "ner"])  # type: ignore
                self.matcher = Matcher(self.nlp.vocab)
                self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                self._setup_rules()

            except Exception as _e:
                self.nlp = None
                self.matcher = None
                self.phrase_matcher = None

    def _setup_rules(self):
        if not self.nlp or not self.matcher or not self.phrase_matcher:
            return

        for s in self.COMMON_SINGLE:
            pattern = [{"LOWER": s}]
            self.matcher.add(s.upper(), [pattern])

        patterns = [self.nlp.make_doc(text) for text in self.COMMON_MULTI]
        if patterns:
            self.phrase_matcher.add("MULTI_WORD", patterns)

    def parse_symptoms(self, text: str) -> Set[str]:
        """
        Parse the input text and return a set of symptom strings in lowercase.
        Non-destructive: returns empty set if none found.
        """
        if not text:
            return set()

        if self.nlp and self.matcher and self.phrase_matcher:
            try:
                doc = self.nlp(text)
            except Exception:
                return self._regex_fallback(text)

            found = set()

            for match_id, start, end in self.matcher(doc):
                span = doc[start:end].text.strip().lower()
                if span:
                    found.add(span)

            for match_id, start, end in self.phrase_matcher(doc):
                span = doc[start:end].text.strip().lower()
                if span:
                    found.add(span)
        else:
            found = set()

        regex_map = {
            "headache": r"\bheadaches?\b",
            "body ache": r"\bbody\s+aches?\b",
            "difficulty breathing": r"\bdifficulty\s+(in\s+)?breathing\b"
        }

        for name, pat in regex_map.items():
            if re.search(pat, text, re.IGNORECASE):
                found.add(name)

        if "high fever" in found:
            found.add("fever")

        if not found:
            return self._regex_fallback(text)

        normalized = set()
        for s in found:
            s2 = s.strip().lower()
            if s2 in ("stomach pain", "abdominal pain"):
                normalized.add("stomach pain")
            elif s2 in ("body ache", "body aches"):
                normalized.add("body aches")
            else:
                normalized.add(s2)

        print(f"Parsed symptoms: {normalized}")
        return normalized

    def _regex_fallback(self, text: str) -> Set[str]:
        matches = self.FALLBACK_PATTERN.findall(text or "")
        normalized = set(m.lower() for m in matches)
        if normalized:
            print("parse_symptoms: using regex fallback ->", normalized)
        return normalized
