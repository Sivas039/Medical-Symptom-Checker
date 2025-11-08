

from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from IndicTransToolkit.processor import IndicProcessor
    import torch
    _HAS_TRANSLATION = True
except Exception:
    _HAS_TRANSLATION = False

class Translator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") if _HAS_TRANSLATION else "cpu"

        if not _HAS_TRANSLATION:
            print("Translator: IndicTransToolkit or HF models not available. Translator will be no-op.")
            return

        print(f"Loading translation models on {self.device}...")
        self.tokenizer_indic_en = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True)
        self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True).to(self.device)
        self.ip_indic_en = IndicProcessor(inference=True)

        self.tokenizer_en_indic = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
        self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True).to(self.device)
        self.ip_en_indic = IndicProcessor(inference=True)

        print("Translator ready.")

    def translate_to_english(self, text: str, src_lang: str) -> str:
        if not _HAS_TRANSLATION or not text:
            return text

        batch = self.ip_indic_en(text, src_lang=src_lang, tgt_lang="eng_Latn")
        inputs = self.tokenizer_indic_en(batch, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated = self.model_indic_en.generate(**inputs, num_beams=4, max_length=256)
        decoded = self.tokenizer_indic_en.batch_decode(generated, skip_special_tokens=True)
        return decoded[0] if decoded else text

    def translate_from_english(self, text: str, tgt_lang: str) -> str:
        if not _HAS_TRANSLATION or not text:
            return text

        batch = self.ip_en_indic(text, src_lang="eng_Latn", tgt_lang=tgt_lang)
        inputs = self.tokenizer_en_indic(batch, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated = self.model_en_indic.generate(**inputs, num_beams=4, max_length=256)
        decoded = self.tokenizer_en_indic.batch_decode(generated, skip_special_tokens=True)
        return decoded[0] if decoded else text
