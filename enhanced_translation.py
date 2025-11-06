"""
Enhanced Translation System for ISL Bridge
"""
import pyttsx3
from typing import Optional, Dict, List, Tuple
import logging

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    print("Google Translate not available. Install with: pip install googletrans==4.0.0rc1")

try:
    from config import TRANSLATION_CONFIG
except ImportError:
    TRANSLATION_CONFIG = {
        "default_language": "en",
        "supported_languages": ["en", "hi", "ta", "te", "bn", "gu", "mr", "pa"],
        "tts_enabled": True,
        "tts_rate": 150,
        "tts_volume": 0.8
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLanguageTranslator:
    """Multi-language translation and text-to-speech system"""
    
    def __init__(self):
        self.tts_engine: Optional[pyttsx3.Engine] = None
        self.google_translator = None
        self.current_language = TRANSLATION_CONFIG.get("default_language", "en")
        
        import threading
        self.tts_lock = threading.Lock()
        self.tts_running = False
        
        self.setup_translation()
        self.setup_tts()
        
        self.gesture_dictionary = {
            "A": {"en": "A", "hi": "A", "ta": "A", "te": "A"},
            "B": {"en": "B", "hi": "B", "ta": "B", "te": "B"},
            "C": {"en": "C", "hi": "C", "ta": "C", "te": "C"},
            "D": {"en": "D", "hi": "D", "ta": "D", "te": "D"},
            "E": {"en": "E", "hi": "E", "ta": "E", "te": "E"},
            "F": {"en": "F", "hi": "F", "ta": "F", "te": "F"},
            "G": {"en": "G", "hi": "G", "ta": "G", "te": "G"},
            "H": {"en": "H", "hi": "H", "ta": "H", "te": "H"},
            "I": {"en": "I", "hi": "I", "ta": "I", "te": "I"},
            "J": {"en": "J", "hi": "J", "ta": "J", "te": "J"},
            "K": {"en": "K", "hi": "K", "ta": "K", "te": "K"},
            "L": {"en": "L", "hi": "L", "ta": "L", "te": "L"},
            "M": {"en": "M", "hi": "M", "ta": "M", "te": "M"},
            "N": {"en": "N", "hi": "N", "ta": "N", "te": "N"},
            "O": {"en": "O", "hi": "O", "ta": "O", "te": "O"},
            "P": {"en": "P", "hi": "P", "ta": "P", "te": "P"},
            "Q": {"en": "Q", "hi": "Q", "ta": "Q", "te": "Q"},
            "R": {"en": "R", "hi": "R", "ta": "R", "te": "R"},
            "S": {"en": "S", "hi": "S", "ta": "S", "te": "S"},
            "T": {"en": "T", "hi": "T", "ta": "T", "te": "T"},
            "U": {"en": "U", "hi": "U", "ta": "U", "te": "U"},
            "V": {"en": "V", "hi": "V", "ta": "V", "te": "V"},
            "W": {"en": "W", "hi": "W", "ta": "W", "te": "W"},
            "X": {"en": "X", "hi": "X", "ta": "X", "te": "X"},
            "Y": {"en": "Y", "hi": "Y", "ta": "Y", "te": "Y"},
            "Z": {"en": "Z", "hi": "Z", "ta": "Z", "te": "Z"},
            
            "0": {"en": "0", "hi": "0", "ta": "0", "te": "0"},
            "1": {"en": "1", "hi": "1", "ta": "1", "te": "1"},
            "2": {"en": "2", "hi": "2", "ta": "2", "te": "2"},
            "3": {"en": "3", "hi": "3", "ta": "3", "te": "3"},
            "4": {"en": "4", "hi": "4", "ta": "4", "te": "4"},
            "5": {"en": "5", "hi": "5", "ta": "5", "te": "5"},
            "6": {"en": "6", "hi": "6", "ta": "6", "te": "6"},
            "7": {"en": "7", "hi": "7", "ta": "7", "te": "7"},
            "8": {"en": "8", "hi": "8", "ta": "8", "te": "8"},
            "9": {"en": "9", "hi": "9", "ta": "9", "te": "9"},
            
            "HELLO": {"en": "Hello", "hi": "नमस्ते", "ta": "வணக்கம்", "te": "నమస్కారం"},
            "THANK_YOU": {"en": "Thank you", "hi": "धन्यवाद", "ta": "நன்றி", "te": "ధన్యవాదాలు"},
            "PLEASE": {"en": "Please", "hi": "कृपया", "ta": "தயவுசெய்து", "te": "దయచేసి"},
            "SORRY": {"en": "Sorry", "hi": "माफ़ करें", "ta": "மன்னிக்கவும்", "te": "క్షమించండి"},
            "YES": {"en": "Yes", "hi": "हाँ", "ta": "ஆம்", "te": "అవును"},
            "NO": {"en": "No", "hi": "नहीं", "ta": "இல்லை", "te": "కాదు"},
            "HELP": {"en": "Help", "hi": "मदद", "ta": "உதவி", "te": "సహాయం"},
            "WATER": {"en": "Water", "hi": "पानी", "ta": "தண்ணீர்", "te": "నీరు"},
            "FOOD": {"en": "Food", "hi": "खाना", "ta": "உணவு", "te": "ఆహారం"},
            "BATHROOM": {"en": "Bathroom", "hi": "बाथरूम", "ta": "குளியலறை", "te": "బాత్రూమ్"},
            "STOP": {"en": "Stop", "hi": "रुको", "ta": "நிறுத்து", "te": "ఆపు"},
            "GO": {"en": "Go", "hi": "जाओ", "ta": "போ", "te": "వెళ్ళు"},
            
            "MOTHER": {"en": "Mother", "hi": "माँ", "ta": "அம்மா", "te": "అమ్మ"},
            "FATHER": {"en": "Father", "hi": "पिता", "ta": "அப்பா", "te": "నాన్న"},
            "BROTHER": {"en": "Brother", "hi": "भाई", "ta": "அண்ணா", "te": "అన్న"},
            "SISTER": {"en": "Sister", "hi": "बहन", "ta": "அக்கா", "te": "అక్క"},
            
            "HUNGRY": {"en": "Hungry", "hi": "भूखा", "ta": "பசி", "te": "ఆకలి"},
            "THIRSTY": {"en": "Thirsty", "hi": "प्यासा", "ta": "தாகம்", "te": "దాహం"},
            "TIRED": {"en": "Tired", "hi": "थका", "ta": "சோர்வு", "te": "అలసట"},
            "HAPPY": {"en": "Happy", "hi": "खुश", "ta": "மகிழ்ச்சி", "te": "సంతోషం"},
            "SAD": {"en": "Sad", "hi": "दुखी", "ta": "துக்கம்", "te": "దుఃఖం"},
        }
    
    def setup_translation(self):
        """Initialize Google Translate if available"""
        if GOOGLETRANS_AVAILABLE:
            try:
                self.google_translator = Translator()
                logger.info("Google Translator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Translator: {e}")
                self.google_translator = None
        else:
            logger.info("Using local translation dictionary only")
    
    def setup_tts(self) -> None:
        """Initialize TTS engine with optimized settings"""
        try:
            self.tts_engine = pyttsx3.init()
            
            rate = TRANSLATION_CONFIG.get("tts_rate", 150)
            volume = TRANSLATION_CONFIG.get("tts_volume", 0.8)
            
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', volume)
            
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if self.current_language in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
    
    def set_language(self, language_code: str) -> bool:
        """Change the target language for translation"""
        supported_languages = TRANSLATION_CONFIG.get("supported_languages", ["en", "hi"])
        
        if language_code in supported_languages:
            self.current_language = language_code
            logger.info(f"Language changed to: {language_code}")
            return True
        else:
            logger.warning(f"Language {language_code} not supported")
            return False
    
    def translate_gesture(self, gesture: str, target_language: Optional[str] = None) -> str:
        """
        Translate ISL gesture to target language
        
        Args:
            gesture: Recognized gesture
            target_language: Target language code (defaults to current language)
            
        Returns:
            Translated text
        """
        target_lang = target_language or self.current_language
        
        if gesture.upper() in self.gesture_dictionary:
            gesture_translations = self.gesture_dictionary[gesture.upper()]
            if target_lang in gesture_translations:
                return gesture_translations[target_lang]
        
        if self.google_translator and len(gesture.split()) > 1:
            try:
                translated = self.google_translator.translate(
                    gesture, 
                    src='en', 
                    dest=target_lang
                )
                return translated.text
            except Exception as e:
                logger.error(f"Google Translate failed: {e}")
        
        return gesture
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with their names"""
        language_names = {
            "en": "English",
            "hi": "हिन्दी (Hindi)",
            "ta": "தமிழ் (Tamil)",
            "te": "తెలుగు (Telugu)",
            "bn": "বাংলা (Bengali)",
            "gu": "ગુજરાતી (Gujarati)",
            "mr": "मराठी (Marathi)",
            "pa": "ਪੰਜਾਬੀ (Punjabi)"
        }
        
        supported = TRANSLATION_CONFIG.get("supported_languages", ["en", "hi"])
        return [
            {"code": lang, "name": language_names.get(lang, lang.upper())}
            for lang in supported
            if isinstance(lang, str)
        ]
    
    def translate_sentence(self, sentence: str, target_language: Optional[str] = None) -> str:
        """
        Translate a complete sentence
        
        Args:
            sentence: Input sentence
            target_language: Target language code
            
        Returns:
            Translated sentence
        """
        target_lang = target_language or self.current_language
        
        words = sentence.split()
        if all(len(word) == 1 and word.isalnum() for word in words):
            translated_words = []
            for word in words:
                translated_words.append(self.translate_gesture(word, target_lang))
            return " ".join(translated_words)
        
        if self.google_translator:
            try:
                translated = self.google_translator.translate(
                    sentence,
                    src='en',
                    dest=target_lang
                )
                return translated.text
            except Exception as e:
                logger.error(f"Sentence translation failed: {e}")
        
        return sentence
    
    def speak_text(self, text: str, language: Optional[str] = None) -> bool:
        """
        Convert text to speech in specified language (non-blocking)
        For letters/numbers, speak the full word name for clarity
        
        Args:
            text: Text to speak
            language: Language code for TTS
            
        Returns:
            True if successful, False otherwise
        """
        if not self.tts_engine:
            logger.error("TTS engine not available")
            return False
        
        try:
            lang = language or self.current_language
            
            if len(text) == 1 and text.isalnum():
                spoken_map = {
                    # Letters - spoken form
                    "A": {"en": "Letter A", "hi": "अक्षर ए", "ta": "எழுத்து அ", "te": "అక్షరం అ"},
                    "B": {"en": "Letter B", "hi": "अक्षर बी", "ta": "எழுத்து பி", "te": "అక్షరం బి"},
                    "C": {"en": "Letter C", "hi": "अक्षर सी", "ta": "எழுத்து சி", "te": "అక్షరం సి"},
                    "D": {"en": "Letter D", "hi": "अक्षर डी", "ta": "எழுத்து டி", "te": "అక్షరం డి"},
                    "E": {"en": "Letter E", "hi": "अक्षर ई", "ta": "எழுத்து ஈ", "te": "అక్షరం ఈ"},
                    "0": {"en": "Zero", "hi": "शून्य", "ta": "பூஜ்யம்", "te": "శూన్యం"},
                    "1": {"en": "One", "hi": "एक", "ta": "ஒன்று", "te": "ఒకటి"},
                    "2": {"en": "Two", "hi": "दो", "ta": "இரண்டு", "te": "రెండు"},
                    "3": {"en": "Three", "hi": "तीन", "ta": "மூன்று", "te": "మూడు"},
                    "4": {"en": "Four", "hi": "चार", "ta": "நான்கு", "te": "నాలుగు"},
                    "5": {"en": "Five", "hi": "पांच", "ta": "ஐந்து", "te": "ఐదు"},
                    "6": {"en": "Six", "hi": "छह", "ta": "ஆறு", "te": "ఆరు"},
                    "7": {"en": "Seven", "hi": "सात", "ta": "ஏழு", "te": "ఏడు"},
                    "8": {"en": "Eight", "hi": "आठ", "ta": "எட்டு", "te": "ఎనిమిది"},
                    "9": {"en": "Nine", "hi": "नौ", "ta": "ஒன்பது", "te": "తొమ్మిది"},
                }
                
                if text.upper() in spoken_map:
                    text_to_speak = spoken_map[text.upper()].get(lang, text)
                else:
                    text_to_speak = text
            else:
                text_to_speak = self.translate_gesture(text, lang)
            
            if self.tts_running:
                logger.debug("TTS already running, skipping this request")
                return False
            
            def run_tts():
                with self.tts_lock:
                    try:
                        if self.tts_engine and not self.tts_running:
                            self.tts_running = True
                            self.tts_engine.say(text_to_speak)
                            self.tts_engine.runAndWait()
                    except Exception as e:
                        logger.error(f"TTS thread error: {e}")
                    finally:
                        self.tts_running = False
            
            import threading
            threading.Thread(target=run_tts, daemon=True).start()
            return True
        
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

TranslationHandler = MultiLanguageTranslator
