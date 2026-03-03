"""
Language codes and configurations for multi-language support
"""

SUPPORTED_LANGUAGES = {
    # Indian Languages (Priority)
    'hi': {'name': 'Hindi', 'native': 'हिंदी', 'rtl': False, 'voice': 'hi-IN'},
    'ta': {'name': 'Tamil', 'native': 'தமிழ்', 'rtl': False, 'voice': 'ta-IN'},
    'te': {'name': 'Telugu', 'native': 'తెలుగు', 'rtl': False, 'voice': 'te-IN'},
    'kn': {'name': 'Kannada', 'native': 'ಕನ್ನಡ', 'rtl': False, 'voice': 'kn-IN'},
    'ml': {'name': 'Malayalam', 'native': 'മലയാളം', 'rtl': False, 'voice': 'ml-IN'},
    'bn': {'name': 'Bengali', 'native': 'বাংলা', 'rtl': False, 'voice': 'bn-IN'},
    'mr': {'name': 'Marathi', 'native': 'मराठी', 'rtl': False, 'voice': 'mr-IN'},
    'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી', 'rtl': False, 'voice': 'gu-IN'},
    'pa': {'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ', 'rtl': False, 'voice': 'pa-IN'},
    'ur': {'name': 'Urdu', 'native': 'اردو', 'rtl': True, 'voice': 'ur-IN'},
    
    # English
    'en': {'name': 'English', 'native': 'English', 'rtl': False, 'voice': 'en-US'},
    
    # Other Major Languages
    'es': {'name': 'Spanish', 'native': 'Español', 'rtl': False, 'voice': 'es-ES'},
    'fr': {'name': 'French', 'native': 'Français', 'rtl': False, 'voice': 'fr-FR'},
    'de': {'name': 'German', 'native': 'Deutsch', 'rtl': False, 'voice': 'de-DE'},
    'pt': {'name': 'Portuguese', 'native': 'Português', 'rtl': False, 'voice': 'pt-PT'},
    'ru': {'name': 'Russian', 'native': 'Русский', 'rtl': False, 'voice': 'ru-RU'},
    'zh': {'name': 'Chinese', 'native': '中文', 'rtl': False, 'voice': 'zh-CN'},
    'ja': {'name': 'Japanese', 'native': '日本語', 'rtl': False, 'voice': 'ja-JP'},
    'ko': {'name': 'Korean', 'native': '한국어', 'rtl': False, 'voice': 'ko-KR'},
    'ar': {'name': 'Arabic', 'native': 'العربية', 'rtl': True, 'voice': 'ar-SA'},
    'tr': {'name': 'Turkish', 'native': 'Türkçe', 'rtl': False, 'voice': 'tr-TR'},
    'pl': {'name': 'Polish', 'native': 'Polski', 'rtl': False, 'voice': 'pl-PL'},
    'nl': {'name': 'Dutch', 'native': 'Nederlands', 'rtl': False, 'voice': 'nl-NL'},
    'it': {'name': 'Italian', 'native': 'Italiano', 'rtl': False, 'voice': 'it-IT'},
    'vi': {'name': 'Vietnamese', 'native': 'Tiếng Việt', 'rtl': False, 'voice': 'vi-VN'},
    'th': {'name': 'Thai', 'native': 'ไทย', 'rtl': False, 'voice': 'th-TH'},
    'id': {'name': 'Indonesian', 'native': 'Bahasa Indonesia', 'rtl': False, 'voice': 'id-ID'},
}

def get_language_info(code: str) -> dict:
    """Get language information by code"""
    return SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES['en'])

def is_rtl_language(code: str) -> bool:
    """Check if language is right-to-left"""
    return SUPPORTED_LANGUAGES.get(code, {}).get('rtl', False)

def get_voice_code(code: str) -> str:
    """Get voice synthesis code for language"""
    return SUPPORTED_LANGUAGES.get(code, {}).get('voice', 'en-US')
