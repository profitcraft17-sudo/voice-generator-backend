import os

# ==========================================
# CORE ENGINE SYSTEM CONFIGURATIONS
# ==========================================

# Paths setup matching the frozen folder structure exactly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default values for RVC Inference (Optimal for natural tone)
DEFAULT_PITCH_CHANGE = 0      # 0 = Original sample pitch, (+) keys increase pitch, (-) keys deepens pitch
DEFAULT_F0_METHOD = "rmvpe"   # 'rmvpe' is the best and cleanest method for vocals/singing/speaking
DEFAULT_INDEX_RATE = 0.75     # Controls how much feature index identity to apply (0.0 to 1.0)
DEFAULT_VOLUME_ENVELOPE = 0.25# Protects against audio distortion and clipping

# ==========================================
# SMART DYNAMIC VOICE PROFILES
# ==========================================
# Har model folder ke liye dynamic internal adjustments
VOICE_PROFILES = {
    "cute_boy": {
        "folder_name": "cute_boy",
        "pitch_shift": 1,          # Slight positive shift (+1 or +2) to make the voice sound younger/sweeter
        "f0_method": "rmvpe",      # High-fidelity tracking
        "index_rate": 0.80,        # Higher index brings more original character identity
        "protect_voiceless": 0.33  # Protects silent/whisper portions from robotic glitching
    }
}

def get_voice_config(model_name: str) -> dict:
    """
    Fetches custom parameters for a selected model.
    If model doesn't exist, safely fallbacks to standard defaults.
    """
    if model_name in VOICE_PROFILES:
        return VOICE_PROFILES[model_name]
    
    # Secure Fallback Mechanism
    return {
        "folder_name": model_name,
        "pitch_shift": DEFAULT_PITCH_CHANGE,
        "f0_method": DEFAULT_F0_METHOD,
        "index_rate": DEFAULT_INDEX_RATE,
        "protect_voiceless": DEFAULT_VOLUME_ENVELOPE
}
