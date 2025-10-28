"""
Configuration settings for the video editing system
"""

# Model settings
VIDEO_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
DEVICE = "cuda"  # or "cpu"

# Video processing settings
CLIP_DURATION = 3.0  # seconds (set to None to use full clips)
MIN_CLIP_DURATION = 8.0  # Minimum clip duration when using full clips
MAX_CLIP_DURATION = 60.0  # Maximum clip duration when using full clips
TRANSITION_DURATION = 1.0  # seconds
TARGET_HEIGHT = 720  # pixels
USE_STYLE_TRANSFER = False  # Set to True for ML-based style transfer (slower)

# Scene detection settings (for advanced mode)
USE_SCENE_DETECTION = False  # Set to True to use scene detection
TARGET_VIDEO_DURATION = 300  # Target duration in seconds (5 minutes)
MIN_SCENE_DURATION = 3  # Minimum scene duration
MAX_SCENE_DURATION = 30  # Maximum scene duration
SCENE_DETECTION_THRESHOLD = 27.0  # Sensitivity (lower = more scenes detected)

# Output settings
OUTPUT_FILENAME = "cinematic_output.mp4"
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'

# Paths
INPUT_VIDEOS_DIR = "input_videos"
INPUT_MUSIC_DIR = "input_music"
INPUT_STYLE_DIR = "input_style"
OUTPUT_DIR = "output"

# Emotion to music mapping
# Map emotions to your music file names
MUSIC_LIBRARY = {
    "epic": "epic.mp3",
    "calm": "calm.mp3",
    "tense": "tense.mp3",
    "joyful": "joyful.mp3",
    "neutral": "neutral.mp3"
}

# Kinetics-400 action labels to emotion mapping
KINETICS_TO_EMOTION_MAP = {
    # Epic / High-Energy
    "surfing water": "epic", "skiing": "epic", "snowboarding": "epic",
    "playing basketball": "epic", "playing american football": "epic",
    "rock climbing": "epic", "running on treadmill": "epic", "dancing": "epic",
    "parkour": "epic", "skydiving": "epic", "driving car": "epic",
    "riding mechanical bull": "epic", "bungee jumping": "epic",
    "skateboarding": "epic", "motorcycling": "epic", "playing tennis": "epic",

    # Calm / Serene
    "painting": "calm", "reading book": "calm",
    "drinking coffee": "calm", "yoga": "calm", "tai chi": "calm",
    "catching fish": "calm", "sailing": "calm", "sunbathing": "calm",
    "making tea": "calm", "arranging flowers": "calm", "meditation": "calm",

    # Tense / Suspenseful
    "blowing leaves (pile)": "tense", "fencing (sport)": "tense", 
    "archery": "tense", "boxing": "tense",

    # Joyful / Neutral
    "laughing": "joyful", "smiling": "joyful", "hugging": "joyful",
    "playing guitar": "joyful", "cooking": "neutral", "shopping": "neutral",
    "celebrating": "joyful", "playing drums": "joyful", "clapping": "joyful",
}

DEFAULT_EMOTION = "neutral"
