"""
Emotion analysis module for video clips
"""

import os
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from moviepy.editor import VideoFileClip
from collections import Counter
from config import VIDEO_MODEL_NAME, KINETICS_TO_EMOTION_MAP, DEFAULT_EMOTION


class EmotionAnalyzer:
    """Analyzes video clips to determine emotional content"""
    
    def __init__(self):
        # Clear any cached HuggingFace tokens
        os.environ.pop('HF_TOKEN', None)
        os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
        
        self.feature_extractor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the VideoMAE model for action recognition"""
        print("Loading VideoMAE model...")
        try:
            self.feature_extractor = VideoMAEImageProcessor.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False
            )
            self.model = VideoMAEForVideoClassification.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False
            )
            print("✅ Video model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Trying with force_download...")
            self.feature_extractor = VideoMAEImageProcessor.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False,
                force_download=True
            )
            self.model = VideoMAEForVideoClassification.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False,
                force_download=True
            )
            print("✅ Video model loaded with force_download.")
    
    def get_clip_emotion(self, video_path, num_frames=16):
        """
        Analyze a video clip to determine its emotional content
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample for analysis
            
        Returns:
            Detected emotion string (e.g., 'epic', 'calm', 'joyful')
        """
        clip = VideoFileClip(video_path)
        
        # Extract frames evenly spaced throughout the video
        frames = [clip.get_frame((i / num_frames) * clip.duration) 
                  for i in range(num_frames)]
        clip.close()
        
        # Process frames through the model
        inputs = self.feature_extractor(list(frames), return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get predicted action label
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class_idx]
        
        # Map action label to emotion
        emotion = KINETICS_TO_EMOTION_MAP.get(predicted_label, DEFAULT_EMOTION)
        
        print(f"  - '{video_path}': Label='{predicted_label}', Emotion='{emotion}'")
        return emotion
    
    def analyze_clips(self, video_paths):
        """
        Analyze multiple video clips and determine dominant emotion
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            Tuple of (dominant_emotion, all_emotions_list)
        """
        print("\nAnalyzing video clips for emotional content...")
        clip_emotions = [self.get_clip_emotion(path) for path in video_paths]
        
        # Find the most common emotion
        if clip_emotions:
            dominant_emotion = Counter(clip_emotions).most_common(1)[0][0]
        else:
            dominant_emotion = DEFAULT_EMOTION
        
        print(f"\n✅ Dominant emotion: '{dominant_emotion}'")
        return dominant_emotion, clip_emotions
