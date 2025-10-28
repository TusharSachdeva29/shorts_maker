"""
Main video processing and assembly module
"""

import os
from moviepy.editor import (VideoFileClip, AudioFileClip, concatenate_videoclips,
                            vfx, afx)
from config import (CLIP_DURATION, TRANSITION_DURATION, TARGET_HEIGHT, 
                   MUSIC_LIBRARY, OUTPUT_FILENAME, VIDEO_CODEC, AUDIO_CODEC)


class VideoProcessor:
    """Handles video processing and assembly"""
    
    def __init__(self, video_filter):
        self.video_filter = video_filter
        self.processed_clips = []
    
    def process_single_clip(self, video_path, clip_duration=None, target_height=None):
        """
        Process a single video clip with filters and effects
        
        Args:
            video_path: Path to input video
            clip_duration: Duration to trim clip to (None = use config)
            target_height: Target height for resizing (None = use config)
            
        Returns:
            Processed VideoFileClip
        """
        clip_duration = clip_duration or CLIP_DURATION
        target_height = target_height or TARGET_HEIGHT
        
        clip = VideoFileClip(video_path)
        
        # 1. Trim to desired duration
        start_time = max(0, (clip.duration / 2) - (clip_duration / 2))
        clip = clip.subclip(start_time, start_time + clip_duration)
        
        # 2. Resize to standard height
        clip = clip.resize(height=target_height)
        
        # 3. Apply filter
        filter_func = self.video_filter.get_filter_function()
        print(f"  - Applying filter to '{os.path.basename(video_path)}'...")
        clip = clip.fl_image(filter_func)
        
        # 4. Add fade transitions
        clip = clip.fx(vfx.fadein, TRANSITION_DURATION).fx(vfx.fadeout, TRANSITION_DURATION)
        
        return clip
    
    def process_all_clips(self, video_paths):
        """
        Process all video clips
        
        Args:
            video_paths: List of paths to video files
        """
        print("\nProcessing video clips...")
        self.processed_clips = []
        
        for path in video_paths:
            clip = self.process_single_clip(path)
            self.processed_clips.append(clip)
        
        print("✅ All clips processed.")
    
    def assemble_video(self, music_path=None):
        """
        Assemble all processed clips into final video
        
        Args:
            music_path: Path to background music file
            
        Returns:
            Final VideoFileClip
        """
        print("\nAssembling final video...")
        
        # Concatenate all clips
        final_video = concatenate_videoclips(self.processed_clips, method="compose")
        
        # Add music if provided
        if music_path and os.path.exists(music_path):
            print("Adding background music...")
            main_audio = AudioFileClip(music_path)
            
            # Loop audio to match video duration
            final_video = final_video.set_audio(
                main_audio.fx(afx.audio_loop, duration=final_video.duration)
            )
            
            # Fade out audio at the end
            final_video = final_video.fx(afx.audio_fadeout, TRANSITION_DURATION)
        
        print("✅ Video assembly complete.")
        return final_video
    
    def export_video(self, final_video, output_path=None):
        """
        Export final video to file
        
        Args:
            final_video: VideoFileClip to export
            output_path: Output file path (None = use config)
        """
        output_path = output_path or OUTPUT_FILENAME
        
        print(f"\nExporting video to '{output_path}'...")
        print("This may take several minutes...")
        
        try:
            final_video.write_videofile(
                output_path,
                codec=VIDEO_CODEC,
                audio_codec=AUDIO_CODEC,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            print(f"\n🎉 Success! Video saved to: {output_path}")
            return True
        except Exception as e:
            print(f"\n⛔ Error during export: {e}")
            return False
    
    def select_music(self, emotion, music_dir):
        """
        Select appropriate music file based on detected emotion
        
        Args:
            emotion: Detected emotion string
            music_dir: Directory containing music files
            
        Returns:
            Path to selected music file or None
        """
        music_filename = MUSIC_LIBRARY.get(emotion)
        
        if music_filename:
            music_path = os.path.join(music_dir, music_filename)
            if os.path.exists(music_path):
                print(f"Selected music: '{music_filename}' (emotion: {emotion})")
                return music_path
            else:
                print(f"⚠️ Music file '{music_filename}' not found.")
        
        # Fallback: use any available music file
        if os.path.exists(music_dir):
            music_files = [f for f in os.listdir(music_dir) 
                          if f.endswith(('.mp3', '.wav', '.m4a'))]
            if music_files:
                fallback_path = os.path.join(music_dir, music_files[0])
                print(f"Using fallback music: '{music_files[0]}'")
                return fallback_path
        
        print("⚠️ No music files available.")
        return None
