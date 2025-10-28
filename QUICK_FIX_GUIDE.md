# 🔧 Quick Fix Implementation Guide

## CRITICAL FIX: Video Duration (5-10 minutes requirement)

This guide shows you EXACTLY what to change to fix the video duration issue.

---

## Option 1: Quick Fix (30 minutes) - Use Full Clips

### Step 1: Modify `config.py`

**Find these lines** (around line 13-14):

```python
CLIP_DURATION = 3.0  # seconds
TRANSITION_DURATION = 1.0  # seconds
```

**Change to**:

```python
CLIP_DURATION = None  # Use full clip or max duration
MIN_CLIP_DURATION = 8.0  # Minimum 8 seconds per clip
MAX_CLIP_DURATION = 60.0  # Maximum 60 seconds per clip
TRANSITION_DURATION = 1.0  # seconds
```

---

### Step 2: Modify `video_processor.py`

**Find this function** (around line 18):

```python
def process_single_clip(self, video_path, clip_duration=None, target_height=None):
    """
    Process a single video clip with filters and effects
    ...
    """
    clip_duration = clip_duration or CLIP_DURATION
    target_height = target_height or TARGET_HEIGHT

    clip = VideoFileClip(video_path)

    # 1. Trim to desired duration
    start_time = max(0, (clip.duration / 2) - (clip_duration / 2))
    clip = clip.subclip(start_time, start_time + clip_duration)
```

**Replace with**:

```python
def process_single_clip(self, video_path, clip_duration=None, target_height=None):
    """
    Process a single video clip with filters and effects
    ...
    """
    from config import MIN_CLIP_DURATION, MAX_CLIP_DURATION

    target_height = target_height or TARGET_HEIGHT
    clip = VideoFileClip(video_path)

    # 1. Determine clip duration
    if clip_duration is None:
        # Use CLIP_DURATION from config if set, otherwise use full clip
        clip_duration = CLIP_DURATION

    if clip_duration is None:
        # Use as much of the clip as possible (up to MAX)
        clip_duration = min(clip.duration, MAX_CLIP_DURATION)
        clip_duration = max(clip_duration, MIN_CLIP_DURATION)

        # Use the full clip if it's within bounds
        if clip.duration <= MAX_CLIP_DURATION:
            # Use entire clip - no trimming
            pass
        else:
            # Clip is too long, use best segment
            start_time = max(0, (clip.duration / 2) - (clip_duration / 2))
            clip = clip.subclip(start_time, start_time + clip_duration)
    else:
        # Use specified duration from center of clip
        start_time = max(0, (clip.duration / 2) - (clip_duration / 2))
        clip = clip.subclip(start_time, start_time + clip_duration)
```

---

### Step 3: Test

```powershell
# Add 5-6 video clips of 60 seconds each to input_videos/
# Run the system
python main.py

# Expected output: ~5 minutes (5 clips × 60 seconds)
```

**That's it!** You now have 5-10 minute videos. ✅

---

## Option 2: Better Fix (3-4 hours) - Scene Detection

This is higher quality but takes longer to implement.

### Step 1: Install SceneDetect

```powershell
pip install scenedetect[opencv]
```

### Step 2: Create new file `scene_detector.py`

```python
"""
Scene detection and extraction module
"""

from scenedetect import detect, ContentDetector
from moviepy.editor import VideoFileClip
import numpy as np


class SceneDetector:
    """Detects and scores scenes in videos"""

    def __init__(self, threshold=27.0):
        self.threshold = threshold

    def detect_scenes(self, video_path):
        """
        Detect scene boundaries in a video

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        scene_list = detect(video_path, ContentDetector(threshold=self.threshold))

        # Convert to seconds
        scenes = []
        for start_time, end_time in scene_list:
            scenes.append((
                start_time.get_seconds(),
                end_time.get_seconds()
            ))

        return scenes

    def score_scene(self, video_path, start_time, end_time):
        """
        Score a scene based on:
        1. Duration (prefer 5-20 seconds)
        2. Motion (more motion = more interesting)
        3. Brightness (not too dark)

        Returns:
            Score between 0 and 1
        """
        clip = VideoFileClip(video_path).subclip(start_time, end_time)
        duration = end_time - start_time

        # Duration score (prefer 5-20 seconds)
        if 5 <= duration <= 20:
            duration_score = 1.0
        elif duration < 5:
            duration_score = duration / 5.0  # Penalize short scenes
        else:
            duration_score = 20.0 / duration  # Penalize long scenes

        # Motion score (sample a few frames)
        motion_score = self._calculate_motion(clip)

        # Brightness score
        brightness_score = self._calculate_brightness(clip)

        clip.close()

        # Weighted combination
        total_score = (
            duration_score * 0.3 +
            motion_score * 0.4 +
            brightness_score * 0.3
        )

        return total_score

    def _calculate_motion(self, clip):
        """Calculate motion score from frame differences"""
        # Sample 5 frames
        sample_times = np.linspace(0, clip.duration, 6)[:-1]

        motion_scores = []
        prev_frame = None

        for t in sample_times:
            frame = clip.get_frame(t)

            if prev_frame is not None:
                # Calculate frame difference
                diff = np.abs(frame.astype(float) - prev_frame.astype(float))
                motion = np.mean(diff) / 255.0
                motion_scores.append(motion)

            prev_frame = frame

        # Average motion (normalize to 0-1)
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        # Scale: 0.05 is typical, 0.15 is high motion
        return min(avg_motion / 0.15, 1.0)

    def _calculate_brightness(self, clip):
        """Calculate brightness score"""
        # Sample middle frame
        frame = clip.get_frame(clip.duration / 2)

        # Convert to grayscale
        gray = np.mean(frame, axis=2)
        avg_brightness = np.mean(gray) / 255.0

        # Prefer brightness between 0.3 and 0.7
        if 0.3 <= avg_brightness <= 0.7:
            return 1.0
        elif avg_brightness < 0.3:
            # Too dark
            return avg_brightness / 0.3
        else:
            # Too bright
            return (1.0 - avg_brightness) / 0.3

    def extract_best_scenes(self, video_paths, target_duration=300,
                           min_scene_duration=3, max_scene_duration=30):
        """
        Extract best scenes from multiple videos

        Args:
            video_paths: List of video file paths
            target_duration: Target total duration in seconds (default 5 minutes)
            min_scene_duration: Minimum scene duration in seconds
            max_scene_duration: Maximum scene duration in seconds

        Returns:
            List of (video_path, start_time, end_time, score) tuples
        """
        all_scenes = []

        print("\nDetecting and scoring scenes...")

        for video_path in video_paths:
            print(f"  - Processing '{video_path}'...")

            # Detect scenes
            scenes = self.detect_scenes(video_path)

            # Score each scene
            for start_time, end_time in scenes:
                duration = end_time - start_time

                # Filter by duration
                if min_scene_duration <= duration <= max_scene_duration:
                    score = self.score_scene(video_path, start_time, end_time)
                    all_scenes.append((video_path, start_time, end_time, score))

        # Sort by score (best first)
        all_scenes.sort(key=lambda x: x[3], reverse=True)

        print(f"✅ Found {len(all_scenes)} quality scenes")

        # Select scenes until we reach target duration
        selected_scenes = []
        total_duration = 0

        for scene in all_scenes:
            video_path, start_time, end_time, score = scene
            duration = end_time - start_time

            if total_duration < target_duration:
                selected_scenes.append(scene)
                total_duration += duration
                print(f"  - Selected scene: {duration:.1f}s (score: {score:.2f})")
            else:
                break

        print(f"\n✅ Selected {len(selected_scenes)} scenes ({total_duration:.1f}s total)")

        return selected_scenes
```

---

### Step 3: Modify `video_processor.py`

**Add this import at the top**:

```python
from scene_detector import SceneDetector
```

**Add this new method to the `VideoProcessor` class**:

```python
def process_clips_with_scene_detection(self, video_paths, target_duration=300):
    """
    Process videos using scene detection

    Args:
        video_paths: List of video file paths
        target_duration: Target total duration in seconds
    """
    print("\n" + "="*60)
    print("Using Scene Detection Mode")
    print("="*60)

    # Detect and select best scenes
    detector = SceneDetector()
    selected_scenes = detector.extract_best_scenes(
        video_paths,
        target_duration=target_duration
    )

    # Process each selected scene
    print("\nProcessing selected scenes...")
    self.processed_clips = []

    for video_path, start_time, end_time, score in selected_scenes:
        clip = VideoFileClip(video_path).subclip(start_time, end_time)

        # Resize
        clip = clip.resize(height=TARGET_HEIGHT)

        # Apply filter
        filter_func = self.video_filter.get_filter_function()
        print(f"  - Applying filter to scene from '{video_path}'...")
        clip = clip.fl_image(filter_func)

        # Add transitions
        clip = clip.fx(vfx.fadein, TRANSITION_DURATION).fx(vfx.fadeout, TRANSITION_DURATION)

        self.processed_clips.append(clip)

    print("✅ All scenes processed.")
```

---

### Step 4: Modify `main.py`

**Find this section** (around line 60):

```python
# Step 3: Process videos
print("\n" + "=" * 60)
print("STEP 3: Video Processing")
print("=" * 60)
processor = VideoProcessor(video_filter)
processor.process_all_clips(video_paths)
```

**Replace with**:

```python
# Step 3: Process videos
print("\n" + "=" * 60)
print("STEP 3: Video Processing")
print("=" * 60)
processor = VideoProcessor(video_filter)

# Choose processing mode
USE_SCENE_DETECTION = True  # Set to False to use old method

if USE_SCENE_DETECTION:
    # Use scene detection for better quality
    processor.process_clips_with_scene_detection(video_paths, target_duration=300)
else:
    # Use simple clip processing
    processor.process_all_clips(video_paths)
```

---

### Step 5: Add to `config.py`

```python
# Scene detection settings
USE_SCENE_DETECTION = True
TARGET_VIDEO_DURATION = 300  # 5 minutes
MIN_SCENE_DURATION = 3
MAX_SCENE_DURATION = 30
SCENE_DETECTION_THRESHOLD = 27.0
```

---

### Step 6: Test

```powershell
python main.py
```

**Expected output**: High-quality 5-minute video with best scenes from all input videos! ✅✅✅

---

## Which Option to Choose?

### Choose Option 1 if:

- ✅ You need quick fix (30 minutes)
- ✅ You have 5-6 good quality full clips
- ✅ You want simple, reliable solution

### Choose Option 2 if:

- ✅ You have time (3-4 hours)
- ✅ You want best quality output
- ✅ You want to impress judges with sophistication
- ✅ Your input clips are long and varied

---

## Testing Checklist

After implementing either option:

- [ ] Video output is 5-10 minutes long
- [ ] Music loops properly for entire duration
- [ ] No audio glitches or cuts
- [ ] Transitions are smooth
- [ ] Video quality is good (720p+)
- [ ] Emotions are detected correctly
- [ ] Music matches emotional theme
- [ ] Processing completes in reasonable time
- [ ] No errors or crashes

---

## Need Help?

1. **Read**: `IMPROVEMENTS_SUGGESTIONS.md` for more details
2. **Check**: `HACKATHON_REQUIREMENTS.md` for context
3. **Review**: `README.md` for usage guide

**Good luck! 🚀**
