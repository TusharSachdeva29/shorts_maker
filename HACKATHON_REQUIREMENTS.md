# 🎯 Hackathon Requirements Analysis

## Official Requirements Breakdown

### Round 1 Requirements (Online Submission)

✅ **Team Size**: 2-3 members
✅ **Themes**: ML, Computer Vision, Automation
✅ **Submission Format**:

- 6-slide PPT (excluding title & conclusion) ✅
- GitHub link with collaborator access ✅
- Demo video ✅
- PDF format ✅

### Round 2 Requirements (Offline at NSUT)

✅ **Model Capabilities**:

- Accept videos from local folder path ✅
- Automatically process ✅
- Export final edited video (5-10 minutes) ⚠️ **NEEDS WORK**

✅ **Technical Constraints**:

- GPU access: NVIDIA A100
- 60 min runtime limit
- Live demonstration (7-10 min presentation)

---

## ✅ Current Implementation Status

### What's Working Well:

1. ✅ Takes multiple raw video clips as input
2. ✅ Applies cinematic transitions (fade in/out)
3. ✅ Applies filters (OpenCV + optional ML style transfer)
4. ✅ Adds background music based on scene context (emotion detection)
5. ✅ Outputs polished video
6. ✅ ML-based (VideoMAE for emotion recognition)
7. ✅ Computer Vision (video processing, filters)
8. ✅ Automated (no manual intervention needed)
9. ✅ Works with local folder paths
10. ✅ GPU accelerated (CUDA support)

### ⚠️ Critical Issue - Video Duration

**PROBLEM**: Current output is only **9-12 seconds** for 3-4 clips
**REQUIREMENT**: Must output **5-10 minutes** (300-600 seconds)

**Gap**: You need ~25-50x more video duration!

---

## 🚨 Must-Fix Before Hackathon

### 1. Video Duration Issue (CRITICAL)

#### Quick Solution Options:

**Option A: Use Full Clips (Easiest)**

```python
# In config.py
CLIP_DURATION = None  # Use full clip duration
MIN_CLIP_DURATION = 10.0  # Minimum 10 seconds per clip
MAX_CLIP_DURATION = 60.0  # Maximum 60 seconds per clip

# In video_processor.py - modify process_single_clip
def process_single_clip(self, video_path, clip_duration=None, target_height=None):
    clip = VideoFileClip(video_path)

    if clip_duration is None:
        # Use full clip or cap at MAX_CLIP_DURATION
        clip_duration = min(clip.duration, MAX_CLIP_DURATION)

    # Don't trim if we want full duration
    if clip.duration <= MAX_CLIP_DURATION:
        # Use entire clip
        pass
    else:
        # Use best segment
        start_time = max(0, (clip.duration / 2) - (clip_duration / 2))
        clip = clip.subclip(start_time, start_time + clip_duration)

    # Rest of processing...
```

**Estimated time**: 30 minutes
**Result**: If you have 5 clips of 60 seconds each = 5 minutes ✅

---

**Option B: Repeat Clips with Variations (Medium)**

```python
def extend_video_duration(self, processed_clips, target_duration=300):
    """
    Extend video to target duration by:
    1. Repeating clips with different filters
    2. Using different segments of same clip
    3. Slow motion variations
    """
    extended_clips = []
    current_duration = sum(c.duration for c in processed_clips)

    while current_duration < target_duration:
        for original_clip in processed_clips:
            if current_duration >= target_duration:
                break

            # Create variation
            variation = self.create_clip_variation(original_clip)
            extended_clips.append(variation)
            current_duration += variation.duration

    return processed_clips + extended_clips

def create_clip_variation(self, clip):
    """Create variation of clip (different speed, different segment, etc.)"""
    # Try different segment
    # Or apply slow motion
    # Or different filter intensity
    return varied_clip
```

**Estimated time**: 2 hours
**Result**: Reaches 5-10 minutes but with some repetition

---

**Option C: Scene Detection & Intelligent Extraction (Best Quality)**

```python
# Install: pip install scenedetect[opencv]
from scenedetect import detect, ContentDetector

def extract_all_good_scenes(video_paths, min_duration=5, max_duration=30):
    """
    Extract ALL good scenes from videos, not just center segment
    """
    all_scenes = []

    for video_path in video_paths:
        # Detect all scenes in video
        scene_list = detect(video_path, ContentDetector(threshold=27.0))

        for i, (start_time, end_time) in enumerate(scene_list):
            duration = (end_time - start_time).get_seconds()

            # Keep scenes between min and max duration
            if min_duration <= duration <= max_duration:
                scene_clip = VideoFileClip(video_path).subclip(
                    start_time.get_seconds(),
                    end_time.get_seconds()
                )

                # Score scene quality
                score = score_scene_quality(scene_clip)
                all_scenes.append((scene_clip, score))

    # Sort by quality and keep best scenes
    all_scenes.sort(key=lambda x: x[1], reverse=True)

    # Take enough scenes to reach 5-10 minutes
    selected_scenes = []
    total_duration = 0
    target_duration = 300  # 5 minutes minimum

    for scene, score in all_scenes:
        if total_duration < target_duration or total_duration < 600:  # max 10 min
            selected_scenes.append(scene)
            total_duration += scene.duration

    return selected_scenes
```

**Estimated time**: 3-4 hours
**Result**: High quality 5-10 minute video with best content from all source videos ✅✅✅

---

### 2. Music Duration Matching

```python
# Ensure music loops properly for longer videos
def add_music_with_proper_looping(self, final_video, music_path):
    """
    For videos longer than music:
    1. Loop music
    2. Add fade transitions between loops
    3. Fade out at end
    """
    if not music_path:
        return final_video

    music = AudioFileClip(music_path)
    video_duration = final_video.duration

    if music.duration < video_duration:
        # Need to loop
        num_loops = int(video_duration / music.duration) + 1

        # Create smooth loops with crossfade
        looped_audio = music
        for i in range(num_loops - 1):
            # Add crossfade between loops (2 seconds)
            looped_audio = afx.audio_loop(looped_audio, duration=video_duration)

        final_video = final_video.set_audio(looped_audio)
    else:
        # Music is longer than video
        final_video = final_video.set_audio(music.subclip(0, video_duration))

    # Fade out at end
    final_video = final_video.fx(afx.audio_fadeout, 2.0)

    return final_video
```

---

## 📊 Recommended Implementation Plan

### Before Submission (26th October)

#### Day 1 (4-5 hours):

1. ⏱️ **Fix video duration** (Option C - Scene Detection)

   - Install scenedetect
   - Implement scene extraction
   - Test with sample videos
   - **Goal**: 5-10 minute output

2. 🎵 **Improve music handling**
   - Add 2-3 songs per emotion
   - Implement proper looping
   - Test with long videos

#### Day 2 (3-4 hours):

3. 🎭 **Enhance emotion detection**

   - Add more Kinetics-400 labels to emotion map
   - Test accuracy with event footage
   - Fine-tune mappings

4. 📹 **Test with real event footage**
   - Use provided dataset
   - Test full pipeline
   - Verify 5-10 minute requirement

#### Day 3 (2-3 hours):

5. 📊 **Create presentation**

   - 6 slides (architecture, emotion detection, results, demo)
   - Record demo video
   - Prepare GitHub repo

6. 🐛 **Bug fixes and polish**
   - Error handling
   - Progress indicators
   - Clean code

---

## 🎥 Presentation Outline (6 Slides)

### Slide 1: Problem Statement

- Challenge: Manual video editing is time-consuming
- Solution: Automated ML-based cinematic editor
- Key insight: Emotion-aware music selection

### Slide 2: System Architecture

- Input: Multiple raw clips
- Process: Emotion detection → Scene selection → Filter application → Music matching
- Output: 5-10 min cinematic video

### Slide 3: Emotion Detection (ML Component)

- VideoMAE model (Kinetics-400)
- Action → Emotion mapping
- 5 emotion categories
- Example detections

### Slide 4: Video Processing Pipeline

- Scene detection & quality scoring
- Cinematic filters (OpenCV/Neural Style Transfer)
- Smooth transitions
- Duration control (5-10 minutes)

### Slide 5: Results & Demo

- Before/After comparisons
- Emotion detection accuracy
- Sample outputs
- Processing time metrics

### Slide 6: Innovation & Future Work

- Innovations:
  - Context-aware music selection
  - Quality-based scene selection
  - Automated full pipeline
- Future: Beat-sync, multi-modal emotion, text overlays

---

## 🎯 Demo Video Script (3-4 minutes)

1. **Introduction** (30s)

   - Problem statement
   - Our solution

2. **How It Works** (60s)

   - Show folder structure
   - Run command
   - Show processing steps

3. **Results** (90s)

   - Show input clips
   - Show output video
   - Highlight features (transitions, music, emotion detection)

4. **Technical Details** (30s)
   - ML models used
   - Key algorithms
   - Performance metrics

---

## 🏆 Competitive Advantages

Your current strengths:

1. ✅ Clean, modular code
2. ✅ Good ML model choice (VideoMAE)
3. ✅ Professional transitions
4. ✅ Configurable and extensible
5. ✅ GPU accelerated

Add these to stand out:

1. 🎯 **Scene quality scoring** (not just random clips)
2. 🎵 **Intelligent music selection** (emotion-based)
3. 📊 **Automatic duration control** (meets 5-10 min requirement)
4. 🔄 **Scene detection** (natural segment boundaries)
5. 💾 **Efficient processing** (under 60 min on A100)

---

## ⚡ Quick Win Checklist

Before submission:

- [ ] Video duration reaches 5-10 minutes ✅
- [ ] Scene detection implemented
- [ ] Music loops properly for long videos
- [ ] Tested with dataset videos
- [ ] Error handling for edge cases
- [ ] GitHub repo clean and documented
- [ ] Demo video recorded (< 5 min)
- [ ] PPT created (6 slides + title + conclusion)
- [ ] Added Innovisionnsut48 as collaborator

---

## 💡 Pro Tips

1. **Test Early**: Test with actual event footage from dataset NOW
2. **Time Management**: Focus on duration fix first (most critical)
3. **Demo Quality**: Make demo video high quality (judges watch this carefully)
4. **Edge Cases**: Handle errors gracefully (missing files, corrupt videos, etc.)
5. **Performance**: Optimize to finish under 60 minutes on A100
6. **Documentation**: Clean README helps judges understand your work

---

## 📞 Final Recommendations

### Must Do (Critical):

1. Fix video duration to 5-10 minutes (Scene detection approach)
2. Test with event footage from dataset
3. Ensure music loops properly

### Should Do (Important):

4. Add more music variety (2-3 per emotion)
5. Improve emotion detection accuracy
6. Add progress indicators
7. Error handling

### Nice to Have (Time Permitting):

8. Color grading per emotion
9. Smart clip ordering
10. Text overlays

**Focus on the MUST DO items first!** These are blocking issues for the hackathon requirements.

Good luck! 🚀🎉
