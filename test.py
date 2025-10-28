# Install necessary libraries
!pip install moviepy transformers torch torchvision torchvideo
!pip install -q "transformers[video]"
!pip install scipy

print("✅ Libraries installed.")

# Import all the required modules
import os
import torch
import numpy as np
import cv2
import scipy
from moviepy.editor import *
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from google.colab import files
from collections import Counter # To find the most common emotion

# --- PyTorch Hub for Style Transfer ---
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("✅ Modules imported.")





# --- 1. Upload Videos ---
print("Please upload 2 or more video clips...")
uploaded_videos = files.upload()
video_paths = sorted(list(uploaded_videos.keys()))
print(f"Uploaded videos: {video_paths}")

# --- 2. Upload Music ---
print("\nPlease upload your music tracks (e.g., 'epic.mp3', 'calm.mp3')...")
uploaded_music = files.upload()
music_paths = list(uploaded_music.keys())
print(f"Uploaded music: {music_paths}")

# --- 3. Upload Style Image ---
print("\nPlease upload your single style image (e.g., 'style.jpg')...")
uploaded_style = files.upload()
style_image_path = list(uploaded_style.keys())[0]
print(f"Uploaded style image: {style_image_path}")


from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification



from collections import Counter
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from moviepy.editor import VideoFileClip
import os

# Clear any cached HuggingFace tokens that might be causing issues
os.environ.pop('HF_TOKEN', None)
os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)

# This is a broad mapping from Kinetics-400 labels to emotional themes.
# You can expand this dictionary with more labels.
KINETICS_TO_EMOTION_MAP = {
    # Epic / High-Energy
    "surfing water": "epic", "skiing": "epic", "snowboarding": "epic",
    "playing basketball": "epic", "playing american football": "epic",
    "rock climbing": "epic", "running on treadmill": "epic", "dancing": "epic",
    "parkour": "epic", "skydiving": "epic", "driving car": "epic",
    "riding mechanical bull": "epic", "bungee jumping": "epic",

    # Calm / Serene
    "painting": "calm", "reading book": "calm",
    "drinking coffee": "calm", "yoga": "calm", "tai chi": "calm",
    "catching fish": "calm", "sailing": "calm", "sunbathing": "calm",
    "making tea": "calm", "arranging flowers": "calm",

    # Tense / Suspenseful
    "blowing leaves (pile)": "tense", "walking the dog": "neutral",
    "fencing (sport)": "tense", "archery": "tense",

    # Joyful / Neutral
    "laughing": "joyful", "smiling": "joyful", "hugging": "joyful",
    "playing guitar": "joyful", "cooking": "neutral", "shopping": "neutral",
    "celebrating": "joyful", "playing drums": "joyful",
}

DEFAULT_EMOTION = "neutral"  # Fallback emotion

# --- Load the VideoMAE Model ---
print("Loading VideoMAE model...")
try:
    # Use the correct model name and force no authentication
    video_feature_extractor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        use_auth_token=False
    )
    video_model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        use_auth_token=False
    )
    print("✅ Video model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTrying alternative approach with force_download...")
    # If the above fails, try with force_download to bypass cache
    video_feature_extractor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        use_auth_token=False,
        force_download=True
    )
    video_model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        use_auth_token=False,
        force_download=True
    )
    print("✅ Video model loaded with force_download.")

def get_clip_emotion(video_path):
    """
    Analyzes a video file, gets its action label, and maps it to an emotion.
    """
    clip = VideoFileClip(video_path)

    # Extract 16 frames evenly spaced
    frames = [clip.get_frame((i / 16.0) * clip.duration) for i in range(16)]
    clip.close()

    inputs = video_feature_extractor(list(frames), return_tensors="pt")

    with torch.no_grad():
        outputs = video_model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = video_model.config.id2label[predicted_class_idx]

    # Map the specific label (e.g., "surfing water") to a general emotion (e.g., "epic")
    emotion = KINETICS_TO_EMOTION_MAP.get(predicted_label, DEFAULT_EMOTION)

    print(f"  - Clip '{video_path}': Label='{predicted_label}', Mapped Emotion='{emotion}'")
    return emotion

# --- Analyze all clips and find the dominant emotion ---
print("\nAnalyzing all video clips...")
clip_emotions = [get_clip_emotion(path) for path in video_paths]

# Find the most common emotion
if clip_emotions:
    dominant_emotion = Counter(clip_emotions).most_common(1)[0][0]
else:
    dominant_emotion = DEFAULT_EMOTION

print(f"\n✅ Dominant emotional theme for the video: '{dominant_emotion}'")



# --- !! YOUR ACTION REQUIRED !! ---
# Define your music library by mapping moods to your uploaded files.
# The keys (e.g., "epic", "calm") MUST match the emotions from the map above.

music_library = {
    "epic": "epic.mp3",          # <-- Change this to your file name
    "calm": "calm.mp3",          # <-- Change this to your file name
    "tense": "tense.mp3",        # <-- Change this to your file name
    "joyful": "joyful.mp3",    # <-- Change this to your file name
    "neutral": "neutral.mp3"     # <-- Change this to your file name
}

# --- Logic to select the music ---
selected_music_file = music_library.get(dominant_emotion)

if selected_music_file and selected_music_file in music_paths:
    print(f"Selected music track: '{selected_music_file}'")
    main_audio = AudioFileClip(selected_music_file)
else:
    print(f"⚠️ Warning: Music file for theme '{dominant_emotion}' ('{selected_music_file}') not found.")
    print("Using first available music track as fallback.")
    if music_paths:
        main_audio = AudioFileClip(music_paths[0])
    else:
        print("⛔ Error: No music files were uploaded. Audio will be silent.")
        main_audio = None




# --- !! ACTION REQUIRED !! ---
# Set this to True to use the slow, high-quality ML filter.
# Set this to False to use the fast, simple OpenCV filter.
USE_STYLE_TRANSFER = False
# -----------------------------

# --- Set up device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Helper functions for image loading and processing ---
def load_image(filename, size=None):
    img = Image.open(filename).convert('RGB')
    if size:
        img = img.resize((size, size), Image.ANTIALIAS)

    # Pre-process for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)) # Model expects 0-255 range
    ])
    img = transform(img).unsqueeze(0)
    return img.to(device)

def post_process_image(tensor):
    img = tensor.clone().detach().squeeze(0)
    img = img.cpu().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8") # Convert from CHW to HWC
    return img

# --- Define the Style Transfer Model Architecture ---
# (This is a standard implementation of a fast NST model)
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = F.relu(self.in2(self.conv2(y)))
        y = F.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = F.relu(self.in4(self.deconv1(y)))
        y = F.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

# --- Global variable for the style model ---
style_model = None

if USE_STYLE_TRANSFER:
    print("\nLoading Neural Style Transfer model...")
    # --- Download a pre-trained model (e.g., "mosaic") ---
    # We will load the pre-trained weights from a saved file
    # This is a common way to do it in Colab
    !wget -q https://storage.googleapis.com/download.tensorflow.org/models/tflite/art_transfer/style_predict_f16_640.tflite -O style_predict.tflite
    !wget -q https://github.com/pytorch/examples/raw/main/fast_neural_style/models/mosaic.pth

    style_model = TransformerNet()
    style_model.load_state_dict(torch.load("mosaic.pth"))
    style_model.to(device)
    style_model.eval()
    print("✅ NST model loaded ('mosaic' style).")

    # This is the function that moviepy will call for EACH frame
    def apply_style_transfer(frame):
        # Convert frame (numpy array) to torch tensor
        img_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)
        img_tensor = img_tensor.mul(255) # Model expects 0-255

        with torch.no_grad():
            styled_tensor = style_model(img_tensor)

        # Convert back to numpy array
        styled_frame = post_process_image(styled_tensor)
        return styled_frame

else:
    print("\nUsing fast OpenCV filter (Style Transfer is OFF).")
    # This is our fast, simple filter from the first notebook
    def apply_cinematic_filter_opencv(frame):
        # 1. Increase contrast
        alpha = 1.1 # Contrast
        beta = 5    # Brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # 2. Apply a "cool" tint
        b, g, r = cv2.split(adjusted)
        b = cv2.add(b, 15)
        r = cv2.subtract(r, 10)
        final_frame = cv2.merge((b, g, r))

        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
        return final_frame
    

print("\nStarting final video assembly...")

processed_clips = []
clip_duration = 3.0  # Let's standardize clip length to 3 seconds for this demo
transition_duration = 1.0  # 1-second fade

print("Processing clips (applying filters and resizing)...")

for i, path in enumerate(video_paths):
    clip = VideoFileClip(path)

    # 1. Cut/Resize all clips to be the same length
    start_time = max(0, (clip.duration / 2) - (clip_duration / 2))
    clip = clip.subclip(start_time, start_time + clip_duration)

    # 2. Resize all clips to a standard 720p
    clip = clip.resize(height=720)

    # 3. Apply the selected filter to every frame
    if USE_STYLE_TRANSFER:
        print(f"  - Applying ML Style Transfer to '{path}'... (This will be SLOW)")
        clip = clip.fl_image(apply_style_transfer)
    else:
        print(f"  - Applying fast OpenCV filter to '{path}'...")
        clip = clip.fl_image(apply_cinematic_filter_opencv)

    # 4. Add fade in/out to all clips for smooth transitions
    clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)

    processed_clips.append(clip)

print("✅ All clips processed.")

# --- 5. Simple concatenation ---
final_video = concatenate_videoclips(processed_clips, method="compose")

# --- 6. Add the music ---
if main_audio:
    print("Adding music...")
    # Loop the audio to match the video's total duration
    final_video = final_video.set_audio(
        main_audio.fx(afx.audio_loop, duration=final_video.duration)
    )
    # Fade out the audio at the end
    final_video = final_video.fx(afx.audio_fadeout, transition_duration)

print("✅ Assembly complete.")




output_filename = "cinematic_output_advanced.mp4"
print(f"\nWriting final video file: {output_filename}")
print("This may take a few minutes...")

try:
    # Use 'libx264' for video (good quality/compression) and 'aac' for audio
    final_video.write_videofile(
        output_filename,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    print(f"\n🎉🎉🎉 Success! Your video is ready. 🎉🎉🎉")

    # Offer the file for download
    files.download(output_filename)

except Exception as e:
    print(f"\n⛔ An error occurred during file writing: {e}")
    print("This can sometimes happen in Colab. Try re-running the cell or using a different codec (e.g., 'mpeg4').")