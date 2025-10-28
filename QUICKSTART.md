# Quick Setup Guide

## Installation Steps

1. **Create Virtual Environment**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Setup Folders**

   - Place videos in `input_videos/`
   - Place music in `input_music/` (epic.mp3, calm.mp3, etc.)

4. **Run**
   ```powershell
   python main.py
   ```

## Expected Runtime (for 3-4 clips on GPU)

- With `USE_STYLE_TRANSFER = False`: ~5-10 minutes
- With `USE_STYLE_TRANSFER = True`: ~30-60 minutes

## Output

- Check `output/cinematic_output.mp4`
