[Demo of Generated Video](https://youtu.be/HHNtKrzMp5Y)

-----------------------------------------

# Video Automation and Upload Script

This repository contains a comprehensive script designed to automate the creation and upload of video content to YouTube. The script performs several tasks including text-to-speech conversion, image processing, video editing, and interaction with the YouTube Data API for video upload.

## Features

1. **Text-to-Speech Conversion**:
    - Converts script text to audio using a specified TTS model.
2. **Image Processing**:
    - Downloads images based on search queries.
    - Resizes and processes images for use in videos.
3. **Video Creation**:
    - Generates background and foreground videos using processed images.
    - Creates animation videos.
    - Combines audio and video clips.
    - Adds subtitles to videos.
4. **YouTube Upload**:
    - Authenticates YouTube channel.
    - Uploads videos to YouTube.
    - Sets video thumbnail and metadata.
    - Adds videos to specific playlists.

## Requirements

### Python Libraries

- `collections`
- `faster_whisper`
- `google-auth`
- `moviepy`
- `PIL`
- `pydub`
- `ffmpeg`
- `google_auth_oauthlib`
- `googleapiclient`
- `hashlib`
- `imagehash`
- `numpy`
- `openai`
- `paramiko`
- `pickle`
- `random`
- `re`
- `requests`
- `shutil`
- `subprocess`

### External Tools

- `FFmpeg`
- `Google Images Search API`
- `OpenAI API`
- `Pushover API` (for notifications)

## Setup

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API Keys and Credentials**:
    - Place your OpenAI API key in `openai_key.txt`.
    - Place your Google Images API keys in `googleimages_key.txt`.
    - Configure YouTube API client secrets in the specified path.
    - Configure Pushover credentials in `pushover.txt`.

4. **Set Up Directory Structure**:
    - Ensure the necessary directories exist within the main directory for images, audio, and workspace.

## Usage

Run the main script with the specified channel name:

```bash
python main.py <channel_name>
```

## Directory Structure

The script expects the following directory structure:

```
<main_directory>/
│
├── <channel_name>/
│   ├── Images/
│   ├── Uncropped Images/
│   ├── Audio/
│   ├── Music/
│   ├── Workplace/
│   ├── topics.txt
│   ├── description.txt
│   └── client_secret_*.json
├── openai_key.txt
├── googleimages_key.txt
├── pushover.txt
├── ffmpeg
└── AutomatedYoutube.py
```

### Example

To run the script for the "HistorysDarkestQuestions" channel:

```
python main.py HistorysDarkestQuestions
```

## Function Descriptions

### Text-to-Speech Conversion
- text_to_audio_file: Converts text to audio using the specified TTS model.

### Image Processing
- resize_image: Resizes an image to fit within specified dimensions.
- resize_and_blur_background: Creates a blurred background for an image.
- find_and_convert_images_to_clips: Converts images in a directory to video clips.

### Video Creation
- create_video_with_audio: Creates a video from images and an audio file.
- overlay_videos: Overlays one video onto another.
- create_animation_video: Creates an animation video.

### YouTube Upload
- authenticate_channel: Authenticates YouTube channel using OAuth.
- upload_video: Uploads a video to YouTube.
- set_thumbnail: Sets the thumbnail for a YouTube video.

## Error Handling

The script includes extensive error handling and logging. In case of an error, a notification is sent via Pushover, and detailed logs are saved.