from collections import defaultdict
from faster_whisper import WhisperModel
from google.auth.transport.requests import Request
from moviepy.editor import *
from moviepy.video.io import ffmpeg_writer
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, UnidentifiedImageError,ImageFont
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from random import randint
from time import sleep
import ffmpeg
import glob
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import hashlib
import imagehash
import math
import numpy as np
import openai
import os
import paramiko
import pickle
import random
import re
import requests
import shutil
import subprocess

def resize_image(img, output_width=1920, output_height=1080):

    min_width = output_width*0.5
    min_height = output_height*0.5
    max_width = output_width*0.7
    max_height = output_height*0.7

    original_width, original_height = img.size

    # Check if the image dimensions are within the maximum dimensions
    if original_width <= max_width and original_height <= max_height:
        if original_width >= min_width or original_height >= min_height:
            foreground_img = img
        else:
            # Enlarge the image if it's below the minimum dimensions
            width_ratio = min_width / original_width if original_width < min_width else 1
            height_ratio = min_height / original_height if original_height < min_height else 1
            resize_ratio = max(width_ratio, height_ratio)
            
            new_width = int(original_width * resize_ratio)
            new_height = int(original_height * resize_ratio)
            foreground_img = img.resize((new_width, new_height), Image.LANCZOS)
    else:
        # Calculate the resize ratio to ensure the image is not larger than the maximum allowed size
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        resize_ratio = min(width_ratio, height_ratio)

        # Calculate the new dimensions
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)

        # Check if the resized dimensions are less than the minimum dimensions and adjust if necessary
        if new_width < min_width or new_height < min_height:
            width_ratio = min_width / new_width if new_width < min_width else 1
            height_ratio = min_height / new_height if new_height < min_height else 1
            additional_ratio = max(width_ratio, height_ratio)

            new_width = int(new_width * additional_ratio)
            new_height = int(new_height * additional_ratio)

        foreground_img = img.resize((new_width, new_height), Image.LANCZOS)

    return foreground_img

def resize_and_blur_background(img_path):
    img = Image.open(img_path)
    original_aspect_ratio = img.width / img.height

    # Desired output dimensions
    output_width = 1920
    output_height = 1080
    output_aspect_ratio = output_width / output_height

    # Calculate new dimensions for the image
    if original_aspect_ratio > output_aspect_ratio:
        # Image is wider than target aspect ratio
        new_height = output_height
        new_width = int(original_aspect_ratio * new_height)
    else:
        # Image is taller than target aspect ratio or equal
        new_width = output_width
        new_height = int(new_width / original_aspect_ratio)

    # Resize image to new dimensions
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create a blurred background
    if new_width < output_width:
        # If the resized width is less than desired, scale up to the minimum width
        scale_factor = output_width / new_width
        large_img = resized_img.resize((int(new_width * scale_factor), int(new_height * scale_factor)), Image.LANCZOS)
        large_img = large_img.filter(ImageFilter.GaussianBlur(15))
        # Crop to target dimensions
        background = large_img.crop(((large_img.width - output_width) // 2, (large_img.height - output_height) // 2,
                                     output_width + (large_img.width - output_width) // 2, output_height + (large_img.height - output_height) // 2))
    else:
        # If the resized width is enough or more, apply blur and then crop
        blurred_img = resized_img.filter(ImageFilter.GaussianBlur(15))
        # Crop to target dimensions
        background = blurred_img.crop(((new_width - output_width) // 2, (new_height - output_height) // 2,
                                       output_width + (new_width - output_width) // 2, output_height + (new_height - output_height) // 2))

    foreground_img = resize_image(img)

    return background, foreground_img

def create_centered_blurred_background_image(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Define the target dimensions for the blurred background
    target_width, target_height = 1920, 1080
    max_foreground_height = 720

    # Resize the foreground image to have a max height of 720px while maintaining aspect ratio
    aspect_ratio = img.width / img.height
    new_foreground_width = int(max_foreground_height * aspect_ratio)
    new_foreground_height = max_foreground_height
    
    # Resize the foreground image
    foreground_img = img.resize((new_foreground_width, new_foreground_height), Image.LANCZOS)

    # Create a blurred version of the original image for the background
    blurred_background = img.resize((target_width, target_height), Image.LANCZOS).filter(ImageFilter.GaussianBlur(15))
    
    # Calculate the position to center the resized foreground image on the blurred background
    paste_x = (target_width - new_foreground_width) // 2
    paste_y = (target_height - new_foreground_height) // 2
    
    # Paste the resized foreground image on top of the blurred background
    blurred_background.paste(foreground_img, (paste_x, paste_y))

    return blurred_background

def find_and_convert_images_to_clips(directory):
    clips = []  # List to store all the ImageClips
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                full_path = os.path.join(root, file)
                try:
                    fixed_image = resize_image(Image.open(full_path))
                    clip = ImageClip(np.array(fixed_image))
                    clips.append(clip)
                except Exception as e:
                    os.remove(full_path)
    
    return clips

def convert_imageclip_to_pil(image_clip):
    # Extract a frame from the ImageClip at the first second
    frame = image_clip.get_frame(0)  # You can choose any t (time in seconds)
    
    # Convert the NumPy array (frame) to a PIL Image
    pil_image = Image.fromarray(frame)
    
    return pil_image

def remove_trailing_silence(audio_path, silence_threshold=-50.0, chunk_size=10):
    """
    Removes trailing silence from an audio file.

    Args:
    audio_path (str): Path to the audio file.
    silence_threshold (float): The lower the value, the more strictly it defines silence.
                               Measured in dBFS (decibels relative to full scale).
    chunk_size (int): Duration of a chunk in milliseconds used to analyze audio.

    Returns:
    AudioSegment: The trimmed audio segment.
    """
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Use detect_nonsilent to find non-silent chunks
    # Returns a list of [start, end] times of non-silent sections
    nonsilent_times = detect_nonsilent(audio, min_silence_len=chunk_size, silence_thresh=silence_threshold)
    
    if not nonsilent_times:
        return audio  # Return the original if no non-silent sections found

    # Extract the end time of the last non-silent chunk
    start_trim = nonsilent_times[-1][1]
    
    # Trim the audio to remove silence at the end
    trimmed_audio = audio[:start_trim]

    trimmed_audio.export(audio_path)

def wrap_text_simple(text, max_chars_per_line):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > max_chars_per_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line += " " + word if current_line else word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)

def create_video_with_audio(image_dir, audio_file_path, output_video_path):
    # List images from the directory
    images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Get the duration of the audio file
    audio_input = ffmpeg.probe(audio_file_path)
    audio_duration = float(audio_input['format']['duration'])
    
    # Calculate the total time needed for one loop of images
    image_display_duration = 8  # seconds each image is displayed
    total_image_time = len(images) * image_display_duration
    
    # Calculate how many times to repeat the image sequence
    if total_image_time == 0:
        raise ValueError("No images found in the specified directory.")
    repeat_count = int(audio_duration / total_image_time) + 1
    
    # Create a video stream by repeating images
    inputs = []
    for i in range(repeat_count):
        for image in images:
            inputs.append(
                ffmpeg.input(image, loop=1, t=image_display_duration)
            )

    # Concatenate all image inputs
    video = ffmpeg.concat(*inputs, v=1, a=0, unsafe=True)

    # Combine video with audio
    ffmpeg.output(video, ffmpeg.input(audio_file_path), output_video_path, vcodec='hevc_nvenc', acodec='aac', shortest=None).global_args('-y').run()

def new_resize_and_center_image(image_path, workplace_directory, counter):
    # Open the original image
    original = Image.open(image_path)

    # Calculate the new size preserving the aspect ratio
    aspect_ratio = original.width / original.height
    if aspect_ratio > 1.777:  # More wide than 16:9
        new_width = min(1344, original.width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(756, original.height)
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized = original.resize((new_width, new_height), Image.LANCZOS)

    # Create the background (1920x1080)
    background = original.resize((1920, 1080), Image.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(radius=15))

    background_image = os.path.join(workplace_directory, str(counter)+'background.png')
    foreground_image = os.path.join(workplace_directory, str(counter)+'foreground.png')
    background.save(background_image)

    # Calculate the position to paste the resized image on the background
    x = (1920 - new_width) // 2
    y = (1080 - new_height) // 2

    # Paste the resized image onto the background
    background.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)

    # Save the result
    background.save(image_path)

def new_process_directory_images(directory, workplace_directory):
    # Create the output directory if it doesn't exist
    
    # Walk through the directory
    counter = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}")
                new_resize_and_center_image(image_path, workplace_directory, counter)

            counter+=1

def find_largest_image(images_directory):
    max_size = 0
    largest_image_path = None
    for root, _, files in os.walk(images_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        current_size = img.width * img.height
                        if current_size > max_size:
                            max_size = current_size
                            largest_image_path = path
                except Exception:
                    os.remove(path)
    return largest_image_path

def resize_and_blur_background(image_path, output_path):
    """ Resize, crop, and blur the background image to 1920x1080 """
    (
        ffmpeg
        .input(image_path)
        .filter('scale', '1920:1080', force_original_aspect_ratio='decrease')
        .filter('pad', '1920:1080:(ow-iw)/2:(oh-ih)/2')
        .filter('gblur', sigma=30)
        .output(output_path, vframes=1)
        .run()
    )

def process_images_bg_fg(directory):
    # Paths for the new subdirectories
    foreground_path = os.path.join(directory, 'Foreground')
    background_path = os.path.join(directory, 'Background')

    # Create subdirectories if they don't exist
    os.makedirs(foreground_path, exist_ok=True)
    os.makedirs(background_path, exist_ok=True)

    # Get a list of image files in the given directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Copy files to both directories
    for file in image_files:
        shutil.copy2(os.path.join(directory, file), foreground_path)
        shutil.copy2(os.path.join(directory, file), background_path)

    # Process each file in the background directory
    for file in os.listdir(background_path):
        img_path = os.path.join(background_path, file)
        img = Image.open(img_path)
        
        # Resize with cropping or zooming to maintain aspect ratio
        img = resize_image_aspect_ratio_bg_fg(img, 1920, 1080)
        img = img.filter(ImageFilter.GaussianBlur(radius=15))  # Apply Gaussian blur

        img.save(img_path, format='PNG')

    # Process each file in the foreground directory
    for file in os.listdir(foreground_path):
        img_path = os.path.join(foreground_path, file)
        img = Image.open(img_path)
        
        # Resize to be within given dimensions, maintaining aspect ratio
        img = resize_within_bounds_bg_fg(img, 960, 1344, 540, 756)

        # Create a 1920x1080 transparent background
        background = Image.new('RGB', (1920, 1080), (144, 238, 144))

        # Calculate the position to paste the image so it is centered
        x = (background.width - img.width) // 2
        y = (background.height - img.height) // 2

        # Paste the image onto the transparent background
        background.paste(img, (x, y))

        background.save(img_path, format='JPEG')
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def resize_image_aspect_ratio_bg_fg(img, target_width, target_height):
    img_width, img_height = img.size

    # Calculate aspect ratios
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height

    if img_ratio > target_ratio:
        # Image is wider than the target ratio
        scale_factor = target_height / img_height
    else:
        # Image is taller or equal to the target ratio
        scale_factor = target_width / img_width

    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Crop the center part of the image to fit the exact target size
    x0 = (new_width - target_width) // 2
    y0 = (new_height - target_height) // 2
    img = img.crop((x0, y0, x0 + target_width, y0 + target_height))
    return img

def resize_within_bounds_bg_fg(img, min_width, max_width, min_height, max_height):
    img_width, img_height = img.size

    # Scale down to the maximum bounds
    if img_width > max_width or img_height > max_height:
        if (img_width / max_width) > (img_height / max_height):
            scale_factor = max_width / img_width
        else:
            scale_factor = max_height / img_height
    # Scale up to the minimum bounds
    elif img_width < min_width or img_height < min_height:
        if (min_width / img_width) > (min_height / img_height):
            scale_factor = min_width / img_width
        else:
            scale_factor = min_height / img_height
    else:
        return img  # No resizing needed

    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    return img

def concatenate_audio_files(audio_files, output_audio):
    """ Concatenate all audio files into one and return the path of the merged audio file. """
    inputs = [ffmpeg.input(audio_file) for audio_file in audio_files]
    joined_audio = ffmpeg.concat(*inputs, v=0, a=1)
    joined_audio.output(output_audio).global_args('-y').run()
    return output_audio

def get_audio_length(audio_file):
    """ Get the duration of the audio file. """
    return float(ffmpeg.probe(audio_file)['format']['duration'])

def generate_video_from_images(image_files, audio_file, output_video, temp_audio_duration, temp_txt_file, image_display_time=8):

    extra_time_for_last_image = temp_audio_duration % image_display_time
    number_of_images_needed = temp_audio_duration // image_display_time
    image_files_to_use = []
    image_counter = 0
    
    while(len(image_files_to_use) < number_of_images_needed):
        image_files_to_use.append(image_files[image_counter])
        image_counter+=1
        if image_counter == len(image_files):
            image_counter = 0

    # Step 1: Create a temporary file listing all images with their duration
    start_command = "ffmpeg -y"
    with open(temp_txt_file, "w") as f:
        for index, image in enumerate(image_files_to_use):
            if index == (len(image_files_to_use))-1:
                image_display_time_add = int(math.ceil(image_display_time + extra_time_for_last_image))
                start_command+=' -loop 1 -t '+str(image_display_time_add)+' -i "'+image+'"'
                break
            else:
                start_command+=' -loop 1 -t '+str(image_display_time)+' -i "'+image+'"'
    
    start_command += ' -filter_complex "concat=n='+str(number_of_images_needed)+':v=1:a=0:unsafe=1" -c:v hevc_nvenc -pix_fmt yuv420p -r 30 -movflags +faststart "'+output_video.replace('.mp4', '_temp.mp4')+'"'

    subprocess.run(start_command, check=True)

    start_command = 'ffmpeg -i "'+output_video.replace('.mp4', '_temp.mp4')+'" -i "'+audio_file+'" -c:v copy -map 0:v -map 1:a -y "'+output_video+'"'
    
    subprocess.run(start_command, check=True)

    print(f"Video created successfully: {output_video}")

def overlay_audio_on_video(video_path, audio_path, output_path):
    """
    Overlay an additional audio track onto a video using FFmpeg.
    The new audio is looped or cut to match the duration of the video.

    Args:
    video_path (str): Path to the input video file.
    audio_path (str): Path to the input audio file.
    output_path (str): Path to save the output video file.
    """
    command = [
        'ffmpeg',
        '-i', video_path,   # Input video file
        '-stream_loop', '-1',  # Loop the audio as many times as needed
        '-i', audio_path,  # Input audio file
        '-filter_complex', '[1:a]aloop=-1:0[firsta];[0:a][firsta]amix=inputs=2:duration=first[a]',  # Loop and mix audio
        '-map', '0:v',  # Map video from the first input
        '-map', '[a]',  # Map mixed audio
        '-c:v', 'copy',  # Copy the video codec
        '-c:a', 'aac',  # Use AAC for the audio codec
        '-shortest',  # Stop encoding when the shortest input stream ends
        '-y',  # Overwrite output file without asking
        output_path  # Output file path
    ]

    # Execute the FFmpeg command
    subprocess.run(command, check=True)
    os.remove(video_path)
    os.rename(output_path, video_path)
    print(f"Video created successfully: {output_path}")

def reformat_subtitles(input_file, output_file, max_line_length=120):
    # Read subtitle content from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into subtitle blocks
    blocks = content.strip().split('\n\n')

    # Concatenate lines within each subtitle block
    concatenated_blocks = []
    for block in blocks:
        lines = block.strip().split('\n')
        index = lines[0]
        timestamp = lines[1]
        
        # Concatenate lines ensuring each line doesn't exceed max_line_length
        concatenated_lines = []
        current_line = ''
        start = True
        for line in lines[2:]:
            # If adding the current line would exceed max_line_length, start a new line
            if len(current_line) + len(line) > max_line_length:
                concatenated_lines.append(current_line)
                current_line = line
            else:
                if start == True:
                    current_line = line.strip()
                    start = False
                else:
                    current_line += ' ' + line.strip()
        # Append the last line to the concatenated lines
        concatenated_lines.append(current_line)
        
        # Construct concatenated block
        concatenated_block = index + '\n'+timestamp+'\n'+('\n'.join(concatenated_lines))+"\n"
        concatenated_blocks.append(concatenated_block)

    # Write concatenated subtitle content to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(concatenated_blocks))

def split_sentences_to_max_char(sentences, max_chars=50):
    segments = []

    for sentence in sentences:
        words = sentence.split()
        current_segment = []

        current_length = 0
        for word in words:
            # Include a space in the length if the current segment is not empty
            word_length = len(word) + (1 if current_segment else 0)

            # If adding this word would exceed the max length, append the current segment
            # to segments and start a new one
            if current_length + word_length > max_chars:
                segments.append(' '.join(current_segment))
                current_segment = [word]
                current_length = len(word)
            else:
                # Otherwise, add the word to the current segment
                if current_segment:
                    current_segment.append(word)
                else:
                    current_segment = [word]
                current_length += word_length

        # Add the last segment if any words remain
        if current_segment:
            segments.append(' '.join(current_segment))

    return segments

def create_video_with_text_and_audio(image_path, audio_path, output_path, text, duration, font_path, font_size=100):
    """
    Create a video from an image with each line of text independently centered using FFmpeg.
    This function ensures that the fontfile path is handled correctly and includes center alignment for the text.
    """

    # Properly escape text to prevent issues in filter_complex
    safe_text = text.replace("'", "\\'").replace(":", "\\:")

    # Escape file paths for use in FFmpeg filters (Windows specific issue)
    font_path_escaped = font_path.replace("\\", "\\\\\\\\").replace("C:", "C\\:")

    # Split the text into lines and prepare drawtext filters
    lines = safe_text.split('\n')
    num_lines = len(lines)
    line_height = font_size + 10  # Adjust spacing based on font size
    start_y = (1080 - line_height * num_lines) // 2  # Center the block vertically on a 1080p frame

    filter_complex = []
    for i, line in enumerate(lines):
        y_position = start_y + i * line_height
        filter_complex.append("drawtext=text='"+line.replace("'", '')+"':fontfile='"+font_path_escaped+"':fontsize="+str(font_size)+":fontcolor=white:borderw=6:bordercolor=black:x=((w-text_w)/2):y="+str(y_position)) #has trouble with multi-line titles

    filter_complex_string = ', '.join(filter_complex)# + ":format=yuv420p[v]"  # Combine all filters and add video format
    print(filter_complex_string)
    # exit()
    # Full command setup for subprocess
    command = [
        'ffmpeg',
        '-loop', '1',
        '-i', image_path,
        '-i', audio_path,
        '-filter_complex', filter_complex_string,
        '-map', '[v]',
        '-map', '1:a',
        '-c:v', 'hevc_nvenc',  # using hevc_nvenc for video encoding
        '-c:a', 'aac',
        '-pix_fmt', 'yuv420p',
        '-t', str(duration),
        '-shortest',
        '-loglevel', 'debug',
        '-y',  # Ensure this is uncommented to overwrite output files without asking
        output_path
    ]

    # Execute the FFmpeg command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Video created successfully: {output_path}")
    except Exception:
        command = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-filter_complex', filter_complex_string,
            # '-map', '[v]',
            '-map', '1:a',
            '-c:v', 'hevc_nvenc',  # using hevc_nvenc for video encoding
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-t', str(duration),
            '-shortest',
            '-loglevel', 'debug',
            '-y',  # Ensure this is uncommented to overwrite output files without asking
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Video created successfully: {output_path}")

def hardcode_subtitles(video_path, subtitle_path, output_path, font_path, font_size=16, subtitle_margin_from_bottom=1000):
    """
    Hardcode subtitles into a video file with customized font settings using FFmpeg.
    This version places subtitles at the bottom middle of the screen, 60 pixels up,
    with a white font and black outline, and ensures multi-line text is centered.

    Args:
    video_path (str): Path to the input video file.
    subtitle_path (str): Path to the subtitle file (.srt, .ass, etc.).
    output_path (str): Path to save the output video file.
    font_path (str): Path to the .ttf font file.
    font_size (int): Font size of the subtitles.
    subtitle_margin_from_bottom (int): Margin from the bottom in pixels.
    """

    def escape_ffmpeg_path(path):
        return path.replace('\\', '\\\\').replace(':', '\\:')

    subtitle_path = escape_ffmpeg_path(subtitle_path)
    font_path = escape_ffmpeg_path(font_path)

    # Subtitle filter with custom styling
    subtitles_filter = (
        f"subtitles='{subtitle_path}':"
        f"force_style='FontName={font_path},"
        f"FontSize={font_size},"
        f"Alignment=2,"
        f"MarginV={10},"
        f"PrimaryColour=&H00FFFFFF,"  # White, BGR order
        f"OutlineColour=&H00000000,"  # Black, BGR order
        f"BorderStyle=1,"  # 1 for outline (vs box)
        f"Outline=2,"  # Outline thickness
        f"Shadow=0'"  # No shadow
    )
    
    # FFmpeg command construction
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', subtitles_filter,
        '-c:v', 'hevc_nvenc',
        '-crf', '23',
        '-preset', 'fast',
        '-c:a', 'copy',
        '-loglevel', 'debug',
        output_path
    ]

    # Execute the FFmpeg command

    subprocess.run(command, check=True)
    print(f"Video with customized subtitles created successfully: {output_path}")

def create_animation_ffmpeg(output_video, video_length=10, fps=30):
    """
    Creates a video of a square bouncing around a 1920x1080 frame.
    
    Args:
    - output_video (str): The path to save the output video.
    - video_length (int): The length of the video in seconds.
    - fps (int): The frames per second of the video.
    """
    # Calculate the total number of frames
    total_frames = video_length * fps
    
    # FFmpeg command to create the bouncing square animation
    command = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f"color=c=black:s=1920x1080:r={fps}",
        '-vf', f"drawbox=y='if(gte(t,1), sin(t*4)*450+540, 0)':x='if(gte(t,1), cos(t*3)*900+960, 0)':w=100:h=100:c=red@0.5:t=fill; format=yuv420p",
        '-t', str(video_length),
        '-r', str(fps),
        '-c:v', 'hevc_nvenc',  # Using NVIDIA hardware encoder for HEVC (H.265)
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-y', output_video
    ]
    
    # Execute the FFmpeg command
    subprocess.run(command, check=True)
    print(f"Video created successfully: {output_video}")

def add_dummy_audio(video_input):
    """ Add a silent audio track to a video. """
    temp_file = os.path.join(os.path.dirname(video_input), 'temp.mp4')
    command = [
        'ffmpeg',
        '-f', 'lavfi',               # Use the libavfilter input virtual device
        '-i', 'anullsrc',            # Generate silent audio source
        '-i', video_input,           # Input video
        '-c:v', 'copy',              # Copy video codec settings
        '-c:a', 'aac',               # Set audio codec to AAC
        '-shortest',                 # Stop encoding when the shortest stream ends
        '-y', temp_file           # Output video path, overwrite if necessary
    ]
    subprocess.run(command, check=True)
    os.remove(video_input)
    os.rename(temp_file, video_input)
    print(f"Dummy audio added successfully: {video_input }")

def overlay_videos(background_video, overlay_video, output_video):
    """
    Overlay one video onto another while making a specific color (light green) transparent.
    The output video is the length of the background video.
    
    Args:
    background_video (str): Path to the background video file.
    overlay_video (str): Path to the overlay video file with the light green background.
    output_video (str): Path where the output video will be saved.
    """
    command = [
        'ffmpeg',
        '-i', f'"{background_video}"',  # Input background video
        '-i', f'"{overlay_video}"',     # Input overlay video with light green background #90EE90
        '-filter_complex', '[1:v]colorkey=0x90EE90:0.1:0.1[ckout];[0:v][ckout]overlay=shortest=1[out]',
        '-map', '[out]',        # Map the output of the overlay filter to the output file
        '-map', '0:a',        
        '-c:v', 'hevc_nvenc',   # Video codec for output
        '-c:a', 'aac',     
        '-crf', '23',          # Quality of the output video
        '-preset', 'fast',     # Encoding speed/quality trade-off
        '-f', 'mp4',           # Output format 'mp4
        '-y',                  # Overwrite output file without asking
        f'"{output_video}"'           # Output file path
    ]

    # Execute the FFmpeg command
    subprocess.run(' '.join(command), shell=True, check=True)
    print(f"Video created successfully: {output_video}")

def create_animation_video(output_path, duration, width=1920, height=1080):
    choice = randint(0, 2)
    choice = 2

    bar_width = random.choice([50, 100, 150, 200])  # Width of the vertical bar
    bar_height = random.choice([50, 100, 150, 200])  # Height of the vertical bar
    initial_velocity = random.choice([50, 60])  # Reduced initial velocity for slower movement
    diagonal_velocity = (initial_velocity / np.sqrt(2), initial_velocity / np.sqrt(2))  # Initial velocity for diagonal movement
    position = [width // 2, height // 2]  # Initial position for the bar [x, y]

    light_green = (144, 238, 144)  # Light green background color
    grey = (0, 0, 0)  # Grey color for the bar with 50% opacity

    def make_frame_diagonal_bouncing_bar(t):

        nonlocal diagonal_velocity

        frame_time = 1 / 30  # Assuming 30 fps

        # Update position based on velocity
        position[0] += diagonal_velocity[0] * frame_time
        position[1] += diagonal_velocity[1] * frame_time

        # Check for collisions and reverse direction if necessary
        if position[0] <= 0 or position[0] + bar_width >= width:
            diagonal_velocity = (-diagonal_velocity[0], diagonal_velocity[1])
            position[0] = max(0, min(width - bar_width, position[0]))
        if position[1] <= 0 or position[1] + bar_height >= height:
            diagonal_velocity = (diagonal_velocity[0], -diagonal_velocity[1])
            position[1] = max(0, min(height - bar_height, position[1]))

        # Create the background frame
        frame = np.full((height, width, 3), light_green, dtype='uint8')  # RGB

        # Draw the semi-transparent grey bar
        frame[int(position[1]):int(position[1] + bar_height), int(position[0]):int(position[0] + bar_width)] = grey

        return frame

    # Create the video clip object
    animation_clip = VideoClip(make_frame_diagonal_bouncing_bar, duration=duration)

    # Writing the video file using the HEVC codec
    animation_clip.write_videofile(output_path, fps=30, codec='hevc_nvenc', preset='fast')

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Function to compute perceptual hash for images
def image_phash(filepath):
    try:
        with Image.open(filepath) as img:
            return imagehash.phash(img)
    except Exception as e:
        os.remove(filepath)
        return None

# Function to compare similarity of perceptual hashes
def phash_similarity(hash1, hash2):
    return (hash1 - hash2) <= 5

# Function to find duplicates based on perceptual hashing
def find_similar_images(folder, exceptions):
    phash_map = {}
    duplicates = defaultdict(list)
    
    for dirpath, dirnames, filenames in os.walk(folder):
        # Skip any exception subfolders
        if any(x in dirpath for x in exceptions):
            continue
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                filepath = os.path.join(dirpath, filename)
                img_phash = image_phash(filepath)
                if img_phash is not None:
                    found_duplicate = False
                    for stored_phash in list(phash_map):
                        if phash_similarity(img_phash, stored_phash):
                            duplicates[phash_map[stored_phash]].append(filepath)
                            found_duplicate = True
                            break
                    if not found_duplicate:
                        phash_map[img_phash] = filepath
    return duplicates

# Function to delete files in a list, keeping the first (original)
def delete_duplicates(duplicates):
    for original, dup_list in duplicates.items():
        for dup in dup_list:
            os.remove(dup)
            print(f"Deleted duplicate: {dup}")

# Function to check subfolders and return those with less than min_images images
def populate_subfolders_with_images(parent_folder, source_folder, exceptions=None, min_files=3):
    """
    Populates subfolders within the parent_folder with images from source_folder
    until each subfolder has at least min_files files, excluding specified subfolders.
    
    :param parent_folder: Path to the directory to check and populate.
    :param source_folder: Directory from which to copy images.
    :param exceptions: List of directory names to exclude from checking and populating.
    :param min_files: Minimum number of files each subdirectory should contain.
    """
    if exceptions is None:
        exceptions = []

    # Normalize the parent folder path to avoid mismatches due to trailing slashes
    parent_folder = os.path.normpath(parent_folder)

    # Prepare to ignore paths that contain any of the exceptions
    for dirpath, dirnames, filenames in os.walk(parent_folder):
        # Normalize the directory path
        dirpath = os.path.normpath(dirpath)

        # Skip the parent directory to avoid copying files into it
        if dirpath == parent_folder:
            continue

        # Check if current path should be excluded based on any part of its sub-path
        if any(exc in dirpath for exc in exceptions):
            print(f"Skipping excluded directory: {dirpath}")
            continue
        
        # Calculate current file count in the subdirectory
        file_count = len(filenames)
        
        # If the file count is less than the minimum required, calculate the deficiency
        if file_count < min_files:
            needed_files = min_files - file_count
            added_count = 0
            
            # Iterate over the files in the source directory
            for src_filename in os.listdir(source_folder):
                if src_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    src_filepath = os.path.join(source_folder, src_filename)
                    target_filepath = os.path.join(dirpath, src_filename)
                    
                    # Only copy if the file doesn't already exist in the target directory
                    if not os.path.exists(target_filepath):
                        shutil.copy(src_filepath, target_filepath)
                        added_count += 1
                        print(f"Copied {src_filepath} to {target_filepath}")
                        
                        # Stop once we have added enough files
                        if added_count >= needed_files:
                            break

            # Log the status after attempting to add files
            if added_count > 0:
                print(f"Updated {dirpath} with {added_count} files.")
        else:
            print(f"No need to update {dirpath}, it has sufficient files.")

# Function to copy images from source_folder to target_folder until num_required images are copied
def copy_images(source_folder, target_folder, num_required):
    images_copied = 0
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            src_file = os.path.join(source_folder, filename)
            dest_file = os.path.join(target_folder, filename)
            if not os.path.exists(dest_file):
                shutil.copy(src_file, dest_file)
                images_copied += 1
            if images_copied >= num_required:
                break

def overlay_videos_moviepy(background_video_path, overlay_video_path, output_video_path, pos=("center", "center")):
    """
    Overlay one video over another video at a specified position.

    Args:
    background_video_path (str): Path to the background video file.
    overlay_video_path (str): Path to the overlay video file.
    output_video_path (str): Path to save the output video.
    pos (tuple): A tuple (x, y) representing the position where the overlay video will be placed on the background.
    """
    # Load the background video
    background_clip = VideoFileClip(background_video_path, fps=30)
    
    # Load the overlay video
    overlay_clip = VideoFileClip(overlay_video_path, fps=30, has_mask=True).set_position(pos)
    
    # Create a composite video clip
    final_clip = CompositeVideoClip([background_clip, overlay_clip])

    # Set the duration of the final clip to the duration of the longest clip
    final_duration = max(background_clip.duration, overlay_clip.duration)
    final_clip = final_clip.set_duration(final_duration)

    # Write the result to a file
    final_clip.write_videofile(output_video_path, codec="hevc_nvenc", fps=30)

def create_blurred_background_image(image_path, output_path, audio_duration):
    (
        ffmpeg
        .input(image_path)
        .filter('gblur', sigma=15)
        .filter('scale', 1920, 1080)
        .output(output_path, vframes=1, format='image2', vcodec='png')
        .run()
    )
    # Create a video from the blurred image
    (
        ffmpeg
        .input(output_path, loop=1, framerate=25)
        .output('blurred_background.mp4', t=audio_duration, pix_fmt='yuv420p', vcodec='hevc_nvenc')
        .run()
    )

def create_blurred_background_video(image_path, output_path, duration):
    # Step 1: Apply Gaussian Blur and Resize the image
    intermediate_image = 'temp_blurred.png'
    (
        ffmpeg
        .input(image_path)
        .filter('gblur', sigma=15)
        .filter('scale', 1920, 1080)
        .output(intermediate_image, vframes=1)
        .run(overwrite_output=True)
    )

    # Step 2: Create a video from the blurred image
    (
        ffmpeg
        .input(intermediate_image, loop=1, framerate=25)
        .output(output_path, t=duration, pix_fmt='yuv420p', vcodec='hevc_nvenc')
        .run(overwrite_output=True)
    )

    # Optionally remove the intermediate image if you don't need it after creating the video
    os.remove(intermediate_image)

def create_video_from_image(image_path, output_video_path, duration):
    # Loop an image to create a video for the specified duration
    (
        ffmpeg
        .input(image_path, loop=1, framerate=25)  # Loop the image
        .output(output_video_path, t=duration, pix_fmt='yuv420p', vcodec='hevc_nvenc')  # Set duration and video codec
        .run(overwrite_output=True)
    )

def composite_clips(background_path, text_path, output_path):
    ffmpeg.input(background_path).overlay(
        ffmpeg.input(text_path),
        x='(main_w-overlay_w)/2',
        y='(main_h-overlay_h)/2',
        shortest=True,
    ).output(output_path, vcodec='hevc_nvenc', pix_fmt='yuv420p', crf=23, preset='fast').run()

def concatenate_audio(audio_files, output_file):
    inputs = [ffmpeg.input(audio_file) for audio_file in audio_files]
    ffmpeg.concat(*inputs, v=0, a=1).output(output_file).run()

def create_color_video_with_audio(duration, color, audio_file, output_video):
    (
        ffmpeg
        .input('color=c={}:s=1920x1080:r=30:d={}'.format(color, duration), f='lavfi')
        .input(audio_file)
        .output(output_video, vcodec='hevc_nvenc', pix_fmt='yuv420p', acodec='aac', shortest=None)
        .run()
    )

def concatenate_videos(video_paths, output_path):
    """
    Concatenates multiple videos into a single video using ffmpeg-python.
    
    Args:
    video_files (list of str): A list of video file paths to concatenate.
    output_file (str): The path where the concatenated video will be saved.
    """
    # Create an input stream for each video file
    inputs = [ffmpeg.input(file) for file in video_paths]
    
    # Concatenate video files using the concat filter
    joined = ffmpeg.concat(*inputs, v=1, a=1).node
    
    # Take the output streams from the concat filter node
    v = joined[0]
    a = joined[1]
    
    # Create the output file stream, specifying video and audio codecs
    output = ffmpeg.output(v, a, output_path, vcodec='hevc_nvenc', acodec='aac', strict='experimental')
    
    # Run the ffmpeg process
    ffmpeg.run(output, overwrite_output=True)

def create_video_from_images(image_paths, durations, output_path):
    inputs = []
    for img_path, duration in zip(image_paths, durations):
        inputs.append(ffmpeg.input(img_path, t=duration, f='image2pipe', r=1))

    ffmpeg.concat(*inputs, v=1, a=0).output(output_path, vcodec='hevc_nvenc', pix_fmt='yuv420p').run()

def overlay_audio(script_audio_path, music_directory):
    music_file_path = os.path.join(music_directory, random.choice(os.listdir(music_directory)))
    output_audio_path = 'final_mixed_audio.mp3'
    (
        ffmpeg
        .input(script_audio_path)
        .input(music_file_path)
        .filter_('amix', inputs=2, duration='first', dropout_transition=3)
        .output(output_audio_path)
        .run()
    )
    return output_audio_path

def create_silent_audio(duration, output_path):
    (
        ffmpeg
        .input('anullsrc', format='lavfi', channel_layout='stereo', sample_rate=44100)
        .output(output_path, t=duration, acodec='pcm_s16le')
        .run(overwrite_output=True)
    )

def convert_rgba_to_rgb(directory):
    # Walk through all files and subdirectories
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                # Attempt to open an image file
                with Image.open(filepath) as img:
                    if img.mode == 'RGBA':
                        print(f"Converting {filepath} from RGBA to RGB.")
                        # Convert the image from RGBA to RGB
                        rgb_img = img.convert('RGB')
                        # Save the converted image
                        rgb_img.save(filepath)
                        print(f"Saved converted image to {filepath}")
                    else:
                        print(f"Skipping non-RGBA image: {filepath}")
            except IOError:
                # If opening the file as an image fails, skip it
                print(f"Skipped non-image file: {filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

def convert_rgb_to_rgba(directory):
    # Walk through all files and subdirectories
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                # Attempt to open an image file
                with Image.open(filepath) as img:
                    if img.mode == 'RGB':
                        print(f"Converting {filepath} from RGB to RGBA.")
                        # Convert the image from RGB to RGBA by adding an alpha channel
                        rgba_img = img.convert('RGBA')
                        # Save the converted image
                        rgba_img.save(filepath)
                    else:
                        print(f"Skipping non-RGB image: {filepath}")
            except IOError:
                # If opening the file as an image fails, skip it
                print(f"Skipped non-image file: {filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

def gpu_accelerated_write(clip, output_path):

    ffmpeg_writer.ffmpeg_write_video(
        clip=clip,             # Your VideoClip instance
        filename=output_path, # Desired output file name
        fps=30,                      # Frames per second
        codec="h264_nvenc",          # Using NVIDIA GPU acceleration
        preset="fast",
        logger='bar',
        write_logfile=True,
        pix_fmt='rgb24'                   # Number of threads to use                                     # Progress logger type
    )

    return output_path

def find_and_convert_images_to_clips_ffmpeg(directory):
    clips = []  # List to store all file paths
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                full_path = os.path.join(root, file)
                clips.append(full_path)
    return clips

def create_exponential_growth_video(clips, audio, total_duration, channel_video, start_duration=1.8462):

    # Generate durations until the sum reaches or exceeds the total_duration
    canvas = Image.new('RGB', (1920, 1080), (0, 0, 0))
    final_video = []
    duration_acc = 0
    min_duration = (start_duration/8)
    amount_of_clips = len(clips)-1
    counter = 0
    current_step = 0
    interval = 8

    channel_picture_length = (32.3085-29.5392)

    while duration_acc < (total_duration-channel_picture_length):

        frame = clips[counter].get_frame(0)  # You can choose any t (time in seconds)
        img = Image.fromarray(frame)

        # Calculate the new width to maintain aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(512 * aspect_ratio)

        # Resize the image to a maximum height of 256 pixels
        resized_img = img.resize((new_width, 512), Image.LANCZOS)

        # Generate random coordinates for pasting
        max_x = canvas.width - resized_img.width
        max_y = canvas.height - resized_img.height
        random_x = random.randint(0, max_x)
        random_y = random.randint(0, max_y)

        # Paste the resized image onto the canvas at a random location
        canvas.paste(resized_img, (random_x, random_y))

        final_video.append(ImageClip(np.array(canvas)).set_duration(start_duration))

        current_step += 1
        if current_step == interval:
            interval*=2
            start_duration/=2
        if start_duration < min_duration:
            start_duration = min_duration
        if counter == amount_of_clips:
            counter = 0

        duration_acc+=start_duration
        counter+=1

    final_clip = concatenate_videoclips(final_video, method="compose")
    final_clip = final_clip.set_audio(audio)
    final_clip = final_clip.subclip(0, (total_duration-channel_picture_length))

    channel_clip = VideoFileClip(channel_video)
    channel_clip = channel_clip.without_audio()

    if channel_clip.duration < channel_picture_length:
        channel_clip = concatenate_videoclips([channel_clip, channel_clip, channel_clip])
        channel_clip = channel_clip.set_duration(channel_picture_length)
    else:    
        channel_clip = channel_clip.subclip(0, channel_picture_length)

    # Concatenate all the clips with adjusted durations
    final_clip = concatenate_videoclips([final_clip, channel_clip], method="compose")
    final_clip = final_clip.set_audio(audio)
    final_clip = final_clip.subclip(0, total_duration)

    return final_clip

def create_animation(duration):
    width, height = 1920, 1080  # Resolution of the video

    def make_frame(t):
        # Create a blank image with transparency
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image, mode='RGB')

        # Calculate the position of the circle
        circle_radius = 50
        x = int((width + 2 * circle_radius) * t / duration) - circle_radius
        y = height // 2

        # Draw a circle that moves horizontally from left to right
        draw.ellipse((x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius), fill=(255, 0, 0))

        arr = np.array(image)

        # Convert the PIL image to an array
        return arr

    # Create a video clip from the frames
    animation_clip = VideoClip(make_frame, duration=duration)
    animation_clip = animation_clip.set_duration(duration)
    return animation_clip

def generator_with_outline(text):

    random_font = random.choice(['Georgia-Regular', 'Times-New-Roman', 'Rockwell', 'Sylfaen', 'Arial'])
    # Main text clip with outline
    txt_clip = TextClip(wrap_text_simple(text, 120), font=random_font, fontsize=40, color='white', stroke_color='black', stroke_width=2, bg_color='None')

    return txt_clip.set_duration(txt_clip.duration)

def text_to_audio_file(text, audio_filename, client, voice_model):

    # speed = 0.9
    # device = 'auto'

    # model = TTS(language='EN_V2', device=device)
    # speaker_ids = model.hps.data.spk2id

    # model.tts_to_file(text, speaker_ids['EN-BR'], audio_filename, speed=speed)

    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice_model,
        input=text
    )

    try:
        response.with_streaming_response.method(audio_filename)
    except Exception:
        print("with_streaming_response.method() failed")
        response.stream_to_file(audio_filename)

    return audio_filename

def find_images(directory):
    """Recursively find all images in the directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                yield os.path.join(root, file)

def copy_random_images(source_directory, target_directory, num_images=5):
    """Copy a random set of images from source to target directory."""
    # Find all images in the source directory
    images = list(find_images(source_directory))
    
    # Select a random subset of images
    if len(images) < num_images:
        print(f"Not enough images found. Only found {len(images)} images.")
        chosen_images = images
    else:
        chosen_images = random.sample(images, num_images)
    
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)
    
    # Copy the selected images to the target directory
    for image in chosen_images:
        # Define the target path
        target_path = os.path.join(target_directory, os.path.basename(image))
        # Copy the image
        shutil.copy2(image, target_path)
        print(f"Copied {image} to {target_path}")

def hashes_are_similar(hash1, hash2, tolerance=5):
    # Calculate the number of differing bits between hashes directly
    if (hash1 - hash2) <= tolerance:
        return True
    else:
        return False

def script_to_subtitles(script_audio, output_filename='script.srt'):
    # model_size = "large-v3"

    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # segments, info = model.transcribe(script_audio, beam_size=5, language="en", condition_on_previous_text=False)

    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type='int8_float16')
    
    # Set smaller chunk lengths and max new tokens to control subtitle line length
    # chunk_length = 5  # in seconds, smaller chunks
    # max_new_tokens = 30  # limits the amount of text generated per chunk
    
    segments, info = model.transcribe(script_audio, beam_size=5, language="en",
                                    #   chunk_length=chunk_length,
                                      #max_new_tokens=max_new_tokens,
                                      condition_on_previous_text=False)
    
    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    with open(output_filename, 'w', encoding='utf-8') as file:
        for i, segment in enumerate(segments, start=1):
            start_time = int(segment.start * 1000)  # Convert seconds to milliseconds
            end_time = int(segment.end * 1000)      # Convert seconds to milliseconds
            start_timestamp = f'{start_time // 3600000:02}:{(start_time % 3600000) // 60000:02}:{(start_time % 60000) // 1000:02},{start_time % 1000:03}'
            end_timestamp = f'{end_time // 3600000:02}:{(end_time % 3600000) // 60000:02}:{(end_time % 60000) // 1000:02},{end_time % 1000:03}'

            # Write the subtitle number
            file.write(f'{i}\n')
            # Write the timestamp
            file.write(f'{start_timestamp} --> {end_timestamp}\n')
            # Write the text
            file.write(f'{(wrap_text_simple(segment.text.strip(), 35).strip())}\n')
            # Write a blank line to separate entries
            file.write('\n')
    
    return output_filename

def remove_unwanted_files(directory, allowed_extensions):

    # Walk through all subdirectories and the main directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the extension of the current file
            ext = os.path.splitext(file)[1]
            # Check if the file extension is not in the allowed list
            if ext not in allowed_extensions:
                # Construct full path to the file
                file_path = os.path.join(root, file)
                # Remove the file
                os.remove(file_path)
                print(f"Removed: {file_path}")

def create_subdirectories(base_dir, my_list):
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory: {base_dir}")
    
    # Create subdirectories within the base directory
    for i in range(len(my_list)):
        subdir_path = os.path.join(base_dir, str(i))
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            print(f"Created subdirectory: {subdir_path}")

def delete_everything_in_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    # Remove all files and subdirectories
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Remove files and links
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directories recursively

    print("All contents of the directory have been deleted.")

def send_notification(notification, title, message):
    msg = notification.msg(message)
    msg.set("title", title)
    try:
        notification.send(msg)
    except Exception as e:
        print(str(e))
        pass

def process_text(text):
    # Normalize newlines to ensure there's only one newline where there are multiples
    text = re.sub(r'\n+', '\n', text)
    # Split text into lines
    split_text = text.split('\n')
    # Filter out any empty strings and strings with less than 6 characters
    filtered_text = [line for line in split_text if line and len(line) > 5]
    return filtered_text

def authenticate_channel(client_secrets_file, scopes, channel_name, full_path):
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_local_server(port=0)
    save_credentials(channel_name, credentials, full_path)
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)

def load_or_authenticate_channel(channel_name, client_secrets_file, full_path):
    credentials = load_credentials(channel_name, full_path)

    if credentials and credentials.valid:
        return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
    else:
        return authenticate_channel(client_secrets_file, ["https://www.googleapis.com/auth/youtube", 'https://www.googleapis.com/auth/youtube.force-ssl'], channel_name, full_path)

def save_credentials(channel_name, credentials, full_path):
    credentials_path = full_path+'\\credentials'+channel_name+'.pkl'
    with open(credentials_path, 'wb') as credentials_file:
        pickle.dump(credentials, credentials_file)

def load_credentials(channel_name, full_path):
    credentials_path = os.path.join(full_path, 'credentials' + channel_name + '.pkl')
    if os.path.exists(credentials_path):
        with open(credentials_path, 'rb') as credentials_file:
            credentials = pickle.load(credentials_file)
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                save_credentials(channel_name, credentials, full_path)  # Save the refreshed credentials
                return credentials
            elif credentials and credentials.valid:
                return credentials
    return None

def run_script(directory):
    """Run the AutomatedYoutube.py script with the given directory."""
    print(f"Running script for {directory}\n")
    subprocess.run(["python", "AutomatedYoutube.py", directory, "-l"], text=True)

def send_notification(notification, title, message):
    msg = notification.msg(message)
    msg.set("title", title)
    try:
        notification.send(msg)
    except Exception:
        pass

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def sort_images(image_paths):
    landscape_images = []
    portrait_images = []

    for image_path in image_paths:
        if os.path.exists(image_path):
            width, height = get_image_size(image_path)
            if width > height:
                landscape_images.append((image_path, width * height))
            else:
                portrait_images.append((image_path, width * height))

    landscape_images.sort(key=lambda x: x[1], reverse=True)
    portrait_images.sort(key=lambda x: x[1], reverse=True)

    return [image[0] for image in landscape_images] + [image[0] for image in portrait_images]

def zoom_and_crop_to_aspect_ratio(filepath, target_aspect_ratio=1.7778):
    # Open the image
    img = Image.open(filepath)
    
    # Calculate the current aspect ratio
    width, height = img.size
    current_aspect_ratio = width / height
    
    # Determine if we need to crop the width or the height to achieve the target aspect ratio
    if current_aspect_ratio > target_aspect_ratio:
        # Current image is too wide, need to crop the width
        new_width = int(height * target_aspect_ratio)
        left = (width - new_width) / 2
        top = 0
        right = (width + new_width) / 2
        bottom = height
    else:
        # Current image is too tall, need to crop the height
        new_height = int(width / target_aspect_ratio)
        top = (height - new_height) / 2
        left = 0
        bottom = (height + new_height) / 2
        right = width
    
    # Crop the image to the target aspect ratio
    img = img.crop((left, top, right, bottom))
    
    # Optionally, resize the cropped image back to the original dimensions or any specific dimensions
    # This step might not be necessary if maintaining the original image size isn't required
    # img = img.resize((original_width, original_height), Image.ANTIALIAS)

    # Convert the image to RGB mode if not already to avoid saving issues with JPEG
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save the edited image back to the original filepath
    img.save(filepath, 'JPEG')

    return filepath

def enhance_image_with_vignette(filepath, vignette_strength=3.0, sharpness_factor=2.0, contrast_factor=1.5):
    # Open the original image
    img = Image.open(filepath)
    width, height = img.size

    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Correctly handle file format based on the file extension
    file_extension = filepath.split('.')[-1].upper()  # Extract file extension and convert to uppercase
    format = 'JPEG' if file_extension in ['JPG', 'JPEG'] else file_extension
    if format not in ['JPEG', 'PNG', 'TIFF', 'BMP', 'GIF']:
        format = 'PNG'  # Default to PNG if unknown or unsupported extension
    img.save(filepath, format=format)

    return filepath

def filter_dictionary(orig_dict):
    # This will create a new dictionary only including key-value pairs
    # where the key is not an empty string and the value is None
    return {k: v for k, v in orig_dict.items() if k and v is None}

def remove_similar_images_in_directory(specific_directory_path, tolerance=5):
    image_hashes = {}
    files_to_remove = []

    for filename in os.listdir(specific_directory_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(specific_directory_path, filename)
            try:
                with Image.open(filepath) as img:
                    img_hash = imagehash.dhash(img)
                    # Check against existing hashes for similarity
                    similar_found = False
                    for existing_hash in list(image_hashes.keys()):
                        if hashes_are_similar(img_hash, existing_hash, tolerance):
                            files_to_remove.append(filepath)
                            print(f"Similar image found: {filename} is similar to {image_hashes[existing_hash]}")
                            similar_found = True
                            break
                    if not similar_found:
                        image_hashes[img_hash] = filename
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Remove similar files
    for filepath in files_to_remove:
        try:
            os.remove(filepath)
            print(f"Removed {filepath}")
        except Exception as e:
            print(f"Failed to remove {filepath}: {e}")

def randomSleep():
    sleep(randint(2,4)*(randint(500,1000)/1000))

def dhash(image, hash_size=6):  # Reduced hash_size
    # Resize and grayscale the image
    resized_image = image.convert('L').resize((hash_size + 1, hash_size))
    
    # Calculate the difference hash
    pixels = list(resized_image.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            diff.append(pixels[col + row * (hash_size + 1)] > pixels[col + row * (hash_size + 1) + 1])
    
    # Build the hash
    return ''.join(str(int(b)) for b in diff)

def process_images_in_directory(directory, image_with_their_search_terms):
    for filename in os.listdir(directory):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        filepath = os.path.join(directory, filename)
        img = Image.open(filepath)

        # Resize image to have a height of 1000 if not already
        if img.height != 1000:
            scale_factor = 1000 / img.height
            img = img.resize((int(img.width * scale_factor), 1000), Image.LANCZOS)

        # If image is wider than 1920px, crop it
        if img.width > 1920:
            left = (img.width - 1920) / 2
            img = img.crop((left, 0, left + 1920, img.height))
        elif img.width < 1920:
            # Create a blurred background
            background = img.copy().resize((1920, 1000)).filter(ImageFilter.GaussianBlur(15))
            enhancer = ImageEnhance.Brightness(background)
            background = enhancer.enhance(0.7)

            # Calculate position to paste the resized image onto the background
            x_offset = (1920 - img.width) // 2
            background.paste(img, (x_offset, 0))
            img = background

        # Add black bar for subtitles
        black_bar = Image.new('RGB', (1920, 1080), (0, 0, 0))
        black_bar.paste(img, (0, 0))
        img = black_bar
        if '(' in filename:
            char1 = '('
        else:
            char1 = '.'

        text = image_with_their_search_terms[filename[:filename.find(char1)]]
        cleaned_text = re.sub(r'\s{2,}', ' ', text)

        img_format = 'PNG' if filename.lower().endswith('.png') else 'JPEG'
        img.save(filepath, format=img_format)  # Specify the format explicitly based on the file e
        paste_text_with_background(filepath, cleaned_text)

def clean_directory(source_directory, target_directory, notification):
    for filename in os.listdir(source_directory):
        filepath = os.path.join(source_directory, filename)
        
        while(1):
            try:
                # Delete files that are not JPG, JPEG, or PNG
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    os.remove(filepath)
                    continue
                
                # Copy certain images to another directory
                img = Image.open(filepath)
                aspect_ratio = img.width / img.height
                if img.width > 800 and img.height > 400 and aspect_ratio > 1.5:
                    shutil.copy(filepath, os.path.join(target_directory, filename))
                break
            except Exception as e:
                send_notification(notification, "Error", str(e))
            
def process_topics_and_paragraphs(list_of_things, max_length_for_topic):
    new_final_script = []
    current_paragraph = ""

    for index, item in enumerate(list_of_things):
        if len(item) < max_length_for_topic:
            # If there's a paragraph being built, append it before adding a new topic
            if current_paragraph:
                new_final_script.append(current_paragraph)
                current_paragraph = ""  # Reset current paragraph
            new_final_script.append(item)  # Append topic directly
        else:
            # For paragraphs, merge with the current_paragraph string
            if current_paragraph:
                current_paragraph += " " + item  # Merge with existing paragraph
            else:
                current_paragraph = item  # Start a new paragraph
    if current_paragraph:
        new_final_script.append(current_paragraph)

    return new_final_script

def extract_initial_number(s):
    match = re.match(r'\d+', s)
    if match:
        return int(match.group())
    else:
        return float('inf')

def reduce_image_size(thumbnail_file):
    if os.path.getsize(thumbnail_file) > 2000000:
        print("Reducing thumbnail size...")
        max_size_bytes = 2 * 1024 * 1024
        img = Image.open(thumbnail_file)
        quality = 85

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        original_extension = os.path.splitext(thumbnail_file)[1]
        # temp_path = os.path.join(os.path.dirname(thumbnail_file), "temporary" + str(time()) + original_extension)
        img.save(thumbnail_file, quality=quality, optimize=True, format='JPEG')

        while os.path.getsize(thumbnail_file) > max_size_bytes:
            quality -= 5
            img.save(thumbnail_file, quality=quality, optimize=True)

            if quality <= 10:
                break

        # thumbnail_file = temp_path
    
    return thumbnail_file

def convert_image_to_1920x1080(input_image_path, output_image_path):
    desired_aspect_ratio = 1920 / 1080  # Aspect ratio of 16:9
    with Image.open(input_image_path) as img:
        original_width, original_height = img.size
        original_aspect_ratio = original_width / original_height
        
        # Calculate the cropping box dimensions
        if original_aspect_ratio > desired_aspect_ratio:
            # The image is wider than needed, crop width
            new_height = original_height
            new_width = int(desired_aspect_ratio * new_height)
            left = (original_width - new_width) / 2
            top = 0
            right = (original_width + new_width) / 2
            bottom = original_height
        else:
            # The image is taller than needed, crop height
            new_width = original_width
            new_height = int(new_width / desired_aspect_ratio)
            left = 0
            top = (original_height - new_height) / 2
            right = original_width
            bottom = (original_height + new_height) / 2
        
        # Crop the image to the calculated box
        img_cropped = img.crop((left, top, right, bottom))
        
        # Resize the image to 1920x1080 if it is not already the desired size
        if img_cropped.size != (1920, 1080):
            img_cropped = img_cropped.resize((1920, 1080), Image.LANCZOS)
        
        # Save the cropped and resized image
        img_cropped.save(output_image_path)

def generate_thumbnail(thumbnail_prompt, directory):

    client = openai.Client(api_key=(open(os.getcwd()+'\\openai_key.txt').read()).strip())

    try:
        generated_image = client.images.generate(
            model="dall-e-3",
            prompt="Make a thumbnail for the following topic: "+thumbnail_prompt+". The thumbnail needs to stand out as irresistibly clickable among a sea of others, compelling viewers to choose it over the rest. Make sure to use strong, bright and contrasting colours. Make sure to use a close-up view. IMPORTANT: Do not put any words on the thumbnail. The thumbnail must be without any platform-specific branding or overlays. Some more pointers: - The thumbnail and title complement eachother - To make them as clear as possible its smart to not put more then 2 objects in them - Make it bright - Facial reactions arent a must, but people use them to awaken emotional reactions: emotion is powerful. Widescreen aspect ratio. Close-up view.",
            n=1,
            size="1792x1024",
        )

        response = requests.get(generated_image.data[0].url)
        thumbnail_path = os.path.join(directory, 'thumbnail.png')

        with open(thumbnail_path, "wb") as file:
            file.write(response.content)
    
        return thumbnail_path

    except Exception as thumbnail_error:
        print(thumbnail_error)
        return False

def is_loosely_grayscale(image_path, sample_size=1000, color_tolerance=30):
    """
    Check if the image is loosely grayscale by sampling some pixels across the image.
    `sample_size` is the number of pixels to sample.
    `color_tolerance` is the allowed difference between RGB values.
    Returns True if most of the sampled pixels fall within the grayscale tolerance.
    """
    img = Image.open(image_path)
    pixels = np.array(img.convert('RGB'))
    
    # Sample a set of pixels from the image
    sample_indices = np.random.choice(pixels.shape[0] * pixels.shape[1], sample_size, replace=False)
    sample_pixels = pixels.reshape(-1, 3)[sample_indices]
    
    # Calculate the color tolerance for each sampled pixel
    max_diffs = np.max(sample_pixels, axis=1) - np.min(sample_pixels, axis=1)
    
    # Determine if the majority of sampled pixels are within the color tolerance
    grayscale_pixels = max_diffs < color_tolerance
    return np.sum(grayscale_pixels) > sample_size * 0.9
 
def delete_if_not_grayscale(image_path):
    """
    Deletes the image if it's not grayscale.
    """
    try:
        if not is_loosely_grayscale(image_path):
            os.remove(image_path)
            print(f"Deleted '{image_path}' as it is not grayscale.")
    except UnidentifiedImageError as e:
        os.remove(image_path)

def paste_text_with_background(image_path, text):
    # Load the image
    text = text.lower()
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Specify font and size (adjust the path to your font file)
    font = ImageFont.truetype("SitkaVF-Italic.ttf", size=30)
    
    # Calculate text size and rectangle size
    text_width, text_height = textsize(text, font=font)
    rectangle_margin = 10  # Margin between the text and rectangle border
    rectangle_size = (text_width + 2 * rectangle_margin, text_height + 2 * rectangle_margin)

    # Calculate rectangle position (top right, 10px from the right edge)
    image_width, image_height = image.size
    rectangle_x = image_width - rectangle_size[0] - 20
    rectangle_y = 20  # 10 pixels from the top
    
    # Draw the black rectangle
    draw.rectangle([rectangle_x, rectangle_y, rectangle_x + rectangle_size[0], rectangle_y + rectangle_size[1]], fill="gray")

    # Draw the text over the rectangle
    text_x = rectangle_x + rectangle_margin
    text_y = rectangle_y + rectangle_margin
    draw.text((text_x, text_y), text, fill="white", font=font)
    
    # Save or display the image
    image.save(image_path)

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def put_bait_text_on_thumbnail(image_path, text):
    
    position=random.choice(['top_middle', 'bottom_middle'])
    text = text.replace('"', '')

    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image, 'RGBA')

    font_size = min(image.size) // 4  # Example: 1/10th of the smallest image dimension
    font_size = int(font_size * (12 / max(len(text), 12)))
    font = ImageFont.truetype("Bangers-Regular.ttf", font_size)

    # Calculate text size
    print(text)
    print(font)
    text_width, text_height = textsize(text, font=font)

    # Determine position and angle
    if position == 'top_middle':
        x = (image.width - text_width) // 2
        y = (image.height)//22
    elif position == 'bottom_middle':
        x = (image.width - text_width) // 2
        y = image.height - text_height - (image.height//22)
    
    # Create a transparent image for the text
    text_image = Image.new('RGBA', image.size, (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)

    # Draw text with a black outline
    outline_range = max(1, font_size // 15)  # Adjust the outline thickness based on font size

    for adj in range(-outline_range, outline_range+1):
        for adjy in range(-outline_range, outline_range+1):
            text_draw.text((x+adj, y+adjy), text, font=font, fill="black")

    # text_draw.text((x, y), text, font=font, fill="white")

    # Draw main text in white
    text_draw.text((x, y), text, font=font, fill="white")

    # Rotate text image and paste onto original image
    image.paste(text_image, (0, 0), text_image)

    # Save or display the image
    # image.show()
    image.save(image_path)

def colourize_image(input_directory, notification):

    mid_output = r'C:\Users\samlb\OneDrive\InterComputer\ColorizeNet\Input Images'
    mid_input = r'C:\Users\samlb\OneDrive\InterComputer\ColorizeNet\Output Images'

    number_of_inputs = len(os.listdir(input_directory))

    for file in glob.glob(os.path.join(mid_input, '*')):
        if os.path.isfile(file):
            os.remove(file)
    
    for file in glob.glob(os.path.join(mid_output, '*')):
        if os.path.isfile(file):
            os.remove(file)

    # Iterate over all files in the source directory
    for filename in os.listdir(input_directory):
        source_path = os.path.join(input_directory, filename)
        destination_path = os.path.join(mid_output, filename)
        
        # Check if it's a file and not a directory
        if os.path.isfile(source_path):
            # Copy each file to the destination folder
            shutil.copy(source_path, destination_path)

    # Set the details of the remote server and the script you want to run
    port = 22  # default SSH port is 22
    username = 'samlb'
    password=''
    hostname = ''

    # Initialize the SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print('x')
    try:
        # Connect to the server
        client.connect(hostname, port, username, password)
        
        # Execute the Python script on the remote server
        stdin, stdout, stderr = client.exec_command(r'python "C:\Users\samlb\OneDrive\InterComputer\ColorizeNet\ColorizeNet-main\colorize.py" "'+mid_output+'" "'+mid_input+'"')

        # Read the output from the remote script
        print(stdout.read().decode())
        print(stderr.read().decode())

    finally:
        # Close the SSH connection
        client.close()
    
    for file in glob.glob(os.path.join(input_directory, '*')):
        if os.path.isfile(file):
            os.remove(file)
    print('y')
    # Iterate over all files in the source directory
    for filename in os.listdir(mid_input):
        send_notification(notification, "ColourizeNet", str(filename))
        source_path = os.path.join(mid_input, filename)
        destination_path = os.path.join(input_directory, filename)
        
        # Check if it's a file and not a directory
        if os.path.isfile(source_path):
            # Copy each file to the destination folder
            shutil.copy(source_path, destination_path)