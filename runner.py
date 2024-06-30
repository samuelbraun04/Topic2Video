test_mode = False

try:

    from AutomatedYoutube import *
    from datetime import datetime
    from google_images_search import GoogleImagesSearch
    from googleapiclient.errors import HttpError
    from math import ceil
    from moviepy.editor import *
    from my_pushover import Pushover
    from PIL import Image
    from pydub import AudioSegment
    from time import time, sleep
    import googleapiclient.discovery
    import googleapiclient.errors
    import inspect
    import openai
    import os
    import random
    import traceback

    start_time = datetime.now()

    #main clients
    if test_mode == False:
        channel_name = sys.argv[1]
    else:
        channel_name = "HistorysDarkestQuestions"
    
    notification = Pushover(open('pushover.txt').readlines()[0].strip())
    notification.user(open('pushover.txt').readlines()[1].strip())

    client = openai.Client(api_key=(open(os.getcwd()+'\\openai_key.txt').read()).strip())
    images_keys = (open(os.getcwd()+'\\googleimages_key.txt', 'r').readlines())
    if (images_keys[2]).strip() == '0':
        images_keys[2] = '1\n'
        google_key = images_keys[0].strip()
        engine_key = images_keys[1].strip()
    elif (images_keys[2]).strip() == '1':
        images_keys[2] = '0\n'
        google_key = images_keys[3]
        engine_key = images_keys[4]
    open(os.getcwd()+'\\googleimages_key.txt', 'w').writelines(images_keys)
    image_engine = GoogleImagesSearch(google_key, engine_key)
    current_time = datetime.now()


    channel_names = {
        "HistorysDarkestQuestions" : 'casio.cdp.240r@gmail.com',
        "AnAlternatePast" : 'earth2hour@gmail.com',
        "ShowerThinkings" : 'samuelbraunhighschool@gmail.com',
    }

    if test_mode == True:
        send_notification(notification, 'TEST MODE ON VIDEO GENERATOR', 'TURN IT OFF')

    #get topic
    main_directory = os.getcwd()
    specific_directory_path = os.path.join(os.getcwd(), channel_name)  # Path to the directory
    topics = open(os.path.join(specific_directory_path, 'topics.txt'), 'r').readlines()
    topic = topics.pop(0)


    images_directory = os.path.join(os.getcwd(), channel_name, 'Images')
    uncropped_images_directory = os.path.join(os.getcwd(), channel_name, 'Uncropped Images')
    audio_directory = os.path.join(os.getcwd(), channel_name, 'Audio')
    music_directory = os.path.join(os.getcwd(), channel_name, 'Music')
    workplace_directory = os.path.join(os.getcwd(), channel_name, 'Workplace')

    script_audio_path = os.path.join(workplace_directory, 'final_script.wav')
    subtitles_path = os.path.join(workplace_directory, 'script.srt')

    topic_audio_path = os.path.join(audio_directory, 'topic_audio.wav')
    channel_audio_path = os.path.join(audio_directory, 'channel_audio.wav')

    temp_topic_audio_path = os.path.join(audio_directory, 'temp_topic_audio.wav')
    temp_channel_audio_path = os.path.join(audio_directory, 'temp_channel_audio.wav')

    description_path = os.path.join(specific_directory_path, 'description.txt')

    topics_path = os.path.join(specific_directory_path, "topics.txt")

    # make exponential growth vi
    exponential_growth_video_path = os.path.join(workplace_directory, 'exponential_growth_video.mp4')
    topic_text_video_path = os.path.join(workplace_directory, 'topic_text.mp4')
    channel_text_video_path = os.path.join(workplace_directory, 'channel_text.mp4')
    
    #topic background image is just the largest image selected 
    topic_background_text_video_path = os.path.join(workplace_directory, 'topic_background.mp4')
    topic_background_text_image_path = os.path.join(workplace_directory, 'topic_background.png')
    channel_background_image_path = os.path.join(specific_directory_path, "dramatic.png")
    channel_background_video_path = os.path.join(workplace_directory, 'channel_background.mp4')
    
    topic_clip_video_path = os.path.join(workplace_directory, 'topic_clip.mp4')
    channel_clip_video_path = os.path.join(workplace_directory, 'channel_clip.mp4')
    ending_clip_video_path = os.path.join(workplace_directory, 'ending_clip.mp4')

    first_audio_transition_path = os.path.join(main_directory, 'snap_1.wav')

    empty_1s_audio_path = os.path.join(workplace_directory, 'empty_1s.wav')
    empty_2s_audio_path = os.path.join(workplace_directory, 'empty_2s.wav')

    clip1_audio_path = os.path.join(workplace_directory, 'clip1_audio.wav')
    clip2_audio_path = os.path.join(workplace_directory, 'clip2_audio.wav')
    clip3_audio_path = os.path.join(workplace_directory, 'clip3_audio.wav')

    background_images_video_path = os.path.join(workplace_directory, 'background_images.mp4')
    foreground_images_video_path = os.path.join(workplace_directory, 'foreground_images.mp4')

    final_video_path = os.path.join(workplace_directory, 'final_video.mp4')

    font_path = (os.path.join(main_directory, "Bangers-Regular.ttf"))
    
    channel_video_path = os.path.join(workplace_directory, 'channel_video.mp4')
    topic_video_path = os.path.join(workplace_directory, 'topic_video.mp4')

    half_a_second_silence_path = os.path.join(main_directory, 'half_a_second_silence.wav')
    full_second_silence_path = os.path.join(main_directory, 'full_second_silence.wav')
    
    audio_files = []
    image_dirs = []
    merged_audio = os.path.join(workplace_directory, 'merged_audio.wav')
    bg_video = os.path.join(workplace_directory, 'bg_video.mp4')
    animation_video = os.path.join(workplace_directory, 'animation_video.mp4')
    intermediate_video = os.path.join(workplace_directory, 'intermediate_video.mp4')
    fg_video = os.path.join(workplace_directory, 'fg_video.mp4')
    final_output = os.path.join(workplace_directory, 'final_output.mp4')
    temp_final_output = os.path.join(workplace_directory, 'temp_final_output.mp4')
    final_final_output = os.path.join(workplace_directory, 'final_final_output.mp4')

    buildup_music_path = os.path.join(main_directory, 'new_new_buildup.wav')
    topic_image_directory = os.path.join(images_directory, "Topic Images")

    # #############################
    # # TO AUTHENTICATE A CHANNEL
    # youtube = load_or_authenticate_channel(channel_name, r"C:\Users\samlb\Documents\Projects\VideoGenerator-v2\client_secret_643590692955-v30jg61vqaue6odc2km5vipni0aopnej.apps.googleusercontent.com.json", specific_directory_path)
    # exit()
    # #############################

    # clean images subdirectories
    delete_everything_in_directory(images_directory)
    delete_everything_in_directory(audio_directory)
    delete_everything_in_directory(workplace_directory)
    [os.remove(file) for file in glob.glob(os.path.join(specific_directory_path, '*.mp4'))]
    
    # main dictionary
    full_dictonary = {}

    # generate full script
    script_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You will be writing for a script for a video essay about a historic topic that will be read out by a TTS. Do not give an opinion or reflect on the subject matter, the sole purpose of this video is to generate dense, shocking content. To maintain viewer retention, make sure to mention or go into factual detail about things that could affect or stir someone's emotions. Have a hard hitting, concise hook at the beginning. Write it in a way that'll keep viewer retention until the very end. Viewer retention is extremely important, do this by any means necessary. Make the conclusion very concise and short. IMPORTANT: Provide a text-only script without any visual or audio directions, formatting, suggestions, labels, headers, titles, or anything at all except the actual paragraph's content. Exact format needed: paragraph_1\n\nparagraph_2\n\nparagraph_3...etc. Provide the text only, with no added commentary, as the output will be fed directly into a program."},
            {"role": "user", "content": "The topic for this video essay is: "+topic+". Aim for 900 words."}
        ]
    )
    string_starting_script = script_response.choices[0].message.content

    starting_script = process_text(string_starting_script)

    for index, x in enumerate(starting_script):
        full_dictonary[index] = [x]
    
    # get image search query for each
    used_keys = []
    if 'Darkest' in channel_name:
        specifics_for_query = 'Exclusively old or historical photographs. Please ensure the query is likely to return results that are clearly from earlier periods, such as specific centuries or notable historical epochs.'
        specifics_for_client = 'Please generate image search queries focusing strictly on factual content and specific events, avoiding any subjective or interpretative terms.'
    if 'Alternate' in channel_name:
        specifics_for_query = 'Exclusively old or historical photographs. Please ensure the query is likely to return results that are clearly from earlier periods, such as specific centuries or notable historical epochs or illustrations of an alternate past.'
        specifics_for_client = 'Please generate image search queries focusing strictly on factual content and specific events, avoiding any subjective or interpretative terms.'
    if 'Shower' in channel_name:
        specifics_for_query = 'Prioritze scientific photographs. When it makes sense for the paragraph, ensure the query is likely to return results that are clearly from scientific sources. If the paragraph is about a noun (person, place, thing, etc) ensure the query is likely to return results that are clearly about that noun (not necessarily scientfic).'
        specifics_for_client = ''

    for key in range(len(full_dictonary)):
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "assistant", "content": "You will be generating a concise, specialized search query that should get only relevant photographs about a paragraph from the google images API. You will be given the paragraph to base the search query off of. "+specifics_for_client+" You will also be given some search queries that you cannot use exactly (they might not even be related but it's just for cautious measures). IMPORTANT: You should only return the search query raw (no quotation marks or anything at all), as I will be feeding your output directly into a program."},
                {"role": "user", "content": "Script: "+full_dictonary[key][0]+". Exact search queries you're not allowed to return: "+str(used_keys)+". Based on the revised instructions, generate a search query based on the paragraph. "+specifics_for_query}
            ]
        )

        sleep(15)

        search_query = (response.choices[0].message.content)
        full_dictonary[key].append(search_query)
        used_keys.append(search_query)

        print('Paragraph: '+full_dictonary[key][0])
        print('Search query: '+search_query+'\n\n')

    voice_model = random.choice(['alloy', 'echo', 'fable', 'onyx', 'shimmer'])

    # generate audio clips
    empty_audio = AudioSegment.empty()
    for index, element in enumerate(full_dictonary):
        audio_filename = os.path.join(audio_directory, str(index)+'_PARAGRAPH.wav')
        temp_audio_filename = os.path.join(audio_directory, str(index)+'_PARAGRAPH_temp.wav')

        text_to_audio_file(full_dictonary[index][0], temp_audio_filename, client, voice_model)
        x = concatenate_audioclips([AudioFileClip(temp_audio_filename), AudioFileClip(half_a_second_silence_path)])
        x.write_audiofile(audio_filename)

        file = AudioSegment.from_file(audio_filename)
        silence_length = ceil(file.duration_seconds) - file.duration_seconds
        file += AudioSegment.silent(duration=ceil(silence_length*1000))
        file.export(audio_filename, format='wav')

        full_dictonary[index].append(audio_filename)
        empty_audio+=file

    script_audio_path = os.path.join(workplace_directory, 'final_script.wav')
    empty_audio.export(script_audio_path, 'wav')
    subtitles_path = script_to_subtitles(script_audio_path, subtitles_path)
    reformat_subtitles(subtitles_path, subtitles_path, 140)

    # make image subdirectory
    create_subdirectories(images_directory, list(full_dictonary.keys()))
    os.makedirs(topic_image_directory, exist_ok=True)

    # get general topic images
    _search_params = {
        'q': topic.strip(),
        'num': 5,
        'safe': 'high',
    }
    search_image_filename = str(time()).replace('.', '')
    image_engine.search(search_params=_search_params, path_to_dir=topic_image_directory, custom_image_name=search_image_filename)


    # scrape the images from the web
    backup_page = 0
    for index, element in enumerate(full_dictonary):
        _search_params = {
            'q': full_dictonary[index][1].strip(),
            'num': 10,
            'safe': 'high',
        }
        search_image_filename = str(time()).replace('.', '')
        sub_image_directory = os.path.join(images_directory, str(index))

        image_engine.search(search_params=_search_params, path_to_dir=sub_image_directory, custom_image_name=search_image_filename)
        full_dictonary[index].append(sub_image_directory)

    remove_unwanted_files(images_directory, {'.jpg', '.png', '.jpeg'})
    if len(os.listdir(topic_image_directory)) == 0:
        copy_random_images(images_directory, topic_image_directory, 5)

    duplicates = find_similar_images(images_directory, [topic_image_directory])
    delete_duplicates(duplicates)
    populate_subfolders_with_images(images_directory, topic_image_directory, [topic_image_directory], 3)

    all_image_clips = find_and_convert_images_to_clips(images_directory)

    # make intro video
    topic_audio_path = text_to_audio_file("Today's video is: "+topic.strip(), topic_audio_path, client, voice_model)
    if channel_name == "HistorysDarkestQuestions":
        channel_name = "History's Darkest Questions"
    if channel_name == 'AnAlternatePast':
        channel_name = "An Alternate Past``````````````````````"
    channel_audio_path = text_to_audio_file("Welcome to "+channel_name, channel_audio_path, client, voice_model)
    
    for key, value in full_dictonary.items():
        audio_files.append(value[2])
        image_dirs.append(value[3])

    # Generate background video
    bg_videos = []
    for audio, images_dir in zip(audio_files, image_dirs):
        process_images_bg_fg(images_dir)
        images_dir = os.path.join(images_dir, 'Background')
        images = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        new_filename = os.path.basename(audio).split('.')[0]
        output_video = os.path.join(workplace_directory, f"bg_{new_filename}.mp4")
        temp_txt_file = os.path.join(workplace_directory, f"bg_{new_filename}.txt")
        temp_audio = AudioFileClip(audio)
        temp_audio_duration = temp_audio.duration
        generate_video_from_images(images, audio, output_video, temp_audio_duration, temp_txt_file, 10)
        bg_videos.append(output_video)

    # Concatenate background videos
    bg_clips = []
    for the_bg_video in bg_videos:
        bg_clips.append(VideoFileClip(the_bg_video))
    concatenate_videoclips(bg_clips).write_videofile(bg_video, codec='hevc_nvenc', audio_codec='aac', fps=30)

    # Generate animation video
    script_length = ceil(AudioFileClip(script_audio_path).duration)
    create_animation_video(animation_video, script_length)

    # Create the topic text clip
    subprocess.run('ffmpeg -i "'+channel_audio_path+'" -i "'+full_second_silence_path+'" -y -filter_complex "[0:0][1:0]concat=n=2:v=0:a=1[out]" -c:v hevc_nvenc -map "[out]" '+temp_channel_audio_path)
    subprocess.run('ffmpeg -i "'+full_second_silence_path+'" -i "'+topic_audio_path+'" -i "'+full_second_silence_path+'" -y -filter_complex "[0:0][1:0][2:0]concat=n=3:v=0:a=1[out]" -c:v hevc_nvenc -map "[out]" '+temp_topic_audio_path)


    if os.path.exists(temp_channel_audio_path) == False:
        raise Exception("Channel audio path does not exist")
    if os.path.exists(temp_topic_audio_path) == False:
        raise Exception("Topic audio path does not exist")

    os.remove(channel_audio_path)
    os.remove(topic_audio_path)
    os.rename(temp_channel_audio_path, channel_audio_path)
    os.rename(temp_topic_audio_path, topic_audio_path)

    channel_audio_clip = AudioFileClip(channel_audio_path)
    topic_audio_clip = AudioFileClip(topic_audio_path) 

    img = resize_image_aspect_ratio_bg_fg(Image.open((os.path.join(topic_image_directory, os.listdir(topic_image_directory)[0]))), 1920, 1080)
    img = img.filter(ImageFilter.GaussianBlur(radius=15))
    img.save(topic_background_text_image_path)

    create_video_with_text_and_audio(channel_background_image_path, channel_audio_path, channel_video_path, wrap_text_simple(channel_name, 40), channel_audio_clip.duration, font_path)
    create_video_with_text_and_audio(topic_background_text_image_path, topic_audio_path, topic_video_path, wrap_text_simple(topic.strip(), 40), topic_audio_clip.duration, font_path)

    intro_music_audio = AudioFileClip(buildup_music_path)
    intro_video = create_exponential_growth_video(all_image_clips, intro_music_audio, intro_music_audio.duration, channel_video_path)
    intro_video.write_videofile(exponential_growth_video_path, codec='hevc_nvenc', audio_codec='aac', fps=30)

    # Overlay animation on background video
    overlay_videos(bg_video, animation_video, intermediate_video)

    # Generate foreground video
    fg_videos = []
    for audio, images_dir in zip(audio_files, image_dirs):
        images_dir = os.path.join(images_dir, 'Foreground')
        images = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        new_filename = os.path.basename(audio).split('.')[0]
        output_video = os.path.join(workplace_directory, f"fg_{new_filename}.mp4")
        temp_txt_file = os.path.join(workplace_directory, f"fg_{new_filename}.txt")
        temp_audio = AudioFileClip(audio)
        temp_audio_duration = temp_audio.duration
        generate_video_from_images(images, audio, output_video, temp_audio_duration, temp_txt_file, 10)
        fg_videos.append(output_video)

    # Concatenate foreground videos
    fg_clips = []
    for the_fg_video in fg_videos:
        fg_clips.append(VideoFileClip(the_fg_video))
    concatenate_videoclips(fg_clips).write_videofile(fg_video, codec='hevc_nvenc', audio_codec='aac', fps=30)

    overlay_videos(intermediate_video, fg_video, final_output)

    overlay_audio_on_video(final_output, os.path.join(music_directory, random.choice(os.listdir(music_directory))), temp_final_output)

    hardcode_subtitles(final_output, subtitles_path, final_final_output, font_path, 12, 1020)

    filename = topic.strip()
    replacements = {
        ":": "_COLON_",
        "#": "_POUND_",
        "%": "_PERCENT_",
        "&": "_AMPERSAND_",
        "{": "_LEFT_CURLY_BRACKET_",
        "}": "_RIGHT_CURLY_BRACKET_",
        "\\": "_BACK_SLASH_",
        "<": "_LEFT_ANGLE_BRACKET_",
        ">": "_RIGHT_ANGLE_BRACKET_",
        "*": "_ASTERISK_",
        "?": "_QUESTION_MARK_",
        "/": "_FORWARD_SLASH_",
        "$": "_DOLLAR_SIGN_",
        "!": "_EXCLAMATION_POINT_",
        "'": "_SINGLE_QUOTE_",
        "\"": "_DOUBLE_QUOTES_",
        "@": "_AT_SIGN_",
        "+": "_PLUS_SIGN_",
        "`": "_BACKTICK_",
        "|": "_PIPE_",
        "=": "_EQUAL_SIGN_"
    }

    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)
    output_video_path = os.path.join(specific_directory_path, filename+'.mp4')

    while(1):
        try:
            concatenate_videoclips([VideoFileClip(exponential_growth_video_path), VideoFileClip(channel_video_path), VideoFileClip(topic_video_path), VideoFileClip(final_final_output)], method='compose').write_videofile(output_video_path, codec='hevc_nvenc', audio_codec='aac', fps=30)
            break
        except Exception as e:
            sleep(1)

    if test_mode == False:

        youtube = load_or_authenticate_channel(channel_name, r"C:\Users\samlb\OneDrive\Projects\VideoGenerator-v2\client_secret_643590692955-v30jg61vqaue6odc2km5vipni0aopnej.apps.googleusercontent.com.json", specific_directory_path)

        print("Applying replacements to the title...")
        replacements = {
            ":": "_COLON_",
            "#": "_POUND_",
            "%": "_PERCENT_",
            "&": "_AMPERSAND_",
            "{": "_LEFT_CURLY_BRACKET_",
            "}": "_RIGHT_CURLY_BRACKET_",
            "\\": "_BACK_SLASH_",
            "<": "_LEFT_ANGLE_BRACKET_",
            ">": "_RIGHT_ANGLE_BRACKET_",
            "*": "_ASTERISK_",
            "?": "_QUESTION_MARK_",
            "/": "_FORWARD_SLASH_",
            "$": "_DOLLAR_SIGN_",
            "!": "_EXCLAMATION_POINT_",
            "'": "_SINGLE_QUOTE_",
            "\"": "_DOUBLE_QUOTES_",
            "@": "_AT_SIGN_",
            "+": "_PLUS_SIGN_",
            "`": "_BACKTICK_",
            "|": "_PIPE_",
            "=": "_EQUAL_SIGN_"
        }
        reversed_replacements = {v: k for k, v in replacements.items()}

        video_tags = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "assistant", "content": "Your task is generate an extremely long, descriptive single paragraph about a video script. The main focus of this paragraph is increase SEO of the video, so mention key words in the descriptive paragraph that people might search. You will be given a script to base long paragraph on. It's important you only return the long descriptive paragraph and nothing else, since I will be feeding your output directly into a program."},
                {"role": "user", "content": "The script to write the paragraph off of: "+string_starting_script}
            ]
        )

        final_description = (((open(description_path, encoding='utf-8').read()).strip()).replace("VIDEO_TITLE", topic.strip()).replace("CHANNEL_NAME", channel_name).replace("RELEVANT_VIDEO_TAGS", (video_tags.choices[0].message.content))+'\n')[:4999]

        #get video ready
        print("\n\n\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n")
        print('Channel name: '+channel_name+'\n')
        print('Video topic: '+topic+'\n\n')
        print('Description: \n:'+final_description)
        print("\n\n\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n")

        body={
            "snippet": {
                "categoryId": "27",
                "description": final_description,
                "title": topic.strip()
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            }
        }

        print("Preparing request for video upload...")
        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=googleapiclient.http.MediaFileUpload(output_video_path, chunksize=-1, resumable=True)
        )

        print(f"Video uploaded.")

        bait_text = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "I need to put some text on a youtube thumbnail that baits users into clicking my video. You will be given the title of the video, so generate a worst case/dramatic 1-3 word question/statement about it.  Make the bait specific to the title. IMPORTANT: Provide the text only, with no added commentary, as the output will be fed directly into a program."},
            {"role": "user", "content": "Title of video: "+topic}
            ]
        )
        bait_text = bait_text.choices[0].message.content

        thumbnail_file = False #temporarily, so we can pick an image

        if thumbnail_file == False:
            status, response = request.next_chunk()        
            video_id = response.get('id')

            print(f"Processing thumbnail for video ID: {video_id}")

            image_files = os.listdir(topic_image_directory)
            thumbnail = os.path.join(topic_image_directory, random.choice(image_files))

            thumbnail = zoom_and_crop_to_aspect_ratio(thumbnail)
            thumbnail = enhance_image_with_vignette(thumbnail)
            put_bait_text_on_thumbnail(thumbnail, bait_text)  

            if os.path.getsize(thumbnail_file) > 2000000:
                thumbnail = reduce_image_size(thumbnail)     

            print(f"Thumbnail file path: {thumbnail}")

            thumbnail_request = youtube.thumbnails().set(
                videoId=video_id,
                media_body=googleapiclient.http.MediaFileUpload(thumbnail)
            )
            thumbnail_response = thumbnail_request.execute()

            print(f"Thumbnail uploaded. Response: {thumbnail_response}")
        else:            
            try:
                status, response = request.next_chunk()
                if 'id' in response:
                    print(f"Video id '{response['id']}' was successfully uploaded.")

                    video_id = response['id']
                    convert_image_to_1920x1080(thumbnail_file, thumbnail_file)

                    put_bait_text_on_thumbnail(thumbnail_file, bait_text)

                    if os.path.getsize(thumbnail_file) > 2000000:
                        thumbnail_file = reduce_image_size(thumbnail_file)

                    thumbnail_request = youtube.thumbnails().set(
                        videoId=video_id,
                        media_body=googleapiclient.http.MediaFileUpload(thumbnail_file)
                    )
                    thumbnail_response = thumbnail_request.execute()
                    print(f"Thumbnail set for video id '{video_id}'")
                else:
                    print(f"The upload failed with an unexpected response: {response}")
            except HttpError as e:
                print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
                response = request.execute()

            print(f"Thumbnail uploaded. File: {thumbnail_file}")

        if 'Darkest' in channel_name:
            playlist_id = 'PLUEHh0XcTcDsW6ixNO0x0WH45Hz-OeCQv'
        elif 'Alternate' in channel_name:
            playlist_id = 'PLUmp-pebytytdLRL8_RnYqPkYP2czIa14'
        elif 'Past' in channel_name:
            playlist_id = 'PLpazy1gYEYr5T85kTZELJ5sKUl7MYdKUB'
        elif 'Shower' in channel_name:
            playlist_id = 'PLj8SaLe_Gdb6yBwJOsKslzRkD6HXSSeJN'
        elif 'Rankists' in channel_name:
            playlist_id = 'PLg49drDO8PhdMnZ8zxEUu6BdXTTZLzXmi'

        playlist_item = { 'snippet': { 'playlistId': playlist_id, 'resourceId': { 'kind': 'youtube#video', 'videoId': response['id'] } } }
        playlist_item = youtube.playlistItems().insert( part='snippet', body=playlist_item ).execute()

        open(topics_path, 'w').writelines(topics)
        topics = open(topics_path, 'r').readlines()

        end_time = datetime.now()
        time_difference = end_time - start_time
        minutes_passed = time_difference.total_seconds() / 60
        print("Minutes passed between the two timestamps:", minutes_passed)

        send_notification(notification, 'VIDUPLOAD: '+channel_name, topic.strip()+'. Time to generate and upload: '+str(minutes_passed)+' - Number of topics left: '+str(len(topics)))
        open(r"C:\Users\samlb\OneDrive\Projects\main_log.txt", 'a+', encoding='utf-8').write(str(datetime.now())+'\n'+str(locals())+'\n\n\n\n')

except Exception as e:
    if test_mode == False:
        debugging = []
        for frame_info in inspect.stack():
            frame = frame_info.frame
            debugging.append(f"Frame {frame_info.function}:\n")
            debugging.append("Local variables:"+str(frame.f_locals)+'\n')
            debugging.append("Global variables:"+str(frame.f_globals)+'\n')
            debugging.append("----------\n\n")
        open(r"C:\Users\samlb\OneDrive\Projects\main_log.txt", 'a+', encoding='utf-8').write(str(datetime.now())+'\n'+traceback.format_exc()+'\n\n'+str(debugging)+'\n\n')
        send_notification(notification, 'VIDERROR: '+channel_name, traceback.format_exc())
        print(traceback.format_exc())
        print(str(debugging))
    else:
        print(traceback.format_exc())
        raise e