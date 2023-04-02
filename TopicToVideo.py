from boto3 import client
from icrawler.builtin import GoogleImageCrawler
from math import ceil
from moviepy.editor import *
from numpy import array
from os import listdir, getcwd, remove, rename
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from random import randint, shuffle
from shutil import copy 
from string import ascii_letters
from time import sleep
import openai
import textwrap

class TopicToYoutube:

    def __init__(self, OPEN_API_KEY):
        
        self.OPEN_API_KEY = OPEN_API_KEY

        self.CONJOINER = '\\'
        self.OVERALL_DIRECTORY = getcwd()+self.CONJOINER
        self.INGREDIENTS_PATH = self.OVERALL_DIRECTORY+'Ingredients'+self.CONJOINER
        self.WORKSPACE_PATH = self.OVERALL_DIRECTORY+'Workspace'+self.CONJOINER
        self.IMAGES_PATH = self.WORKSPACE_PATH+'Images'+self.CONJOINER
        self.AUDIO_PATH = self.WORKSPACE_PATH+'Audio'+self.CONJOINER
        self.AWS_PATH = self.OVERALL_DIRECTORY+'AWS'+self.CONJOINER

        self.FONT = self.INGREDIENTS_PATH+'RobotoSerifMedium.ttf'
        self.MUSIC_PATH = self.INGREDIENTS_PATH+'Music'+self.CONJOINER
        self.LIST_OF_TOPICS = self.INGREDIENTS_PATH+'listOfTopics.txt'
        
        KEY_ID = open(self.AWS_PATH+'aws_access.txt').read()
        SECRET_ID = open(self.AWS_PATH+'aws_secret.txt').read()
        self.polly = client('polly', region_name='us-east-1', aws_access_key_id=KEY_ID , aws_secret_access_key=SECRET_ID)
        self.s3client = client('s3', aws_access_key_id=KEY_ID, aws_secret_access_key=SECRET_ID)

    def chatGPT(self, prompt, chosenTemperature=1):

        openai.api_key = self.OPEN_API_KEY
        sleepTime=2
        while(1):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature = chosenTemperature,
                    messages = prompt
                )
                break
            except Exception as e:
                print(str(e))
                print("Error, sleeping for "+str(sleepTime)+" seconds.")
                sleep(sleepTime)
                sleepTime*=2
        script = ''
        for choice in response.choices:
            script += choice.message.content
        
        return script
    
    def getImages(self, searchTerm, termForImages):
            
        google_Crawler = GoogleImageCrawler(storage = {'root_dir': self.IMAGES_PATH})
        google_Crawler.crawl(keyword = searchTerm, max_num = 15)

        files = listdir(self.IMAGES_PATH)
        imageData = {
            files[0] : searchTerm+termForImages, 
            files[1] : searchTerm+termForImages,
            files[2] : searchTerm+termForImages,
            files[3] : searchTerm+termForImages,
            files[4] : searchTerm+termForImages
        }

        return imageData

    def getThumbnail(self, thumbnail, specialText):
    
        copy(thumbnail, self.WORKSPACE_PATH+'thumbnail.jpg')
        thumbnail = self.WORKSPACE_PATH+'thumbnail.jpg'

        specialText = specialText.replace('-', ' ')
        specialText = specialText.title()
        specialText = textwrap.wrap(specialText, width=10)
        trueText = ""
        for line in range(len(specialText)):
            trueText+=specialText[line]
            if line != (len(specialText)-1):
                trueText+='\n'
        specialText = trueText

        fnt = ImageFont.truetype(self.FONT, 120)
        img = Image.open(thumbnail)
        rgbimg = Image.new("RGBA", img.size)
        rgbimg.paste(img)
        rgba_or_p_im = rgbimg
        if rgba_or_p_im.mode in ["RGBA", "P"]:
            rgb_im = rgba_or_p_im.convert("RGB")
            rgb_im.save(thumbnail)
        else:
            rgbimg.save(thumbnail)

        img = Image.open(thumbnail)
        ratio = 1280/img.width
        img = img.resize((1280, int(img.height*ratio)))
        if img.height >= 720:
            img = img.crop((0, (img.height-720)/2, img.width, img.height-((img.height-720)/2)))
        img.save(thumbnail)

        return thumbnail

    def centerText(self, size, bgImg, message, font, fontColor):
        W, H = size
        draw = ImageDraw.Draw(bgImg)
        _, _, w, h = draw.textbbox((0, 0), message, font=font)
        draw.text(((W-w)/2, (H-h)/2), message, font=font, fill=fontColor)
        return bgImg

    def getVideo(self, script, imageData):
    
        script = script.replace('.', '. ')
        scriptFile = self.textToSpeech(script)
        scriptSegment = AudioSegment.from_file(scriptFile)
        scriptSegment = scriptSegment.fade_out(100)
        scriptSegment = scriptSegment + 6

        musicFiles = listdir(self.MUSIC_PATH)
        backgroundMusicSegment = AudioSegment.from_file(self.MUSIC_PATH+musicFiles[randint(0, len(musicFiles)-1)])
        backgroundMusicSegment = backgroundMusicSegment - 16
        backgroundMusicSegment = backgroundMusicSegment.fade_out(5000)

        mainAudioSegment = scriptSegment.overlay(backgroundMusicSegment, loop=True)
        audioLength = mainAudioSegment.duration_seconds
        mainAudioSegment.export(self.WORKSPACE_PATH+'final.mp3', format='mp3')

        imageClips = []
        formatter = {"PNG": "RGBA", "JPEG": "RGB"}
        lengthPerClip = audioLength/len(listdir(self.IMAGES_PATH))
        fnt = ImageFont.truetype(self.FONT, 25)

        for image in listdir(self.IMAGES_PATH):
            if image in imageData:
                newHeight = 600
            else:
                newHeight = 720
            
            newImage = Image.open(self.IMAGES_PATH+image)
            newImage = newImage.resize((ceil(newHeight*(newImage.width/newImage.height)), newHeight), Image.ANTIALIAS)
            if newImage.width <= 1080:
                blackImage = Image.new(formatter.get(newImage.format, 'RGB'), size=(1080,720), color='black')
                blackImage.paste(newImage, ((int(blackImage.width/2)-int(newImage.width/2)), 0))
                blackImage.save(self.IMAGES_PATH+image)
            elif newImage.width > 1080:
                newImage = newImage.crop((0, 0, 1080, newImage.height))
                rgba_or_p_im = newImage
                if rgba_or_p_im.mode in ["RGBA", "P"]:
                    rgb_im = rgba_or_p_im.convert("RGB")
                    rgb_im.save(self.IMAGES_PATH+image)
                else:
                    newImage.save(self.IMAGES_PATH+image)

            if image in imageData:
                oldImage = Image.open(self.IMAGES_PATH+image)
                draw = ImageDraw.Draw(oldImage)
                avg_char_width = sum(fnt.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
                max_char_count = int(oldImage.size[0]*1.15 / avg_char_width)
                text = textwrap.fill(text=imageData[image], width=max_char_count)
                draw.text(xy=(oldImage.size[0]/2, 650), text=text, font=fnt, fill=(255,255,255), anchor='mm')
                oldImage.save(self.IMAGES_PATH+image)

            img = Image.open(self.IMAGES_PATH+image)
            rgbimg = Image.new(formatter.get(img.format, 'RGB'), img.size)
            rgbimg.paste(img)
            rgbimg.save(self.IMAGES_PATH+image, format=img.format)

            currentImageClip = ImageClip(self.IMAGES_PATH+image)
            currentImageClip = currentImageClip.with_duration(lengthPerClip)
            if image not in imageData:
                currentImageClip = self.zoom_in_effect(currentImageClip, 0.03)
            imageClips.append(currentImageClip)
    
        shuffle(imageClips)
        finalVideo = concatenate_videoclips(imageClips, method='compose')
        finalVideo.audio = AudioFileClip(self.WORKSPACE_PATH+'final.mp3')
        completeFinalVideo = concatenate_videoclips([finalVideo, VideoFileClip(self.INGREDIENTS_PATH+'outro.mp4')], method='compose')
        completeFinalVideo.write_videofile(self.WORKSPACE_PATH+'finalVideo.mp4', fps=24)

        return self.WORKSPACE_PATH+'finalVideo.mp4'

    def randomSleep(self):
        sleep(randint(2,4)*(randint(500,1000)/1000))
    
    def getTopic(self):

        topics = open(self.LIST_OF_TOPICS, 'r', encoding='utf-8').readlines()
        for topic in range(len(topics)):
            topics[topic] = topics[topic]
        chosenTopic = topics[0].strip()
        topics = topics[1:]

        return chosenTopic, topics
    
    def removeUsedTopic(self, topics):
        open(self.LIST_OF_TOPICS, 'w', encoding='utf-8').writelines(topics)

    def textToSpeech(self, TTStext):
        
        startResponse = self.polly.start_speech_synthesis_task(Engine='neural', OutputS3BucketName='braunbucket2004', Text=TTStext, OutputFormat='mp3', VoiceId='Amy')
        taskID = startResponse['SynthesisTask']['TaskId']

        while(1):
            getResponse = self.polly.get_speech_synthesis_task(TaskId=taskID)
            sleep(5)
            if getResponse['SynthesisTask']['TaskStatus'] == 'completed':
                break
        
        filename = getResponse['SynthesisTask']['TaskId']+'.mp3'
        self.s3client.download_file('braunbucket2004', filename, self.WORKSPACE_PATH+filename)
        rename(self.WORKSPACE_PATH+filename, self.WORKSPACE_PATH+'main.mp3')
        return self.WORKSPACE_PATH+'main.mp3'

    def zoom_in_effect(self, clip, zoom_ratio=0.04):
        def effect(get_frame, t):
            img = Image.fromarray(get_frame(t))
            base_size = img.size

            new_size = [
                ceil(img.size[0] * (1 + (zoom_ratio * t))),
                ceil(img.size[1] * (1 + (zoom_ratio * t)))
            ]
            
            new_size[0] = new_size[0] + (new_size[0] % 2)
            new_size[1] = new_size[1] + (new_size[1] % 2)

            img = img.resize(new_size, Image.LANCZOS)

            x = ceil((new_size[0] - base_size[0]) / 2)
            y = ceil((new_size[1] - base_size[1]) / 2)

            img = img.crop([
                x, y, new_size[0] - x, new_size[1] - y
            ]).resize(base_size, Image.LANCZOS)

            result = array(img)
            img.close()

            return result

        return clip.transform(effect)

    def cleanDirectories(self):

        try:
            remove(self.WORKSPACE_PATH+'main.mp3')
        except:
            pass

        try:
            remove(self.WORKSPACE_PATH+'final.mp3')
        except:
            pass

        try:
            remove(self.WORKSPACE_PATH+'finalVideo.mp4')
        except:
            pass

        try:
            remove(self.WORKSPACE_PATH+'thumbnail.jpg')
        except:
            pass

        for image in listdir(self.IMAGES_PATH):
            try:
                remove(self.IMAGES_PATH+image)
            except:
                pass