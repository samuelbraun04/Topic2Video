from TopicToVideo import TopicToVideo

#create instance of TopicToVideo
runner = TopicToVideo('OPEN_API_KEY', 'AWS_ACCESS', 'AWS_SECRET')

#clean the working directories
runner.cleanDirectories()
print("Directories cleaned.")

#get the first topic from the list of topics
chosenTopic, nextRoundOfTopics = runner.getTopic()
chosenTopic = chosenTopic.strip()
print("Topic gotten.")

#generate the script using AI
prompt = [
    {"role" : "user", "content": "Write me an essay of at least 1000 words on the following subject: "+chosenTopic.strip()}
]
videoScript = runner.chatGPT(prompt)
videoScript = videoScript.replace('essay', 'video')
print("Script gotten.")

#get a shortened version of the topic to search for relevant images
shortenedPrompt = [
    {"role" : "user", "content": "shorten the topic "+chosenTopic+" into a few words (under 50 characters)"}
]
shortenedTopic = runner.chatGPT(shortenedPrompt)
shortenedTopic = shortenedTopic.strip()
print("Shortened Topic gotten.")

#search for relevant images
imageData = runner.getImages(shortenedTopic, ' [High-Relevance Image]')
print("Images gotten.")

#generate the video
videoFile = runner.getVideo(videoScript, imageData)
print("Video made.")

#remove the used topic
runner.removeUsedTopic(nextRoundOfTopics)