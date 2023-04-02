# Topic2Video

Topic2Video is a Python application that generates a video based on a user-inputted topic. The project utilizes natural language processing (NLP) and machine learning (ML) algorithms to analyze and synthesize information about the topic and then generate a video essay. The resulting video provides a concise, engaging, and informative overview of the topic.

## Requirements

- Python 3.6 or higher
- FFmpeg

## Installation

1. Clone the repository:

```
git clone https://github.com/samuelbraun04/Topic2Video.git
```

2. Navigate to the project directory:

```
cd Topic2Video
```

3. Install the required Python packages:

```
pip install -r requirements.txt
```

4. Install FFmpeg. On macOS, you can install it using Homebrew: ```brew install ffmpeg```. On Windows, you can download a binary from the official website: https://ffmpeg.org/download.html

5. Set up the OpenAI API key:

    To get an OpenAI API key, you need to create an account on the OpenAI website and then follow these steps:

    * Log in to your OpenAI account at https://beta.openai.com/login/.
    * Once you are logged in, click on your username in the top right corner of the screen and select "Dashboard" from the dropdown menu.
    * On the Dashboard page, click on the "Create API Key" button.
    * Enter a name for your API key, such as "My App Key", and select the permissions you want to grant to the key.
    * Click on the "Create API Key" button to generate your key.
    * Your API key will be displayed on the next screen. Copy the key and store it in a safe place.

    Once you have your API key, you can use it to access OpenAI's APIs and services. Input it in Runner.py at line 5:

    ```
    runner = TopicToVideo('OPEN_API_KEY', 'AWS_ACCESS', 'AWS_SECRET')
    ```

6. Set up the AWS Access Key and AWS Secret Access Key:

    To obtain an AWS access key ID and secret access key, you will need to follow these steps:

    * Go to the AWS Management Console and sign in to your account.
    * Click on your username in the top-right corner of the screen, and select "My Security Credentials" from the dropdown menu.
    * Click on the "Access keys" section in the left-hand menu.
    * Click "Create New Access Key" to generate a new access key ID and secret access key.
    * Download the new access key file and store it in a safe place. Be sure to keep the secret access key confidential, as it provides full access to your AWS account.

    Note: If you already have two access keys associated with your AWS account, you will need to delete one before creating a new one. AWS allows a maximum of two active access keys per user at any given time.

    Once you have your API keys, you can use it to access AWS's APIs and services. Input it in Runner.py at line 5:

    ```
    runner = TopicToVideo('OPEN_API_KEY', 'AWS_ACCESS', 'AWS_SECRET')
    ```

7. Run the project by running the Runner.py file. It will use the top topic in the listOfTopics.txt file.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the GPL License. See the LICENSE file for details.