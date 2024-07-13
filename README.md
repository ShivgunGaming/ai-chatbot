# Personality Discord Bot

## Overview
This Personality Discord Bot is designed to interact with users in a fun and engaging way. It uses the DialoGPT model from Hugging Face to generate responses and includes various commands to entertain and inspire users.

## Features

### Core Commands
- `!ask <query>`: The bot answers questions or responds to prompts in a conversational manner, with added personality and a friendly tone.
- `!joke`: The bot shares a random joke to lighten the mood.
- `!quote`: The bot provides an inspirational quote to motivate and uplift users.

### Personality Enhancements
- **Typing Indicator**: The bot simulates thinking by showing a typing indicator before sending a response.
- **Varied Responses**: Responses are prefixed with different phrases to add variety and personality, making the interaction more dynamic.

## How It Works
The bot uses the DialoGPT-medium model from Hugging Face for generating conversational responses. It maintains a history of interactions to provide contextually relevant answers, ensuring a more coherent and continuous conversation with users.

### Example Usage
- **Ask a question**:
    ```
    User: !ask What's the weather like today?
    Bot: 🤖 AI says: I'm not quite sure about the weather, but I hope it's sunny where you are! 😊
    ```
- **Tell a joke**:
    ```
    User: !joke
    Bot: 😂 Here's a joke for you: Why don't scientists trust atoms? Because they make up everything!
    ```
- **Share an inspirational quote**:
    ```
    User: !quote
    Bot: 🌟 Inspirational Quote: The only limit to our realization of tomorrow is our doubts of today. - Franklin D. Roosevelt
    ```

## Getting Started
To get started with the Personality Discord Bot, you will need:
1. **A Discord Bot Token**: Create a bot on the [Discord Developer Portal](https://discord.com/developers/applications) and get the token.
2. **Python**: Ensure Python 3.6 or higher is installed on your system.
3. **Dependencies**: Install the required dependencies by running `pip install -r requirements.txt`.

After setting up the environment, run the bot using:
```sh
python bottyboy.py
