import os
import discord
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from dotenv import load_dotenv
import random

# Load environment variables from .env file
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN not found in environment variables.")

# Load the model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up intents and bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Ensure this intent is enabled
bot = commands.Bot(command_prefix='!', intents=intents)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conversation context to maintain history
conversation_history = {}

@bot.event
async def on_ready():
    logger.info(f'We have logged in as {bot.user}')
    await bot.change_presence(activity=discord.Game(name="Chatting with users!"))

# Helper function to add personality to responses
def add_personality(response_text):
    responses = [
        "🤖 AI says: ",
        "🧙‍♂️ Wizard AI whispers: ",
        "✨ Here's a thought: ",
        "💬 AI responds: ",
        "🔮 Mystic AI reveals: ",
        "💡 Insightful AI shares: "
    ]
    personality_response = random.choice(responses) + response_text + " 😊"
    return personality_response

# Helper function to manage conversation history length
def manage_conversation_history(user_id, new_entry, max_length=10):
    history = conversation_history.get(user_id, [])
    history.append(new_entry)
    if len(history) > max_length:
        history = history[-max_length:]
    conversation_history[user_id] = history
    return history

@bot.command(name='ask')
async def ask(ctx, *, query: str):
    user_id = ctx.author.id
    logger.info(f'Received message from {ctx.author}: {query}')
    
    try:
        # Tokenize the input query
        inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors="pt")

        # Manage conversation history
        history = manage_conversation_history(user_id, inputs)

        # Concatenate history and generate a response
        inputs = torch.cat(history, dim=-1)
        outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

        # Save the response in history
        response_inputs = tokenizer.encode(response_text + tokenizer.eos_token, return_tensors="pt")
        manage_conversation_history(user_id, response_inputs)

        async with ctx.typing():
            await ctx.send(add_personality(response_text))
            logger.info('Message sent!')
    except Exception as e:
        logger.error(f'Error: {e}')
        await ctx.send('Sorry, there was an error processing your request. Please try again later.')

@bot.command(name='joke')
async def joke(ctx):
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the computer go to the doctor? Because it had a virus!",
        "How do robots pay for things? With cache!",
        "Why was the math book sad? Because it had too many problems!",
        "Why did the scarecrow become a successful neurosurgeon? Because he was outstanding in his field!"
    ]
    response = random.choice(jokes)
    await ctx.send(f"😂 Here's a joke for you: {response}")

@bot.command(name='quote')
async def quote(ctx):
    quotes = [
        "The only limit to our realization of tomorrow is our doubts of today. - Franklin D. Roosevelt",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "Do not watch the clock. Do what it does. Keep going. - Sam Levenson",
        "Believe you can and you're halfway there. - Theodore Roosevelt",
        "The best way to predict the future is to create it. - Peter Drucker"
    ]
    response = random.choice(quotes)
    await ctx.send(f"🌟 Inspirational Quote: {response}")

@bot.command(name='clear_history')
async def clear_history(ctx):
    user_id = ctx.author.id
    conversation_history[user_id] = []
    await ctx.send("🧹 Your conversation history has been cleared!")

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send("🏓 Pong! The bot is responsive.")

@bot.command(name='fact')
async def fact(ctx):
    facts = [
        "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
        "Octopuses have three hearts. Two pump blood to the gills, while the third pumps it to the rest of the body.",
        "A day on Venus is longer than a year on Venus. It takes Venus 243 Earth days to rotate once on its axis, but only 225 Earth days to orbit the Sun.",
        "Bananas are berries, but strawberries aren't. Botanically, a berry has seeds and pulp produced from the ovary of a single flower.",
        "Humans share approximately 60% of their DNA with bananas."
    ]
    response = random.choice(facts)
    await ctx.send(f"📚 Did you know? {response}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CommandNotFound):
        await ctx.send('Oops! That command was not found. Type `!help` to see the list of available commands.')
    elif isinstance(error, commands.errors.MissingRequiredArgument):
        await ctx.send('It looks like you missed a required argument. Please check the command and try again.')
    else:
        logger.error(f'Unexpected error: {error}')
        await ctx.send('An unexpected error occurred. Please try again later.')

bot.run(DISCORD_TOKEN)
