import os
import discord
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from dotenv import load_dotenv

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
    await bot.change_presence(activity=discord.Game(name="Ask me anything!"))

@bot.command(name='ask')
async def ask(ctx, *, query: str):
    user_id = ctx.author.id
    logger.info(f'Received message from {ctx.author}: {query}')
    
    try:
        # Retrieve conversation history for user
        history = conversation_history.get(user_id, [])
        
        # Tokenize the input query
        inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors="pt")
        history.append(inputs)

        # Concatenate history and generate a response
        inputs = torch.cat(history, dim=-1)
        outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

        # Save the response in history
        response_inputs = tokenizer.encode(response_text + tokenizer.eos_token, return_tensors="pt")
        history.append(response_inputs)
        conversation_history[user_id] = history

        await ctx.send(response_text)
        logger.info('Message sent!')
    except Exception as e:
        logger.error(f'Error: {e}')
        await ctx.send('Sorry, there was an error processing your request. Please try again later.')

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
