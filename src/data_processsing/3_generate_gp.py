"""
    - This code generate the general preferences for each user.
    - Optimized with async processing for faster execution
"""

import time
import asyncio
import aiohttp
from typing import Dict, List
import json
import os
import openai
from openai import AsyncOpenAI
import random
from tqdm import tqdm
import tiktoken
from dotenv import load_dotenv

#####################
core = 10
datasets = ["redial", "opendialkg"]
samples = 500
# Max concurrent API calls to prevent rate limiting
MAX_CONCURRENT_REQUESTS = 50
#####################

prompt_path = 'prompts/gp_generation_prompt.txt'

# --- Environment Setup & OpenAI Initialization ---
# Determine the script's directory and the base PEPPER directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming .env.local is in the PEPPER directory, which is ../../ from this script's location
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env.local')

if not os.path.exists(dotenv_path):
    print(f"Warning: .env.local not found at {dotenv_path}. OpenAI API key may not be loaded.")
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL: OPENAI_API_KEY not found in environment. Please set it in .env.local.")
    exit("Exiting due to missing OpenAI API key.")

# Initialize async client
async_client = AsyncOpenAI(api_key=api_key)
# --- End Environment Setup ---

TOTAL_INPUT_TOKENS = 0  
TOTAL_OUTPUT_TOKENS = 0 
PRICE_PER_1K_INPUT = 0.00015  
PRICE_PER_1K_OUTPUT = 0.0006

# Semaphore to limit concurrent API calls
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def get_persona_generation_prompt() -> str:
    # Construct full path relative to project root if prompt_path is relative
    full_prompt_path = os.path.join(project_root, prompt_path)
    
    if os.path.exists(full_prompt_path):
        with open(full_prompt_path, 'r') as file:
            prompt = file.read()
    return prompt

def count_tokens(text: str, model: str = 'gpt-4o-mini') -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
    
def get_data(data_path) -> Dict:
    with open(data_path, 'r') as file:
        data = json.load(file)
    return data

async def call_chat_completion_async(messages, user_id):
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS
    
    # Use semaphore to limit concurrent requests
    async with semaphore:
        for attempt in range(5):
            try:
                params = {
                    "model": 'gpt-4o-mini',
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 512,
                }
                
                response = await async_client.chat.completions.create(**params)
                
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                
                TOTAL_INPUT_TOKENS += prompt_tokens
                TOTAL_OUTPUT_TOKENS += completion_tokens
                
                return user_id, response.choices[0].message.content
                
            except Exception as e:
                if attempt >= 4:
                    print(f"Failed after 5 attempts for user {user_id}: {str(e)}")
                    return user_id, None
                else:
                    backoff_time = min(1 * 2**attempt, 60)
                    print(f"Error for user {user_id}: {str(e)}, retrying in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)

# Create a batch of user tasks
async def process_users_batch(prompt, user_data_batch):
    tasks = []
    for user_id, user_data in user_data_batch:
        seen_movies = user_data["seen movies"]
        seen_genres = user_data["seen genres"]
        likes = user_data["likes"]
        dislikes = user_data["dislikes"]
        
        messages = [{
            "role": "system", 
            "content": prompt.format(seen_movies=seen_movies, seen_genres=seen_genres, likes=likes, dislikes=dislikes)
        }]
        
        task = call_chat_completion_async(messages, user_id)
        tasks.append(task)
    
    # Process all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)
    return results

async def main():
    prompt = get_persona_generation_prompt()
    
    for dataset in datasets:
        data_path = f'dataset/user_data/{dataset}/2_{dataset}_{core}_split.json'
        data = get_data(data_path)
        
        # Randomly sample users
        user_keys = random.sample(list(data.keys()), samples)
        data = {user_id: data[user_id] for user_id in user_keys}
        
        # Create user_data_batch as list of (user_id, user_data) tuples
        user_data_batch = [(user_id, user_data) for user_id, user_data in data.items()]
        
        print(f"Processing {len(user_data_batch)} users from {dataset} dataset...")
        
        # Store results for all users
        user_persona = {}
        
        # Process in batches to avoid overwhelming RAM
        batch_size = 50  # Adjust based on your system capacity
        for i in range(0, len(user_data_batch), batch_size):
            current_batch = user_data_batch[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(user_data_batch) + batch_size - 1)//batch_size}")
            
            results = await process_users_batch(prompt, current_batch)
            
            # Process results
            for user_id, persona in results:
                if persona:
                    user_data = data[user_id]
                    user_persona[user_id] = {
                        "average rating": user_data["average rating"],
                        "seen/target": user_data["seen/target"],
                        "persona": persona,
                        "seen movies": user_data["seen movies"],
                        "seen genres": user_data["seen genres"],
                        "target movies": user_data["target movies"],
                        "likes": user_data["likes"],
                        "dislikes": user_data["dislikes"],
                        "seen": user_data["seen"],
                        "target": user_data["target"]
                    }
        
        os.makedirs(f'dataset/user_data/_ready/{dataset}/', exist_ok=True)
        output_path = f'dataset/user_data/_ready/{dataset}_user_{core}_{samples}.json'
        with open(output_path, 'w') as json_file:
            json.dump(user_persona, json_file, indent=4)

        TOTAL_COST = (TOTAL_INPUT_TOKENS/1000) * PRICE_PER_1K_INPUT + (TOTAL_OUTPUT_TOKENS / 1000) * PRICE_PER_1K_OUTPUT
        print(f"Total API Cost: ${TOTAL_COST:.6f} USD")
        print(f"Saved data to {output_path}")

if __name__ == '__main__':
    asyncio.run(main())
    
    
    
    