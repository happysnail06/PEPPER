import argparse
import copy
import json
import os
import re
import random
import time
import typing
import warnings
import asyncio
import aiofiles
import aiohttp  # If needed for other async operations
import tiktoken
import openai
from tqdm.asyncio import tqdm
import copy

import sys
# Get the absolute path to the 'crs' directory
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
module_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# Add the path to sys.path
sys.path.append(module_path)
sys.path.append(module_path2)

from crs.model.recommender import RECOMMENDER
from src.user_simulator import UserSimulator
from src.utils import *

warnings.filterwarnings('ignore')

# Trial Name
SETTING = 'main'
RATIO = 82
TOTAL_COST = 0

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

async def load_json_async(filepath: str) -> dict:
    async with aiofiles.open(filepath, 'r', encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content)

async def save_json_async(filepath: str, data) -> None:
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))

# Main Method
async def chat(user: str, data: dict, args, recommender: RECOMMENDER, metadata: dict, title_to_id: dict, id2entity: dict, save_dir: str, eval_dir: str, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        global TOTAL_COST
        # Initialize persona
        user_id = user.split('/')[2]
        print(f"Processing User ID: {user_id}")
        persona = data.get("persona", "")
        target_movies = data.get("target movies", [])
        seen_movies = data.get("seen movies", [])
        seeker = UserSimulator(
            args.utterance_prompt_path,
            args.reflection_prompt_path,
            args.api_key,
            args.user_model,
            args.user_temperature,
            args.max_tokens
        )
        # Initialize user info
        await seeker.set_user_info(data)
        # print(seeker.general_preferences)
        
        # For evaluation: save seen+target item list in eval folder
        for_eval = [{
            "seen": seen_movies,
            "target": target_movies
        }]
        
        # For save: record interaction
        dialogue_history = []   # for model
        dialogue_for_save = []  # for save
        
        ################################### Init #####################################
        recommender_init = "Hello, how can I help you?"
        dialogue_history.append({
            'role': "assistant",
            'content': recommender_init
        })

        ### Grouping seen, unseen, target### 
        seen = []
        unseen = []
        target = []
        candidate_list = []
        
        # turn 0
        save_data = [{
            "persona": persona,
            "seen movies": seen_movies,
            "target movies": target_movies,
        }, {
            "turn 0": recommender_init, 
        }]
        #######################################################################################
        
        reflection = ""
        for i in range(args.turn_num):
            turn = i + 1
            ###################################### 1. Generate Short-term Memory  ######################################
            ## Generate reflection ###
            f_cost = 0
            if i != 0:
                reflection, f_cost = await seeker.generate_reflection(seen, unseen, target, dialogue_history)
            TOTAL_COST+=f_cost
            seen.clear()
            unseen.clear()
            target.clear()
        
            ###################################### 2. User Interaction ######################################
            ### Generate User Utterance ###
            seeker_response,r_cost = await seeker.generate_utterance(dialogue_history, reflection)
            TOTAL_COST+=r_cost
        
            # For Model Input
            dialogue_history.append({
                "role": "user",
                'content': seeker_response,
            })
            
            ###################################### 2. Recommender Interaction ######################################
            # MACRS: get_conv returns (response, cost) tuple
            recommender_response, cost = await recommender.crs_model.get_conv(dialogue_history, candidate_list)
            TOTAL_COST+=cost
            
            # TODO: Add MACRS-specific logging or processing here
            # Could include multi-agent decision tracking, agent coordination logs, etc.
            
            # For Model Input
            dialogue_history.append({
                'role': 'assistant',
                'content': recommender_response,
            })
            
            ######################################  3. Update Item list ######################################
            # Retrieve Top-50 Recommendations, get_rec returns item ID
            all_recommendations, _, embedding_cost = await recommender.get_rec(dialogue_history) 
            TOTAL_COST+=embedding_cost
                
            # Convert the ID to String using id2entity
            all_recommendations_str = [f"{id2entity.get(rec_item, 'Untitled')}" for rec_item in all_recommendations]
                
            # Save retrieved top 50 movies for evaluation
            for_eval.append({
                "turn num": turn,
                "recommended": all_recommendations_str
            })  
        
            # Get 4 items for displaying in the item interface
            candidate_list = all_recommendations_str[:4]

            # Augment plot of each recommended movie
            candidate_list_with_description = {
                rec_item: {
                    "title": rec_item,
                    "plot": metadata.get(str(title_to_id.get(rec_item, 'Untitled')), {}).get("plot", "No plot information.")
                }
                for rec_item in candidate_list
            }

            ### Grouping seen, unseen, target and augment plot, reviews of 'seens' ### 
            seen.clear()
            unseen.clear()
            target.clear()
            for item in candidate_list:
                if item in target_movies:
                    target.append(candidate_list_with_description[item])
                elif item in seen_movies:
                    # Find the review for this seen movie
                    for movie in data.get("seen", []):  # Accessing review data
                        if movie.get("title") == item:
                            # Add the movie to the seen list with "title", "plot", and "Your Review"
                            seen.append({
                                "title": item,
                                "plot": metadata.get(str(title_to_id.get(item, '')), {}).get("plot", "No plot information."),
                                "Your Review": movie.get("abstract", "No review provided.")
                            })
                            break  # Exit once the review is found
                else:
                    unseen.append(candidate_list_with_description[item])
            
            ###################################### 4. Record ######################################
            turn_data = {
                f'turn {turn}': {
                    'Reflected_Preferences': reflection,
                    'Seeker': seeker_response,
                    'Recommender': recommender_response,
                    'top_k': candidate_list,
                    'match': {
                        'target': copy.deepcopy(target),
                        'seen': copy.deepcopy(seen),
                        'unseen': copy.deepcopy(unseen)
                    }
                    # TODO: Add MACRS-specific logging fields here
                    # Could include agent decisions, coordination logs, etc.
                }
            }
            # Append the turn data to dialogue_for_save
            dialogue_for_save.append(turn_data)
        
        # Add "dialogue_for_save" at below (like ChatGPT script)
        save_data.extend(dialogue_for_save)
        
        ###################################### save data ######################################
        # Save Dialogue (like ChatGPT script)
        await save_json_async(f'{save_dir}/{user_id}.json', save_data)
        
        # Save items for evaluation (like ChatGPT script)
        await save_json_async(f'{eval_dir}/{user_id}.json', for_eval)

        print(f"\tUser {user_id} conversation completed. Total cost: ${TOTAL_COST:.4f}")

async def main():
    """Main asynchronous function."""
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    global SETTING
    global RATIO
    warnings.filterwarnings("ignore")
    
    args = parse_args_macrs()
    
    # Directory for saving dialogues and recommendation list
    save_dir = os.path.join('experiments/5_2_main/results/dialogue_{}_{}_{}'.format(args.turn_num, args.core_num, SETTING, RATIO), args.crs_model, args.kg_dataset)
    eval_dir = os.path.join(save_dir, 'eval')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    random.seed(args.seed)
    
    # Recommender
    model_args = get_model_args(args.crs_model, args)
    recommender = RECOMMENDER(crs_model=args.crs_model, **model_args)
    
    # Load entity and metadata mappings asynchronously
    entity2id_path = os.path.join('crs_data', args.kg_dataset, 'entity2id.json')
    id2info_path = os.path.join('crs_data', args.kg_dataset, 'id2info.json')
    
    entity2id, metadata = await asyncio.gather(
        load_json_async(entity2id_path),
        load_json_async(id2info_path)
    )
    
    # Create mappings
    title_to_id = {
        info['name']: id for id, info in metadata.items() if info['name'] in entity2id
    }         
    id2entity = {int(v): k for k, v in entity2id.items()}
    
    # Load user data
    user_data_path = args.user_data_path
    user_data = await load_json_async(user_data_path)
    
    # Random sample for testing
    samples = args.num_samples
    random_selection = random.sample(list(user_data.items()), samples)
    user_data_all = dict(random_selection)
    
    user_data = {}
    # Users, already generated
    files_in_directory = os.listdir(save_dir)
    files_in_directory = [file[:-5] if file.endswith('.json') else file for file in files_in_directory]
    
    for user_id in user_data_all:
        user_id_short = user_id.split('/')[2]
        if user_id_short not in files_in_directory:
            user_data[user_id] = user_data_all[user_id]
    
    print(f'{len(user_data)} Users left to be generated')
    
    # Concurrency control
    max_concurrent_users = 100
    semaphore = asyncio.Semaphore(max_concurrent_users)
    
    # Process users concurrently  
    tasks = []
    for user, data in user_data.items():
        tasks.append(chat(user, data, args, recommender, metadata, title_to_id, id2entity, save_dir, eval_dir, semaphore))
    
    # Await all tasks
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    set_openai_api()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    finally:
        print(f"Total API Cost: ${TOTAL_COST:.6f} USD") 