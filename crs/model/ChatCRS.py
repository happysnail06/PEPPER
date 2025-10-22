import typing
import torch
import json
import os

import nltk
import openai
import tiktoken
import numpy as np

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import RetryCallState, _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm
from openai import AsyncOpenAI

# Import KG components from BARCOR
import sys
sys.path.append("..")
from crs.model.barcor.kg_bart import KGForBART

client = AsyncOpenAI()

crs_prompt_path = "prompts/crs_prompt.txt"
goal_planning_prompt_path = "prompts/chatcrs/goal_planning.txt"
chatcrs_prompt_path = "prompts/chatcrs/chatcrs_prompt.txt"

def get_prompt(prompt_path : str) -> str:
    if prompt_path:
        with open(prompt_path, 'r') as file:
            prompt = file.read()
    return prompt

async def annotate(conv_str):
    PRICE_PER_1K_INPUT = 0.0001
    
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    num_tokens = len(encoding.encode(conv_str))
    cost = (num_tokens / 1000) * PRICE_PER_1K_INPUT
    
    response = await client.embeddings.create(
        model='text-embedding-ada-002', input=conv_str
    )
    return response.data[0].embedding, cost

async def annotate_chat(messages, logit_bias=None):
    PRICE_PER_1K_INPUT = 0.0005 
    PRICE_PER_1K_OUTPUT = 0.0015
    if logit_bias is None:
        logit_bias = {}
    
    response = await client.chat.completions.create(
        model='gpt-4o-mini', messages=messages, temperature=0, logit_bias=logit_bias,
    )
    
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    
    TOTAL_COST = (prompt_tokens/1000) * PRICE_PER_1K_INPUT + (completion_tokens / 1000) * PRICE_PER_1K_OUTPUT
    
    return response.choices[0].message.content, TOTAL_COST

class ChatCRS():
    
    def __init__(self, seed, debug, kg_dataset) -> None:
        self.seed = seed
        self.debug = debug
        if self.seed is not None:
            set_seed(self.seed)
        
        self.kg_dataset = kg_dataset
        
        self.kg_dataset_path = f"crs_data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)
            
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        
        # Initialize KG for knowledge retrieval
        self.kg = KGForBART(kg_dataset=self.kg_dataset, debug=self.debug).get_kg_info()
        
        self.item_embedding_path = f"crs/model/chatgpt/{self.kg_dataset}"
        
        item_emb_list = []
        id2item_id = []
        files = os.listdir(self.item_embedding_path)
        for file in tqdm(files, desc="Loading item embeddings"):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
            
        self.chat_recommender_instruction = get_prompt(crs_prompt_path)
        self.goal_planning_prompt = get_prompt(goal_planning_prompt_path)
        self.chatcrs_prompt = get_prompt(chatcrs_prompt_path)
    
    async def get_rec(self, conv_dict):
        
        rec_labels = []
        
        context = conv_dict
        context_list = [] # for model
        
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'assistant'
            else:
                role_str = 'user'
            context_list.append({
                'role': role_str,
                'content': text
            })
        
        conv_str = ""
        
        for context in context_list[-2:]:
            conv_str += f"{context['role']}: {context['content']} "
            
        conv_embed, cost = await annotate(conv_str)
        conv_embed = np.asarray(conv_embed).reshape(1, -1)
        
        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
        
        
        return item_rank_arr[0], rec_labels, cost
    
    async def get_conv(self, conv_dict, candidate_list:list):
        
        # Step 1: Get dialogue goal using goal planning agent
        dialogue_goal, goal_cost = await self.goal_planning_agent(conv_dict)
        
        # Step 2: Get knowledge using knowledge retrieval agent
        knowledge = await self.knowledge_retrieval_agent(conv_dict)
        
        # Step 3: Format dialogue history for prompt
        formatted_history = ""
        for turn in conv_dict:
            role = turn.get('role', '')
            content = turn.get('content', '')
            formatted_history += f"{role}: {content}\n"
        
        # Step 4: Create ChatCRS prompt with all components
        context_list = [{
            'role': 'system',
            'content': self.chatcrs_prompt.format(
                dialogue_history=formatted_history.strip(),
                dialogue_goal=dialogue_goal,
                knowledge=knowledge
            )
        }]
        
        # Step 5: Generate response using ChatCRS framework
        gen_str, conv_cost = await annotate_chat(context_list)
        
        # Return response, total cost, and debug info (goal + knowledge)
        total_cost = goal_cost + conv_cost
        debug_info = {
            'dialogue_goal': dialogue_goal,
            'knowledge': knowledge
        }
        return gen_str, total_cost, debug_info

    async def goal_planning_agent(self, dialogue_history):
        """Goal Planning Agent: Predicts the next dialogue goal given conversation history"""
        
        # Format dialogue history for the prompt
        formatted_history = ""
        for turn in dialogue_history:
            role = turn.get('role', '')
            content = turn.get('content', '')
            formatted_history += f"{role}: {content}\n"
        
        # Create the goal planning prompt
        goal_planning_messages = [{
            'role': 'user',
            'content': self.goal_planning_prompt.format(dialogue_history=formatted_history.strip())
        }]
        
        # Get goal prediction from LLM
        predicted_goal, cost = await annotate_chat(goal_planning_messages)
        
        return predicted_goal.strip(), cost 

    async def knowledge_retrieval_agent(self, dialogue_history):
        """Knowledge Retrieval Agent: Extracts relevant entities from conversation history"""
        
        # Extract entities mentioned in conversation
        mentioned_entities = []
        
        for turn in dialogue_history:
            content = turn.get('content', '').lower()
            
            # Find entities mentioned in the conversation
            for entity_name, entity_id in self.entity2id.items():
                if entity_name.lower() in content:
                    entity_info = self.id2info.get(str(entity_id), {})
                    knowledge_triple = {
                        'entity': entity_name,
                        'entity_id': entity_id,
                        'type': entity_info.get('type', 'unknown'),
                        'info': entity_info
                    }
                    if knowledge_triple not in mentioned_entities:
                        mentioned_entities.append(knowledge_triple)
        
        # Format knowledge for prompt
        knowledge_text = ""
        for entity in mentioned_entities[:3]:  # Limit to top 3 entities
            entity_name = entity['entity']
            entity_type = entity['type']
            
            # Only show type if it's not unknown
            if entity_type == 'unknown':
                knowledge_text += f"- {entity_name}\n"
            else:
                knowledge_text += f"- {entity_name} (type: {entity_type})\n"
        
        return knowledge_text.strip() if knowledge_text else "No specific knowledge retrieved."
    