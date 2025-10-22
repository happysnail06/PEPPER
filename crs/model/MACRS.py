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

client = AsyncOpenAI()

# Use MACRS-specific prompt path
crs_prompt_path = "prompts/macrs/macrs_prompt.txt"

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

class MACRS():
    
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
        
        self.item_embedding_path = f"crs/model/chatgpt/{self.kg_dataset}"
        
        item_emb_list = []
        id2item_id = []
        files = os.listdir(self.item_embedding_path)
        for file in tqdm(files, desc="Loading item embeddings for MACRS"):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
            
        # Load all agent prompts
        self.asking_prompt = get_prompt("prompts/macrs/asking_agent.txt")
        self.chitchat_prompt = get_prompt("prompts/macrs/chitchat_agent.txt")
        self.recommending_prompt = get_prompt("prompts/macrs/recommending_agent.txt")
        self.planner_prompt = get_prompt("prompts/macrs/planner_agent.txt")
        self.info_reflection_prompt = get_prompt("prompts/macrs/info_reflection_agent.txt")
        self.strategy_reflection_prompt = get_prompt("prompts/macrs/strategy_reflection_agent.txt")
        
        # Initialize memory components
        self.dialogue_act_history = []
        self.user_profile = {"current_demand": {}, "browsing_history": []}
        self.strategy_suggestions = {"asking": "", "chitchat": "", "recommending": ""}
        self.corrective_experiences = ""
        self.conversation_trajectory = []
        self.last_recommendation_failure_turn = 0
    
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
        
        # Format dialogue history for agents
        dialogue_history = self._format_dialogue_history(conv_dict)
        
        # Current turn number for logging
        current_turn = len(conv_dict) // 2
        
        # Step 1: Information-level Reflection - Update user profile
        info_cost = 0
        if len(conv_dict) >= 2:  # If there's user feedback
            user_feedback = conv_dict[-1]['content'] if isinstance(conv_dict[-1], dict) else conv_dict[-1]
            system_response = conv_dict[-2]['content'] if isinstance(conv_dict[-2], dict) else conv_dict[-2]
            
            self.user_profile, info_cost = await self._information_reflection(
                dialogue_history, system_response, user_feedback
            )
            
        # Step 2: Strategy-level Reflection - Check for recommendation failure
        strategy_cost = 0
        failure_detected = self._is_recommendation_failure(conv_dict)
        if failure_detected:
            user_rejection = conv_dict[-1]['content'] if isinstance(conv_dict[-1], dict) else conv_dict[-1]
            strategy_cost = await self._strategy_reflection(current_turn, user_rejection)
        
        # Step 3: Generate candidate responses from all three responder agents
        asking_response, ask_cost = await self._generate_asking_response(dialogue_history)
        chitchat_response, chat_cost = await self._generate_chitchat_response(dialogue_history)
        recommending_response, rec_cost = await self._generate_recommending_response(dialogue_history)
        # Step 4: Use planner agent to select best response
        selected_act, final_response, plan_cost = await self._plan_response(
            dialogue_history, asking_response, chitchat_response, recommending_response
        )
        
        # Step 5: Update trajectory and dialogue act history
        last_user_input = conv_dict[-1]['content'] if conv_dict and isinstance(conv_dict[-1], dict) else (conv_dict[-1] if conv_dict else "")
        self._update_trajectory(selected_act, final_response, last_user_input)
        self.dialogue_act_history.append(selected_act)
        if len(self.dialogue_act_history) > 5:  # Keep only recent acts
            self.dialogue_act_history = self.dialogue_act_history[-5:]
        
        total_cost = info_cost + strategy_cost + ask_cost + chat_cost + rec_cost + plan_cost
        
        return final_response, total_cost
    
    def _format_dialogue_history(self, conv_dict):
        """Format conversation history for agent prompts"""
        history = ""
        for i, item in enumerate(conv_dict):
            # Handle dict format (with 'role' and 'content')
            if isinstance(item, dict):
                role = item.get('role', 'unknown')
                content = item.get('content', '')
                if len(content) == 0:
                    continue
                if role == 'assistant':
                    history += f"Assistant: {content}\n"
                elif role == 'user':
                    history += f"User: {content}\n"
            else:
                # Handle string format (fallback)
                if len(item) == 0:
                    continue
                if i % 2 == 0:
                    history += f"Assistant: {item}\n"
                else:
                    history += f"User: {item}\n"
        return history.strip()
    
    async def _information_reflection(self, dialogue_history, system_response, user_feedback):
        """Information-level reflection to update user profile"""
        previous_profile_str = self._format_user_profile_for_prompt()
        
        prompt = self.info_reflection_prompt.format(
            previous_user_profile=previous_profile_str,
            system_response=system_response,
            user_feedback=user_feedback,
            conversation_history=dialogue_history
        )
        
        messages = [{"role": "system", "content": prompt}]
        response, cost = await annotate_chat(messages)
        
        # Parse the response to update user profile structure
        updated_profile = self._parse_user_profile_response(response)
        return updated_profile, cost
    
    def _format_user_profile_for_prompt(self):
        """Format current user profile for prompt"""
        if not self.user_profile["current_demand"] and not self.user_profile["browsing_history"]:
            return "No previous preferences recorded."
        
        profile_str = "### Current User Demand\n"
        for key, value in self.user_profile["current_demand"].items():
            profile_str += f"- {key}: {value}\n"
        
        profile_str += "\n### Browsing History\n"
        for item in self.user_profile["browsing_history"]:
            profile_str += f"- {item}\n"
        
        return profile_str
    
    def _parse_user_profile_response(self, response):
        """Parse LLM response to extract structured user profile"""
        # Simple parsing - in production would use more robust parsing
        profile = {"current_demand": {}, "browsing_history": []}
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "Current User Demand" in line:
                current_section = "demand"
            elif "Browsing History" in line:
                current_section = "history"
            elif line.startswith('- ') and current_section:
                if current_section == "demand" and ':' in line:
                    key_value = line[2:].split(':', 1)
                    if len(key_value) == 2:
                        profile["current_demand"][key_value[0].strip()] = key_value[1].strip()
                elif current_section == "history":
                    profile["browsing_history"].append(line[2:].strip())
        
        return profile
    
    async def _generate_asking_response(self, dialogue_history):
        """Generate response from asking agent"""
        user_profile_str = self._format_user_profile_for_prompt()
        
        prompt = self.asking_prompt.format(
            dialogue_history=dialogue_history,
            user_profile=user_profile_str,
            strategy_suggestions=self.strategy_suggestions["asking"]
        )
        
        messages = [{"role": "system", "content": prompt}]
        response, cost = await annotate_chat(messages)
        return response, cost
    
    async def _generate_chitchat_response(self, dialogue_history):
        """Generate response from chit-chat agent"""
        user_profile_str = self._format_user_profile_for_prompt()
        
        prompt = self.chitchat_prompt.format(
            dialogue_history=dialogue_history,
            user_profile=user_profile_str,
            strategy_suggestions=self.strategy_suggestions["chitchat"]
        )
        
        messages = [{"role": "system", "content": prompt}]
        response, cost = await annotate_chat(messages)
        return response, cost
    
    async def _generate_recommending_response(self, dialogue_history):
        """Generate response from recommending agent"""
        user_profile_str = self._format_user_profile_for_prompt()
        
        prompt = self.recommending_prompt.format(
            dialogue_history=dialogue_history,
            user_profile=user_profile_str,
            strategy_suggestions=self.strategy_suggestions["recommending"]
        )
        
        messages = [{"role": "system", "content": prompt}]
        response, cost = await annotate_chat(messages)
        return response, cost
    
    async def _plan_response(self, dialogue_history, asking_response, chitchat_response, recommending_response):
        """Use planner agent to select best response"""
        recent_acts = ", ".join(self.dialogue_act_history[-3:]) if self.dialogue_act_history else "None"
        
        prompt = self.planner_prompt.format(
            dialogue_history=dialogue_history,
            dialogue_act_history=", ".join(self.dialogue_act_history),
            corrective_experiences=self.corrective_experiences,
            recent_acts=recent_acts,
            asking_response=asking_response,
            chitchat_response=chitchat_response,
            recommending_response=recommending_response
        )
        
        messages = [{"role": "system", "content": prompt}]
        plan_output, cost = await annotate_chat(messages)
        
        # Store planner reasoning for potential logging
        self.last_planner_reasoning = plan_output
        
        # Check for consecutive action limit BEFORE parsing planner decision
        def _check_consecutive_limit(act_to_check):
            if len(self.dialogue_act_history) >= 2:
                last_two = self.dialogue_act_history[-2:]
                if last_two[0] == act_to_check and last_two[1] == act_to_check:
                    return False  # Would be 3rd consecutive, not allowed
            return True
        
        # Parse planner decision with consecutive action enforcement
        plan_output_lower = plan_output.lower()
        
        # Try to extract the exact selected act first
        selected_act = None
        if "asking" in plan_output_lower:
            if _check_consecutive_limit("asking"):
                selected_act = "asking"
                final_response = asking_response
        elif "chitchat" in plan_output_lower:
            if _check_consecutive_limit("chitchat"):
                selected_act = "chitchat"
                final_response = chitchat_response
        elif "recommending" in plan_output_lower:
            if _check_consecutive_limit("recommending"):
                selected_act = "recommending"
                final_response = recommending_response
        
        # If no valid selection or consecutive limit hit, use fallback
        if selected_act is None:
            # Smart fallback that respects consecutive limits
            available_actions = []
            if _check_consecutive_limit("asking"):
                available_actions.append(("asking", asking_response))
            if _check_consecutive_limit("chitchat"):
                available_actions.append(("chitchat", chitchat_response))
            if _check_consecutive_limit("recommending"):
                available_actions.append(("recommending", recommending_response))
            
            if not available_actions:
                # Edge case - all actions blocked, force different action
                last_act = self.dialogue_act_history[-1] if self.dialogue_act_history else None
                if last_act != "asking":
                    selected_act = "asking"
                    final_response = asking_response
                elif last_act != "chitchat":
                    selected_act = "chitchat"
                    final_response = chitchat_response
                else:
                    selected_act = "recommending"
                    final_response = recommending_response
            else:
                # Enhanced fallback logic with priority: recommending > chitchat > asking
                if not self.dialogue_act_history:
                    # First turn - start with asking to gather preferences
                    selected_act = "asking"
                    final_response = asking_response
                else:
                    # Fallback priority order: recommending -> chitchat -> asking
                    fallback_priority = [
                        ("recommending", recommending_response),
                        ("chitchat", chitchat_response), 
                        ("asking", asking_response)
                    ]
                    
                    # Find first available action in priority order
                    for priority_act, priority_response in fallback_priority:
                        if (priority_act, priority_response) in available_actions:
                            selected_act = priority_act
                            final_response = priority_response
                            break
                    
                    # If no action found in priority order, use any available (shouldn't happen)
                    if selected_act is None and available_actions:
                        selected_act, final_response = available_actions[0]
            
            # Fallback selection completed
        
        return selected_act, final_response, cost
    
    def _format_candidate_list(self, candidate_list):
        """Format candidate list for agent prompts"""
        if not candidate_list:
            return "No candidate movies available."
        
        formatted_items = []
        for i, item_id in enumerate(candidate_list[:10]):  # Limit to top 10
            if str(item_id) in self.id2info:
                item_info = self.id2info[str(item_id)]
                formatted_items.append(f"{i+1}. {item_info['name']}")
            else:
                formatted_items.append(f"{i+1}. Movie ID {item_id}")
        
        return "\n".join(formatted_items)
    
    async def _strategy_reflection(self, current_turn, user_rejection):
        """Strategy-level reflection for recommendation failures"""
        # Get trajectory from last failure to current turn
        start_turn = self.last_recommendation_failure_turn
        trajectory = self._format_trajectory(start_turn, current_turn)
        
        prompt = self.strategy_reflection_prompt.format(
            trajectory=trajectory,
            failed_turn=current_turn,
            user_rejection=user_rejection
        )
        
        messages = [{"role": "system", "content": prompt}]
        response, cost = await annotate_chat(messages)
        
        # Parse strategy suggestions and corrective experiences
        self._parse_strategy_reflection_response(response)
        self.last_recommendation_failure_turn = current_turn
        
        return cost
    
    def _is_recommendation_failure(self, conv_dict):
        """Detect if user rejected recommendation"""
        if len(conv_dict) < 2:
            return False
        
        # Extract content from dict format
        user_feedback = conv_dict[-1]['content'].lower() if isinstance(conv_dict[-1], dict) else conv_dict[-1].lower()
        system_response = conv_dict[-2]['content'].lower() if isinstance(conv_dict[-2], dict) else conv_dict[-2].lower()
        
        # Check if system made recommendation and user rejected it
        rec_indicators = ["recommend", "suggest", "try", "watch", "check out"]
        rejection_indicators = ["no", "not interested", "don't like", "not really", "something else", "different"]
        
        system_made_rec = any(indicator in system_response for indicator in rec_indicators)
        user_rejected = any(indicator in user_feedback for indicator in rejection_indicators)
        
        return system_made_rec and user_rejected
    
    def _format_trajectory(self, start_turn, end_turn):
        """Format multi-turn trajectory for strategy reflection"""
        if not self.conversation_trajectory:
            return "No previous trajectory available."
        
        trajectory_str = ""
        for i, turn in enumerate(self.conversation_trajectory[start_turn:end_turn+1]):
            turn_num = start_turn + i + 1
            trajectory_str += f"Turn {turn_num}:\n"
            trajectory_str += f"  User Profile: {turn.get('user_profile', 'N/A')}\n"
            trajectory_str += f"  System Response: {turn.get('system_response', 'N/A')}\n"
            trajectory_str += f"  User Feedback: {turn.get('user_feedback', 'N/A')}\n"
            trajectory_str += f"  Selected Act: {turn.get('selected_act', 'N/A')}\n\n"
        
        return trajectory_str
    
    def _update_trajectory(self, selected_act, system_response, user_feedback):
        """Update conversation trajectory for strategy reflection"""
        turn_info = {
            "user_profile": self._format_user_profile_for_prompt(),
            "system_response": system_response,
            "user_feedback": user_feedback,
            "selected_act": selected_act
        }
        self.conversation_trajectory.append(turn_info)
        
        # Keep only recent trajectory (last 10 turns)
        if len(self.conversation_trajectory) > 10:
            self.conversation_trajectory = self.conversation_trajectory[-10:]
    
    def _parse_strategy_reflection_response(self, response):
        """Parse strategy reflection response to extract suggestions and experiences"""
        lines = response.split('\n')
        current_section = None
        
        # Reset suggestions
        self.strategy_suggestions = {"asking": "", "chitchat": "", "recommending": ""}
        experiences = []
        
        for line in lines:
            line = line.strip()
            
            # Handle new format: **Recommending:** [brief tip]
            if line.startswith("**Recommending:**"):
                self.strategy_suggestions["recommending"] = line.replace("**Recommending:**", "").strip()
            elif line.startswith("**Asking:**"):
                self.strategy_suggestions["asking"] = line.replace("**Asking:**", "").strip()
            elif line.startswith("**Chitchat:**"):
                self.strategy_suggestions["chitchat"] = line.replace("**Chitchat:**", "").strip()
            elif "### Experiences" in line:
                current_section = "experiences"
            elif line.startswith('- ') and current_section == "experiences":
                experiences.append(line[2:].strip())
            
            # Handle old format for backward compatibility
            elif "Recommending Agent Suggestions:" in line:
                current_section = "recommending"
            elif "Asking Agent Suggestions:" in line:
                current_section = "asking"
            elif "Chit-chatting Agent Suggestions:" in line:
                current_section = "chitchat"
            elif "Corrective Experiences for Planning Agent" in line:
                current_section = "experiences"
            elif line.startswith('- ') and current_section in self.strategy_suggestions:
                suggestion = line[2:].strip()
                if self.strategy_suggestions[current_section]:
                    self.strategy_suggestions[current_section] += " " + suggestion
                else:
                    self.strategy_suggestions[current_section] = suggestion
        
        # Combine experiences
        self.corrective_experiences = " ".join(experiences) 