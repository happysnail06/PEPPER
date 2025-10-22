import sys
import tiktoken
from typing import List, Dict
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

sys.path.append('../')
from utils import get_prompt

class UserSimulator:
    """
    User simulator class

    Main attributes:
        Reflected Preference Generation
        Utterance Generation
    """

    def __init__(
        self,
        utterance_prompt_path: str,
        reflection_prompt_path: str,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 512,
    ) -> None:
        """
        Args:
            api_key: OpenAI API key.
            utterance_prompt_path: Path to the utterance prompt
            reflection_prompt_path: Path to reflected preference generation prompt
            model_name: Base model. Default gpt-3.5-turbo
            temperature: Default 0
            max_tokens: Default 256.
        """

        # User attributes, controlled in set_user_persona member function
        self.general_preferences: str = ''
        self.target_movies: List[str] = []
        self.seen_movies: List[str] = []
        self.encoding = tiktoken.encoding_for_model(model_name)

        # Initialize LangChain's ChatOpenAI with asynchronous support
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=20,
        )

        # Get prompts
        utterance_gen_prompt = get_prompt(utterance_prompt_path)
        reflection_gen_prompt = get_prompt(reflection_prompt_path)

        # Intialize prompts
        self.utterance_prompt = HumanMessagePromptTemplate.from_template(utterance_gen_prompt)
        self.reflection_prompt = HumanMessagePromptTemplate.from_template(reflection_gen_prompt)
        
        # Initialize LLMChains
        # Utterance
        self.utterance_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([self.utterance_prompt]),
        )

        # Reflection
        self.reflection_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([self.reflection_prompt]),
        )

    async def set_user_info(self, user_data: Dict) -> None:
        """
        Set general preferences, seen, target movies
        """
        self.general_preferences = user_data.get("persona", "")
        self.seen_movies = user_data.get("seen movies", [])
        self.target_movies = user_data.get("target movies", "")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_price = 0.0005 / 1000
        output_price = 0.0015 / 1000
        return (input_tokens * input_price) + (output_tokens * output_price)

    def _format_conversation_history(self, conv_dict: List[Dict[str, str]]) -> str:
        """
        Format the conversation history into a string for the prompt.

        Args:
            conv_dict: List of conversation turns. {"user" : text, "assistant" : text}
        """
        return "\n".join([f"{utterance['role'].capitalize()}: {utterance['content']}" for utterance in conv_dict])

    async def generate_utterance(self, conv_dict: List[Dict[str, str]], reflection: str) -> str:
        """
        Generate next utterance based on the conversation history and "reflection".

        Args:
            conv_dict: List of conversation turns.
            "reflection": "Reflected Preferences"
        """
        # Format conversation history
        formatted_conv = self._format_conversation_history(conv_dict)

        # Prompt Input
        prompt_input = {
            "general_preferences": self.general_preferences,
            "opinion": reflection,
            "conversation_history": formatted_conv
        }
        
        prompt_text = self.utterance_prompt.format(**prompt_input)
        prompt_text = prompt_text.content
        input_tokens = self.count_tokens(prompt_text)

        try:
            # Generate user utterance
            response = await self.utterance_chain.arun(prompt_input)
            output_tokens = self.count_tokens(response)
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            return response.strip(), cost
        except Exception as e:
            print(f"Error during utterance generation: {e}")
            return "Something went wrong"

    
    async def generate_reflection(self, seen: List[Dict[str, str]], unseen: List[Dict[str, str]], target: List[Dict[str, str]], dialogue_history) -> str:
        """
        Generates "reflection" based on seen, unseen, and target items.
        """
        
        formatted_conv = self._format_conversation_history(dialogue_history)
        
        # Concatenate title and plot
        seen_ = ", ".join([f"{item['title']}: {item['plot']}" for item in seen])
        unseen_ = ", ".join([f"{item['title']}: {item['plot']}" for item in unseen])
        target_ = ", ".join([f"{item['title']}: {item['plot']}" for item in target])

        # Prepare input for the reflection chain
        prompt_input = {
            "general_preferences": self.general_preferences,
            "seen": seen_,
            "unseen": unseen_ + target_,
            "conversation_history": formatted_conv
        }
        
        prompt_text = self.reflection_prompt.format(**prompt_input)
        prompt_text = prompt_text.content
        input_tokens = self.count_tokens(prompt_text)

        try:
            # Async generate reflection
            reflection = await self.reflection_chain.arun(prompt_input)
            output_tokens = self.count_tokens(reflection)
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            return reflection.strip(), cost
        except Exception as e:
            print(f"Error during reflection generation: {e}")
            return "Something went wrong"
