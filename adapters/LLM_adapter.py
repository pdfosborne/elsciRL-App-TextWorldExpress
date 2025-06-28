from torch import Tensor

import numpy as np
from gymnasium.spaces import Box

# Link to relevant ENCODER
from elsciRL.adapters.LLM_state_generators.text_ollama import OllamaAdapter


class Adapter:
    def __init__(self, setup_info:dict={}) -> None:   
        # Define observation space
        self.observation_space = Box(low=-1, high=1, shape=(1,384), dtype=np.float32)


        self.LLM_adapter = OllamaAdapter(
            model_name=setup_info.get('model_name', 'llama3.2'),
            base_prompt=setup_info.get('system_prompt', 'You are playing a Text Game.'),
            context_length=2000,
            action_history_length=setup_info.get('action_history_length', 5),
            encoder=setup_info.get('encoder', 'MiniLM_L6v2')
        )

        
    def adapter(self, state: str, legal_moves:list = None, episode_action_history:list = None, encode:bool=True, indexed: bool = False) -> Tensor:     
        """ Use Language description for every student for current grid position """

        if len(episode_action_history)>0:
            state = state + ' '+ episode_action_history[-1] + '.'

        # Use the elsciRL LLM adapter to transform and encode
        state_encoded = self.LLM_adapter.adapter(
            state=state, 
            legal_moves=legal_moves, 
            episode_action_history=episode_action_history, 
            encode=encode, 
            indexed=indexed
        )

        return state_encoded
    
    
            
            