from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from elsciRL.encoders.poss_state_encoded import StateEncoder

class Adapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, setup_info:dict={}) -> None:
        # TODO: Update this based on the current problem, each requires preset knowledge of all possible states/actions/objects
        # - Possible States
        # - Possible Actions
        # - Prior Actions
        # - Possible Objects
    
        # Initialise encoder based on all possible env states
        all_possible_states = [i for i in range(4*4)]
        self.encoder = StateEncoder(all_possible_states)
    
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
       
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in Adapter._cached_state_idx):
                    Adapter._cached_state_idx[sent] = len(Adapter._cached_state_idx)
                state_indexed.append(Adapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded
