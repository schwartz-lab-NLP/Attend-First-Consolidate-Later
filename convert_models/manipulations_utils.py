import torch


class ManipulatorClass:

    def __init__(self,
                 manipulation_layer: int,
                 manipulation_tokens: list,
                 manipulation_type: str,
                 injection_dict: dict = None,
                 save_manipulation: bool = False):
        """_summary_

        Args:
            manipulation_layer (int): Layer (int) to apply manipulation
            manipulation_tokens (list): List of indexes of tokens to manipulate
            manipulation_type (str): Type of manipulation to apply, from ['freeze', 'information_injection', 'random', 'shuffle']
            injection_dict (dict, optional): Dictionary of injection tensors, from the format {injection_token_idx: injection_tensor}.
            save_manipulation (bool, optional): To save switched representations.

        More about injections: The injection dict is from the format {injection_token_idx: injection_tensor}. 
        Meaning, at layer manipulation_layer, the hidden representation in index injection_token_idx,
        will be replaced with injection_tensor.
        """

        self.manipulation_layer = manipulation_layer
        self.manipulation_tokens = manipulation_tokens
        self.manipulation_type = manipulation_type
        self.injection_dict = injection_dict
        self.save_manipulation = save_manipulation

        self.freeze_dict = {} # Saves frozen representation from previous layers
        self.manipulation_dict = {} # Saves the applied manipulation

        modes_list = ['freeze', 'random', 'shuffle', 'information_injection']

        if manipulation_type not in modes_list:
            print(
                f"manipulation_type unrecognized, is {manipulation_type}. Should be in {modes_list}",
                flush=True)
            exit(-1)
        if manipulation_type == 'information_injection' and (
                injection_dict is None or type(injection_dict) != dict):
            print(
                f"To use information_injection, you must provide a injection_dict. You have provided {injection_dict=}, {type(injection_dict)=}",
                flush=True)
            exit(-1)

    def apply_distruption(self, idx, hidden_states):
        if self.manipulation_type == 'freeze':
            return self.apply_freeze(idx=idx, hidden_states=hidden_states)
        if idx == self.manipulation_layer:
            if self.manipulation_type == 'information_injection':
                return self.apply_injection(hidden_states)
            if self.manipulation_type == 'random':
                return self.apply_random_noise(hidden_states)
            if self.manipulation_type == 'shuffle':
                return self.apply_shuffle_random(idx=idx,
                                                 hidden_states=hidden_states)
        else:
            return hidden_states

    def apply_freeze(self, idx, hidden_states):
        if idx == self.manipulation_layer - 1:
            for token_idx in self.manipulation_tokens:
                if token_idx < hidden_states.shape[1]:
                    self.freeze_dict[token_idx] = hidden_states[0][
                        token_idx].clone()
            return hidden_states
        elif idx >= self.manipulation_layer:
            for token_idx in self.manipulation_tokens:
                if token_idx < hidden_states.shape[1]:

                    hidden_states[0][token_idx] = self.freeze_dict[token_idx]
            return hidden_states
        return hidden_states

    def apply_injection(self, hidden_states):
        for token_idx, v in self.injection_dict.items():
            v = v.to(hidden_states.device)
            self.manipulation_saver(token_idx=token_idx,
                                    og_vector=hidden_states[0][token_idx],
                                    switch_vector=v)
            hidden_states[0][token_idx] = v
        return hidden_states

    def apply_random_noise(self, hidden_states):
        for token_idx in self.manipulation_tokens:
            v = torch.randn_like(hidden_states[0][token_idx])
            # Normalize random vector to current hidden representation:
            v = (v / torch.norm(v, 2)) * torch.norm(
                hidden_states[0][token_idx], 2)
            v = v.to(hidden_states.device)
            self.manipulation_saver(token_idx=token_idx,
                                    og_vector=hidden_states[0][token_idx],
                                    switch_vector=v)
            hidden_states[0][token_idx] = v
        return hidden_states

    def apply_shuffle_random(self, idx, hidden_states):
        for token_idx in self.manipulation_tokens:
            torch.manual_seed(idx * token_idx)
            indexes = torch.randperm(hidden_states[0, token_idx, :].shape[-1])
            shuffled = hidden_states[0][token_idx][indexes].to(
                hidden_states.device)
            self.manipulation_saver(token_idx=token_idx,
                                    og_vector=hidden_states[0][token_idx],
                                    switch_vector=shuffled)
            hidden_states[0][token_idx] = shuffled
        return hidden_states

    def manipulation_saver(self, token_idx, og_vector, switch_vector):
        if self.save_manipulation:
            self.manipulation_dict[token_idx] = {
                'og_vector': og_vector.clone().detach().cpu().numpy(),
                'switch_vector': switch_vector.clone().detach().cpu().numpy()
            }

    def get_manipulation_dict(self):
        return self.manipulation_dict
