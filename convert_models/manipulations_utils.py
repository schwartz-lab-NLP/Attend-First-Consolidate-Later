import math
import numpy as np
import torch


def get_random_orthogonal(u):
    """
    Returns a random vector orthogonal to u.
    """
    v = torch.randn_like(u).to(u.device)
    v = v - (torch.dot(u, v) / torch.linalg.norm(u)**2) * u
    v = (v / torch.linalg.norm(v)) * torch.linalg.norm(u)
    return v


def get_random_vector_at_angle(u, angle):
    """
    angle in degrees!
    Returns a vector v with the same norm as u, such that angle(u,v)=angle
    """
    radians_angle = math.radians(angle)
    rand_o = get_random_orthogonal(u).to(u.device)
    v = math.cos(radians_angle) * u + math.sin(radians_angle) * rand_o
    return v


class DistruptClass:

    def __init__(self,
                 distrupt_layer: int,
                 distrupt_tokens: list,
                 distrupt_type: str,
                 angle=None,
                 switch_dict: dict = None,
                 save_switched: bool = False):
        self.distrupt_layer = distrupt_layer
        self.distrupt_tokens = distrupt_tokens
        self.distrupt_type = distrupt_type
        self.angle = angle
        self.user_switch_dict = switch_dict
        self.freeze_dict = {}
        self.switch_dict = {}
        self.save_switched = save_switched
        self.noise_norm = 0

        # print(f'GOTTEN DISTRUPT TYPE {self.distrupt_type}')

        modes_list = [
            'freeze', 'switch_dict', 'random_switch', 'angle_switch',
            'random_shuffle'
        ]

        if distrupt_type == 'switch_dict':
            print(f'{self.user_switch_dict.keys()=}')

        if distrupt_type not in modes_list:
            print(
                f"distrupt_type unrecognized, is {distrupt_type}. Should be in {modes_list}",
                flush=True)
            exit(-1)
        if distrupt_type == 'switch_dict' and (switch_dict is None
                                               or type(switch_dict) != dict):
            print(
                f"To use switch_dict, you must provide a switch_dict. You have provided {switch_dict=}, {type(switch_dict)=}",
                flush=True)
            exit(-1)
        if distrupt_type == 'angle_switch' and (angle is None):
            print(f'to use angle_switch, you must provide angle.', flush=True)
            exit(-1)

    def apply_distruption(self, idx, hidden_states):
        if self.distrupt_type == 'freeze':
            return self.apply_freeze(idx=idx, hidden_states=hidden_states)
        if idx == self.distrupt_layer:
            if self.distrupt_type == 'switch_dict':
                # print(f'APPLYING DICT SWICH, {idx=}')
                return self.apply_dict_switch(hidden_states)
            if self.distrupt_type == 'random_switch':
                # print(f'APPLYING RANDOM SWITCH')
                return self.apply_random_switch(hidden_states)
            if self.distrupt_type == 'angle_switch':
                return self.apply_angle_switch(hidden_states)
            if self.distrupt_type == 'random_shuffle':
                return self.apply_shuffle_random(idx=idx,
                                                 hidden_states=hidden_states)
        else:
            return hidden_states

    def apply_freeze(self, idx, hidden_states):
        if idx == self.distrupt_layer - 1:
            # print(f'APPLY FREEZE: SAVING HIDDEN STATE, {idx=}')
            for token_idx in self.distrupt_tokens:
                if token_idx < hidden_states.shape[1]:
                    self.freeze_dict[token_idx] = hidden_states[0][
                        token_idx].clone()
            return hidden_states
        elif idx >= self.distrupt_layer:
            # print(f'APPLY FREEZE: USE SAVED HIDDEN STATE, {idx=}')
            for token_idx in self.distrupt_tokens:
                if token_idx < hidden_states.shape[1]:

                    hidden_states[0][token_idx] = self.freeze_dict[token_idx]
            return hidden_states
        return hidden_states

    def apply_dict_switch(self, hidden_states):
        # print(f'INSIDE SWITCH_DICT1, {self.user_switch_dict.keys()=}')
        for token_idx, v in self.user_switch_dict.items():
            v = v.to(hidden_states.device)
            self.switch_saver(token_idx=token_idx,
                              og_vector=hidden_states[0][token_idx],
                              switch_vector=v)
            temp = hidden_states[0][token_idx].clone()
            hidden_states[0][token_idx] = v
            temp2 = temp - hidden_states[0][token_idx]
            # print(f'INSIDE SWITCH_DICT, {temp2.mean()=}')
            del temp
            del temp2
        return hidden_states

    def apply_random_switch(self, hidden_states):
        for token_idx in self.distrupt_tokens:
            v = torch.randn_like(hidden_states[0][token_idx])
            v = (v / torch.norm(v, 2)) * torch.norm(
                hidden_states[0][token_idx], 2)
            v = v.to(hidden_states.device)
            self.switch_saver(token_idx=token_idx,
                              og_vector=hidden_states[0][token_idx],
                              switch_vector=v)
            temp = hidden_states[0][token_idx].clone()
            hidden_states[0][token_idx] = v
            # print(f'AFTER RANDOM SWITCH: {(temp-hidden_states[0][token_idx]).max()=}')
        return hidden_states

    def apply_angle_switch(self, hidden_states):
        for token_idx in self.distrupt_tokens:
            v = get_random_vector_at_angle(hidden_states[0][token_idx],
                                           self.angle)
            v = v.to(hidden_states.device)
            self.switch_saver(token_idx=token_idx,
                              og_vector=hidden_states[0][token_idx],
                              switch_vector=v)
            hidden_states[0][token_idx] = v
        return hidden_states

    def apply_random_noise(self, hidden_states):
        for token_idx in self.distrupt_tokens:
            noise = torch.randn_like(hidden_states[0][token_idx])
            noise = (noise / torch.linalg.vector_norm(noise)) * self.noise_norm
            noise.to(hidden_states.device)
            hidden_states[0][token_idx] += noise

    def apply_shuffle_random(self, idx, hidden_states):
        for token_idx in self.distrupt_tokens:
            torch.manual_seed(idx * token_idx)
            indexes = torch.randperm(hidden_states[0, token_idx, :].shape[-1])
            shuffled = hidden_states[0][token_idx][indexes].to(
                hidden_states.device)
            self.switch_saver(token_idx=token_idx,
                              og_vector=hidden_states[0][token_idx],
                              switch_vector=shuffled)
            hidden_states[0][token_idx] = shuffled
        return hidden_states

    def switch_saver(self, token_idx, og_vector, switch_vector):
        if self.save_switched:
            self.switch_dict[token_idx] = {
                'og_vector': og_vector.clone().detach().cpu().numpy(),
                'switch_vector': switch_vector.clone().detach().cpu().numpy()
            }

    def get_switch_dict(self):
        return self.switch_dict
