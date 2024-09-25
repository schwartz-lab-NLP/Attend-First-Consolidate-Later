import types

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralModel, MistralDecoderLayer

from .llama_skip import llama_skip_decoder_forward, llama_skip_model_forward, llama_skip_casualLM_forward
from .mistral_skip import mistral_skip_model_forward, mistral_skip_decoder_forward, mistral_skip_causalLM_forward

from .llama_manipulate import llama_manipulate_model_forward, llama_manipulate_casualLM_forward
from .mistral_manipulated import mistral_model_manipulated_forward, mistral_model_causalLM_manipulated_forward

from functools import wraps
import inspect


def add_argument(func):

    @wraps(func)
    def wrapper(*args,
                distrupt_type,
                distrupt_layer,
                distrupt_tokens,
                **kwargs):
        return func(*args, **kwargs)

    original_sig = inspect.signature(func)

    a_param = inspect.Parameter('distrupt_type',
                                inspect.Parameter.VAR_KEYWORD,)
    b_param = inspect.Parameter('distrupt_layer',
                                inspect.Parameter.VAR_KEYWORD,)
    c_param = inspect.Parameter('distrupt_tokens', inspect.Parameter.VAR_KEYWORD)
    new_params = list(original_sig.parameters.values()) + [a_param, b_param, c_param]

    new_sig = original_sig.replace(parameters=new_params)

    wrapper.__signature__ = new_sig
    return wrapper

def enable_manipulated_gen(model):
    if isinstance(model, LlamaForCausalLM):
        model.forward = types.MethodType(llama_manipulate_casualLM_forward,
                                         model)

        model.prepare_inputs_for_generation = (add_argument(
            model.prepare_inputs_for_generation), model)
        enable_manipulated_gen(model.model)
        return None

    if isinstance(model, MistralForCausalLM):
        model.forward = types.MethodType(
            mistral_model_causalLM_manipulated_forward, model)
        model.prepare_inputs_for_generation = (add_argument(model.prepare_inputs_for_generation), model)
        enable_manipulated_gen(model.model)
        return None
    if isinstance(model, LlamaModel):
        model.forward = types.MethodType(llama_manipulate_model_forward, model)

    if isinstance(model, MistralModel):
        model.forward = types.MethodType(mistral_model_manipulated_forward,
                                         model)


def enable_skip_attention(model):
    if isinstance(model, MistralForCausalLM):
        model.forward = types.MethodType(mistral_skip_causalLM_forward, model)
        enable_skip_attention(model.model)
        return None

    if isinstance(model, LlamaForCausalLM):
        model.forward = types.MethodType(llama_skip_casualLM_forward, model)
        enable_skip_attention(model.model)
        return None

    if isinstance(model, LlamaModel):
        model.forward = types.MethodType(llama_skip_model_forward, model)

    if isinstance(model, MistralModel):
        model.forward = types.MethodType(mistral_skip_model_forward, model)

    for idx, layer in enumerate(model.layers):
        if isinstance(layer, LlamaDecoderLayer):
            model.layers[idx].forward = types.MethodType(
                llama_skip_decoder_forward, model.layers[idx])

        if isinstance(layer, MistralDecoderLayer):
            model.layers[idx].forward = types.MethodType(
                mistral_skip_decoder_forward, model.layers[idx])
