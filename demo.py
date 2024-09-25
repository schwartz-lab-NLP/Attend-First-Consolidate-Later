import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralModel, MistralDecoderLayer
import argparse
from convert_models.convert_models import enable_manipulated_gen, enable_skip_attention
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        help='Name of huggingface model')
    parser.add_argument('--manipulation_type',
                        choices=[
                            'skip', 'freeze', 'switch_dict', 'random_switch',
                            'random_shuffle'
                        ],
                        required=True,
                        help='Manipulation Type to apply')
    parser.add_argument('--prompt',
                        required=True,
                        help='Prompt for model')
    parser.add_argument('--manipulated_layer',
                        default=20,
                        help='Layer to apply manipulation',
                        type=int)
    parser.add_argument('--huggingface_token', type=str, help="Huggingface authentication token")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 token=args.huggingface_token,
                                                 device_map="auto")

    input_ids = tokenizer(args.prompt,
                          return_tensors="pt").input_ids.to('cuda:0')


    if args.manipulation_type == "skip":
        enable_skip_attention(model)
        output = model(input_ids,
                       skip_attention_layers=range(args.manipulated_layer, 100),
                       )
    else:
        enable_manipulated_gen(model)
        output = model(input_ids,
                       distrupt_type=args.manipulation_type,
                       distrupt_layer=args.manipulated_layer,
                       distrupt_tokens=range(input_ids.shape[1] - 1),
                       )

    logits = torch.nn.Softmax(dim=0)(output.logits[0, -1, :])
    next_token = tokenizer.decode(torch.argmax(logits))
    print(f'{args.prompt}|{next_token}|')

