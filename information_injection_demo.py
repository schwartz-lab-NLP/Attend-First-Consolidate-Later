from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from convert_models.convert_models import enable_manipulated_gen
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        help='Name of huggingface model')
    parser.add_argument('--manipulation_layer',
                        default=10,
                        help='Layer to apply manipulation',
                        type=int)
    parser.add_argument(
        '--hf_token',
        type=str,
        required=True,
        help="huggingface authentication token"
    )
    args = parser.parse_args()
    print(f'{args=}')
    return args


def get_injection_data(text):
    input_ids = tokenizer(text, return_tensors="pt")['input_ids'].to('cuda:0')
    hidden = model(input_ids, output_hidden_states=True).hidden_states
    hidden = hidden[args.manipulation_layer]
    return hidden[0], input_ids


def print_ids(ids):
    ids = list(ids[0])
    d = {idx: tokenizer.decode(token) for idx, token in enumerate(ids)}
    print(d)


def get_next_token(logits):
    soft_logits = torch.nn.Softmax(dim=0)(logits[0, -1, :])
    next_token = tokenizer.decode(torch.argmax(soft_logits))
    return next_token


if __name__ == "__main__":

    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 token=args.hf_token,
                                                 device_map="auto")

    original_prompt = f'Q: What is the capital of Spain? \n A: Madrid \n Q: What is the capital of France? \n A:'
    input_ids = tokenizer(original_prompt, return_tensors="pt")['input_ids']
    france_idx = 24

    injection_prompt = f'The country Italy'
    injection_hidden, injection_ids = get_injection_data(injection_prompt)
    italy_idx = 2 if "Yi" in args.model else 3
    italy_hidden = injection_hidden[italy_idx]

    print('ORIGINAL TOKEN INDEXES')
    print_ids(input_ids)  # Index of France: 24
    print('OTHER PROMPT INDEXES')
    print_ids(injection_ids)  # Index of Italy: 3 (2 in Yi)

    # Injection dict determines what get's injected.
    # It's in the format {injection_index: injection_tensor}
    # i.e. during the generation, the hidden state at injdection_index in manipulaiton_layer
    # get's replaced with the tensor injection_tensor.
    injection_dict = {france_idx: italy_hidden}
    original_output = model(input_ids, output_hidden_states=True)
    original_next_token = get_next_token(original_output.logits)

    enable_manipulated_gen(model)

    manipulated_output = model(input_ids,
                               manipulation_type="information_injection",
                               manipulation_layer=args.manipulation_layer,
                               injection_dict=injection_dict,
                               output_hidden_states=True)
    manipulated_next_token = get_next_token(manipulated_output.logits)

    print(f'{original_prompt=}, {injection_prompt=}')
    print(f'{original_next_token=}, {manipulated_next_token=}')
