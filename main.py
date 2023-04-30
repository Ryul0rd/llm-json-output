from typing import List, Callable, Optional
from dataclasses import dataclass, fields
import torch
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2LMHeadModel, LlamaTokenizer, LlamaForCausalLM
import time
import cProfile

from json_constraint import JsonConstraint

PrefixAllowedTokensFn = Callable[[int, torch.Tensor], List[int]]


def main():
    input_text = [
        "Plain Text:\nClara is a 29 year old woman. She likes cooking and hiking.\nJSON:\n",
        #"Plain Text:\nMax is a 24 year old guy. His hobbies include gaming and martial arts.\nJSON:\n",
        "Plain Text:\nMax is a guy whos age is unknown. His hobbies include gaming and martial arts.\nJSON:\n",
    ]

    gpt2_test(input_text)
    llama_test(input_text)


@dataclass
class Person:
    name: str
    age: Optional[int]
    #number_of_pets: int
    #number_of_hobbies: int
    is_male: bool
    email_address: Optional[str]
    #hobbies: List[str]



def gpt2_test(input_text: List[str]):
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", padding_side="left")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.batch_encode_plus(
        input_text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).input_ids.cuda()

    json_constraint = JsonConstraint(input_ids, tokenizer, schemas=Person)

    start_time = time.time()
    output_tokens = model.generate(input_ids, max_new_tokens=64, prefix_allowed_tokens_fn=json_constraint)
    end_time = time.time()
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    print()
    for sample in output_text:
        print(f"\n{sample}")
    print(f"\nTime taken: {end_time - start_time}")


def llama_test(input_text: List[str]):
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", pad_token="</s>", padding_side="left")
    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()

    input_ids = tokenizer.batch_encode_plus(
        input_text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).input_ids.cuda()

    json_constraint = JsonConstraint(input_ids, tokenizer, schemas=Person)

    start_time = time.time()
    output_tokens = model.generate(input_ids, max_new_tokens=64, eos_token_id=0, prefix_allowed_tokens_fn=json_constraint)
    end_time = time.time()
    output_text = tokenizer.batch_decode(torch.where(output_tokens == -1, 1, output_tokens), skip_special_tokens=True)

    print()
    for sample in output_text:
        print(f"\n{sample}")
    print(f"\nTime taken: {end_time - start_time}")


if __name__ == "__main__":
    main()
    #cProfile.run("main()", filename="profile_data")
