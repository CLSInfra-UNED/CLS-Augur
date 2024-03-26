"""
Parts of this code use resources from Meta and as such is governed by the orginial META license.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
import os

import torch

from functools import lru_cache
from threading import Thread
from typing import Iterator

from transformers import (AutoConfig, 
                          AutoModelForCausalLM, 
                          AutoTokenizer, 
                          TextIteratorStreamer, 
                          BitsAndBytesConfig, 
                          TextStreamer,
                          Conversation, 
                          pipeline,
                          LlamaForCausalLM,
                          GenerationMixin)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomGenerationMixin(GenerationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def generate(self, *args, **kwargs):
        return super().generate(*args, **kwargs)

class AugurLlamaModel(LlamaForCausalLM, CustomGenerationMixin):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)


def get_llama_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        model,
        tokenizer,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_llama_prompt(message, chat_history, system_prompt)
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda:0')

    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=10.,
        skip_prompt=True,
        skip_special_tokens=True    
    )
    
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    
    print(model.device)
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)


def _get_tokenizer_with_system_prompt(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if 'CodeLlama' in model_id:
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate "
            "user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + "
            "message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            )
    
    if 'deepseek' in model_id:
        tokenizer.chat_template = (
            "{% if not add_generation_prompt is defined %}\n"
            "{% set add_generation_prompt = false %}\n"
            "{% endif %}\n{%- set ns = namespace(found=false) -%}\n"
            "{%- for message in messages -%}\n"
            "{%- if message['role'] == 'system' -%}\n"
            "{%- set ns.found = true -%}\n"
            "{%- endif -%}\n{%- endfor -%}\n{{bos_token}}"
            "{%- if not ns.found -%}\n"
            "{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n"
            "{%- endif %}\n{%- for message in messages %}\n"
            "{%- if message['role'] == 'system' %}\n{{ message['content'] }}\n"
            "{%- else %}\n"
            "{%- if message['role'] == 'user' %}\n"
            "{{'### Instruction:\\n' + message['content'] + '\\n'}}\n "
            "{%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n"
            "{%- endif %}\n"
            "{%- endif %}\n"
            "{%- endfor %}\n"
            "{% if add_generation_prompt %}\n"
            "{{'### Response:'}}\n"
            "{% endif %}"
        )
    return tokenizer


@lru_cache(maxsize=1)
def load_quant_model(model_id):
    # This requires the `bitsandbytes` library
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_config = AutoConfig.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = _get_tokenizer_with_system_prompt(model_id)

    return model, tokenizer


def conversational_pipeline(model_id, max_new_tokens=500):
    model, tokenizer = load_quant_model(model_id)

    return pipeline(
        "conversational",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=max_new_tokens,
        device_map="auto",
        #streamer=TextStreamer(tokenizer)
    )

    
def conversation_init(prompt, user_query, few_shot = False, cot = False, rag = False, cheating=False):
    conversation = Conversation()
    #TODO delete cheating
    conversation.add_message({'role':'system', 'content': prompt.SYSTEM})
    conversation.add_message({'role':'user', 'content': prompt.generate_prompt(user_query, few_shot, cot, rag, cheating=cheating)})

    return conversation

def conversation_init_dict(prompt, user_query, few_shot = False, cot = False, rag = False, cheating=False):
    messages=[
        {"role": "system", "content": prompt.SYSTEM},
        {"role": "user", "content": prompt.generate_prompt(user_query, few_shot, cot, rag, cheating=cheating) },
    ]
    return messages

def conversational_pipeline_st(model_id, max_new_tokens = 500):
    model, tokenizer = load_quant_model(model_id)

    streamer = TextStreamer(tokenizer)
    pipe = pipeline("conversational",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    max_new_tokens=max_new_tokens,
                    device_map="auto",
                    streamer=streamer)
    
    return streamer, pipe