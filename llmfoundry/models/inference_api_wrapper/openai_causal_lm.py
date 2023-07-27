# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import os
from time import sleep
from typing import Union

import openai
import tiktoken
import torch

from openai.error import RateLimitError, ServiceUnavailableError, APIError
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from llmfoundry.models.inference_api_wrapper.interface import InferenceAPIEvalWrapper

__all__ = ['OpenAICausalLMEvalWrapper', 'OpenAIChatAPIEvalWrapper', 'OpenAITokenizerWrapper']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
openai.api_key = os.getenv('OPENAI_API_KEY')


class OpenAITokenizerWrapper:
    def __init__(self, name) -> None:
        self.tokenizer = tiktoken.encoding_for_model(name)

    def __call__(self, x, add_special_tokens=False):
        return self.encode(x)

    def encode(self, x, add_special_tokens=False):
        if isinstance(x, str):
            return {
                'input_ids':
                    self.tokenizer.encode(x, allowed_special={'<|endoftext|>'})
            }
        else:
            return {
                'input_ids':
                    self.tokenizer.encode_batch(
                        x, allowed_special={'<|endoftext|>'})
            }

    def decode(self, x):
        return self.tokenizer.decode(x)

    @property
    def pad_token_id(self):
        return self.tokenizer.eot_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eot_token

    def construct_logit_tensor(self, logprobs):
        tensor = torch.tensor([min(logprobs.values()) - 1] *
                              (self.pad_token_id + 1))
        for k in logprobs:
            idx = self.encode(k)['input_ids'][0]
            tensor[idx] = logprobs[k]
        return tensor
    
class OpenAIChatAPIEvalWrapper(InferenceAPIEvalWrapper):
    def get_next_token_logit_tensor(self, prompt):
        while True:
            try:
                chat_completion = openai.ChatCompletion.create(
                                model=self.model_name,
                                messages=[{
                                    'role': 'user',
                                    'content': prompt
                                }],
                                max_tokens=1,
                                temperature=0.0)
                break
            except ServiceUnavailableError as e:
                continue
            except APIError:
                continue
            except RateLimitError as e:
                if 'You exceeded your current quota' in e._message:
                    raise e
                sleep(60)
                continue

        if len(chat_completion['choices']) > 0:
            tensor = self.tokenizer.construct_logit_tensor({
                chat_completion['choices'][0]['message']['content']:
                    0.0
            })
            return tensor
        else:
            # the model sometimes stops early even though we are still requesting tokens!
            # not sure if there's a fix
            return None

class OpenAICausalLMEvalWrapper(InferenceAPIEvalWrapper):
    def get_next_token_logit_tensor(self, prompt):
        while True:
            try:
                chat_completion = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=5,
                    temperature=0.0)
                break
            except ServiceUnavailableError:
                continue
            except APIError:
                continue
            except RateLimitError as e:
                if 'You exceeded your current quota' in e._message:
                    raise e
                sleep(60)
                continue

        if len(chat_completion['choices'][0]['logprobs']
            ['top_logprobs']) > 0:
            tensor = self.tokenizer.construct_logit_tensor(
                dict(chat_completion['choices'][0]['logprobs']
                    ['top_logprobs'][0]))
            return tensor
        else:
            # the model sometimes stops early even though we are still requesting tokens!
            # not sure if there's a fix
            return None
    