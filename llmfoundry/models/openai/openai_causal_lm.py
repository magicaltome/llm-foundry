# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import os
from time import sleep
from typing import Any, Optional, Union

import openai
import tiktoken
import torch
from composer.metrics import InContextLearningMetric
# required for loading a python model into composer
from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError,
                                  InContextLearningMultipleChoiceAccuracy,
                                  InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)
from composer.models import ComposerModel
from openai.error import RateLimitError, ServiceUnavailableError, APIError
from torchmetrics import Metric
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

__all__ = ['OpenAICausalLMEvalWrapper', 'OpenAITokenizerWrapper']

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


class OpenAICausalLMEvalWrapper(ComposerModel):

    def __init__(self, model_cfg, tokenizer):
        self.model_name = model_cfg['version']
        self.tokenizer = tokenizer
        self.chat_model = model_cfg.get('chat_model', False)
        # set up training and eval metrics
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError()
        ]
        self.eval_metrics = {
            metric.__class__.__name__: metric for metric in eval_metrics
        }
        super(OpenAICausalLMEvalWrapper, self).__init__()
        self.mocked_layer = torch.nn.Linear(2, 3)

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = []
        else:
            metrics = self.eval_metrics

        return metrics if metrics else {}

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        output_logits_batch = []
        for tokens, cont_idxs in zip(batch['input_ids'],
                                    batch['continuation_indices']):
            seqlen = tokens.shape[0]
            tokens = tokens.tolist()
            cont_idxs = cont_idxs.tolist()
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]

            output_logits = torch.zeros(cont_idxs[0] - 1,
                                        self.tokenizer.pad_token_id + 1)
            for i in range(len(expected_cont_tokens)):
                # decode one token at a time
                prompt = self.tokenizer.decode(tokens[:cont_idxs[0]] +
                                            expected_cont_tokens[0:i])
                if not self.chat_model:
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
                        except RateLimitError:
                            sleep(60)
                            continue

                    if len(chat_completion['choices'][0]['logprobs']
                        ['top_logprobs']) > 0:
                        tensor = self.tokenizer.construct_logit_tensor(
                            dict(chat_completion['choices'][0]['logprobs']
                                ['top_logprobs'][0]))
                    else:
                        # the model sometimes stops early even though we are still requesting tokens!
                        # not sure if there's a fix
                        continue

                    output_logits = torch.cat(
                        [output_logits, tensor.reshape(1, -1)])
                else:
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
                        except ServiceUnavailableError:
                            continue
                        except APIError:
                            continue
                        except RateLimitError:
                            sleep(60)
                            continue

                    if len(chat_completion['choices']) > 0:
                        tensor = self.tokenizer.construct_logit_tensor({
                            chat_completion['choices'][0]['message']['content']:
                                0.0
                        })
                    else:
                        # the model sometimes stops early even though we are still requesting tokens!
                        # not sure if there's a fix
                        continue

                    output_logits = torch.cat(
                        [output_logits, tensor.reshape(1, -1)])

            output_logits = torch.cat([
                output_logits,
                torch.zeros(seqlen - output_logits.shape[0],
                            self.tokenizer.pad_token_id + 1)
            ])
            output_logits_batch.append(output_logits)

        return torch.stack(output_logits_batch).to(batch['input_ids'].device)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        self.labels = batch.pop('labels')
        self.labels[:, :-1] = self.labels[:, 1:].clone()
        self.labels[:, -1] = -100

        if isinstance(metric, InContextLearningMetric) and batch.get(
                'mode', None) == 'icl_task':
            assert self.labels is not None
            metric.update(batch, outputs, self.labels)
        else:
            metric.update(
                outputs,
                self.labels)  # pyright: ignore [reportGeneralTypeIssues]

    def forward(self):
        pass

    def loss(self):
        pass
