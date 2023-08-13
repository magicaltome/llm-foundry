# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations to wandb from a set of prompts."""
from typing import List, Union, cast

import torch
import wandb
from composer.core import Callback, State, get_precision_context
from composer.loggers import Logger, WandBLogger
from composer.utils import dist, ensure_tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class Generate(Callback):

    def __init__(self, prompts: List[str], batch_log_interval: int, **kwargs):
        """Periodically log generations to wandb from a set of prompts.

        In the main view for a run, there will be a table that will show the _last_ logged generations.
        To compare previous iterations of the generations, you need to
        1. Click on the run
        2. Click on "artifacts" in the menu on the left side of the screen
        3. Click on one of the artifacts called "predictions"
        4. Click on the "files" tab
        5. Click on "predictions.table.json"
        6. On the left hand side, there are different versions of the table produced throughout training. Select one of these.
        7. Now, when you hover over other versions, there will be a "compare" button, which will allow you to compare the currently
            selected version to the version you add via compare.

        Args:
            prompts (List[str]): The list of prompts you would like to produce generations for
            batch_log_interval (int): The interval (in batches) at which this callback runs
            kwargs: All kwargs well be passed along to the call to generate. This is for things like `do_sample`, `top_p`, etc
        """
        self.prompts = prompts
        self.batch_log_interval = batch_log_interval
        self.generate_kwargs = kwargs
        self.wandb_logger = None

    def init(self, state: State, logger: Logger):
        if dist.get_global_rank() == 0:
            for destination in ensure_tuple(logger.destinations):
                if isinstance(destination, WandBLogger):
                    self.wandb_logger = destination

    def batch_checkpoint(self, state: State, logger: Logger):
        if (state.timestamp.batch.value % self.batch_log_interval) == 0:
            self.generate(state, logger)

    def generate(self, state: State, logger: Logger):
        model = state.model
        original_mode = model.training
        model.eval()
        tokenizer = cast(Tokenizer, state.model.tokenizer)
        device = state.device

        # stash the original original value of padding_side because generation requires left padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_input = tokenizer(self.prompts,
                                    max_length=1024,
                                    return_tensors='pt',
                                    padding='max_length')

        for k, v in tokenized_input.items():
            tokenized_input[k] = device.tensor_to_device(v)

        # dummy forward call needed for FSDP to work consistently
        dummy_input = torch.tensor([[0]], dtype=torch.long)
        dummy_input = device.tensor_to_device(dummy_input)
        with get_precision_context(state.precision):
            with torch.no_grad():
              _ = model.model(input_ids=dummy_input)

            n_prompts = len(self.prompts)
            batch_size = 8
            n_batches = int(n_prompts / float(batch_size) + 0.5)
            outputs = []
            for batch, s in enumerate(range(0, n_prompts, batch_size)):
              print(f'[Generating outputs batch={batch}/{n_batches}]')
              e = min(s + batch_size, n_prompts)
              outputs.append(
                model.model.generate(
                  input_ids=tokenized_input['input_ids'][s:e],
                  attention_mask=tokenized_input['attention_mask'][s:e],
                  synced_gpus=True,
                  **self.generate_kwargs,
                )
              )
            
            print('Concatenting tensors')
            output_token_ids = torch.cat(outputs)

        if dist.get_global_rank() == 0:
            if self.wandb_logger is not None:
                assert wandb.run is not None, 'wandb should have started run'

                artifact = wandb.Artifact('generate_samples_' +
                                          str(wandb.run.name),
                                          type='predictions')

                rows = []
                for i in range(len(self.prompts)):
                    prompt = self.prompts[i]
                    output_tokens = output_token_ids[i][
                        tokenized_input['input_ids'].shape[1]:]
                    output_text = tokenizer.decode(output_tokens,
                                                   skip_special_tokens=True)

                    rows.append([prompt, output_text])
                text_table = wandb.Table(data=rows,
                                         columns=['prompt', 'generation'])
                artifact.add(text_table, 'predictions')
                wandb.log_artifact(artifact)
                wandb.log({'generations': text_table},
                          step=state.timestamp.batch.value)

        tokenizer.padding_side = original_padding_side
        model.train(mode=original_mode)
