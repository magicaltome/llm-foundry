# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

from enum import Enum
import math
import re
from typing import Optional

import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['MoedlGauntlet']

class Weighting(Enum):
    EQUAL = 1
    SAMPLE_SZ = 2
    LOG_SAMPLE_SZ = 3
class MoedlGauntlet(Callback):

    def __init__(
        self,
        logger_keys: dict,
        tasks: dict,
        weighting: Weighting = Weighting.EQUAL,
        subtract_random_baseline: bool = True,
        rescale_accuracy: bool = True,
        benchmark_sizes: Optional[dict] = None
    ):
        self.tasks = tasks
        self.weighting = Weighting[weighting]
        self.subtract_random_baseline = subtract_random_baseline
        self.rescale_accuracy = rescale_accuracy
        self.logger_keys = logger_keys
        for category in self.tasks:
            
            for benchmark in category['benchmarks']:
                bench_name = f"{benchmark['name']}/{benchmark['num_fewshot']}-shot"
                cumulative_samples = max(
                    sum(count for name,count in benchmark_sizes.items() if name.startswith(bench_name)),
                    1)
                
                if self.weighting == Weighting.EQUAL:
                    weight = 1
                elif self.weighting == Weighting.SAMPLE_SZ:
                    weight = cumulative_samples
                elif self.weighting == Weighting.LOG_SAMPLE_SZ:
                    weight = max(
                        math.log(cumulative_samples, 2),
                        1
                    )

                benchmark['weighting'] = weight
        
    def compute_averages(self, logger_data):

        results = {}
        pat = re.compile(
            'metrics/(.*?)/(\d+)-shot(/.*?)?/InContextLearning(.*)')
        for key in self.logger_keys:
            match = pat.match(key)
            val = logger_data.data[key][0][1].item()

            if match:
                eval_name = match.group(1)
                num_shot = match.group(2)
                subcat = match.group(3)
                metric = match.group(4)

                if subcat is not None:
                    subcat = subcat[1:]
                    if f'metrics/{eval_name}/{num_shot}-shot/InContextLearning{metric}' not in results:
                        results[f'metrics/{eval_name}/{num_shot}-shot/InContextLearning{metric}'] = []
                    results[
                        f'metrics/{eval_name}/{num_shot}-shot/InContextLearning{metric}'].append(
                            val)
                else:
                    results[key] = [val]
        return {k: sum(v) / len(v) for k, v in results.items()}

    def eval_end(self, state: State, logger: Logger):
        new_metrics = self.compute_averages(logger)
        composite_scores = {}
        for category in self.tasks:
            composite_scores[category['name']] = []
            for benchmark in category['benchmarks']:
                key_pat = re.compile(
                    f"metrics/{benchmark['name']}/{benchmark['num_fewshot']}-shot/.*Accuracy"
                )

                matching_key = [
                    k for k in new_metrics.keys()
                    if key_pat.match(k) is not None
                ]
                if len(matching_key) == 0:
                    print(
                        f"Warning: couldn't find results for benchmark: {benchmark}"
                    )
                else:
                    score = new_metrics[matching_key[0]]

                    if self.subtract_random_baseline:
                        score -= benchmark['scorecard']['random_baseline']

                    if self.rescale_accuracy and self.subtract_random_baseline:
                        score /= 1.0 - benchmark['scorecard']['random_baseline']

                    composite_scores[category['name']].append({
                        'name': benchmark['name'],
                        'score': score,
                        'weighting': benchmark['weighting']
                    })
            total_weight = sum(
                k['weighting'] for k in composite_scores[category['name']])
            composite_scores[category['name']] = sum(
                k['score'] * (k['weighting'] / total_weight)
                for k in composite_scores[category['name']])

        composite_scores = {
            f'metrics/icl_taxonomy/{k}': v for k, v in composite_scores.items()
        }

        composite_scores['metrics/icl_taxonomy/average'] = sum(
            composite_scores.values()) / len(composite_scores.values())
        logger.log_metrics(composite_scores)

        return composite_scores