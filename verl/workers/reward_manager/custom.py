import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from multiprocessing import get_context

import numpy as np
import torch
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    NormalizationConfig,
)
from tqdm import tqdm

from verl import DataProto
from verl.utils.reward_score.custom.custom import custom_compute_score
from verl.workers.reward_manager import register as register
from verl.workers.reward_manager.abstract import AbstractRewardManager

normalization_config = NormalizationConfig(
    basic_latex=True,
    units=True,
    malformed_operators=False,
    nits=False,
    boxed="all",
    equations=False,
)

# NOTE: @goon - It's crucial that gold_extraction_target be a tuple, not just a
# LatexExtractionConfig object.
gold_extraction_target = (
    LatexExtractionConfig(
        normalization_config=normalization_config,
    ),
)
pred_extraction_target = (
    ExprExtractionConfig(),
    LatexExtractionConfig(
        normalization_config=normalization_config,
    ),
)


def parallel_compute_score(
    compute_score_func: Callable[..., float],
    response_str: list[str],
    ground_truth: list[str],
    data_sources: list[str],
    max_workers=16,
    timeout_seconds=30,
) -> list[float]:
    scores_list = []
    with (
        tqdm(total=len(response_str)) as pbar,
        ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor,
    ):
        futures = [
            executor.submit(compute_score_func, resp, gt, ds)
            for resp, gt, ds in zip(response_str, ground_truth, data_sources, strict=True)
        ]

        for future in futures:
            try:
                score = future.result(timeout=timeout_seconds)
                scores_list.append(score)
            except (FutureTimeoutError, Exception) as e:
                warnings.warn(
                    f"Hit exception {e} in parallel_compute_score, scoring as 0.0. "
                    f"\n{response_str=}"
                    f"\n{ground_truth=}"
                    f"\n{data_sources=}",
                    stacklevel=1,
                )
                scores_list.append(0.0)
            pbar.update(1)

    return scores_list


# function for finding repetitions
def find_repetitions(
    tokens: list,
    min_len: int = 2,
    return_spans: bool = False,
):
    # - map from position -> last seen index most
    #   recent
    most_recent_idx = {}

    # - store all the repeats in format
    #   (idx, start_of_repeat)
    repeats = []

    # - some constants
    N = len(tokens)  # length

    # -
    # - start of repeat
    # - current idx
    mri_prev, start_rep, n = None, -1, -1

    # loop over tokens
    while n < N:
        # advance
        x = tokens[n]

        # index which the current token appeared before
        mri_curr = most_recent_idx.get(x)
        most_recent_idx[x] = n  # update

        # - if x was seen before and one of two conds
        # 1. x was also the prev token (i.e, if tokens[n-1] == x)
        # 2. the prev token was seen one position behind position
        #    current token was also last seen
        if (mri_curr is not None) and (
            (n > 0 and x == tokens[n - 1]) or (mri_prev is not None and mri_curr == (mri_prev + 1))
        ):
            repeats.append((n, start_rep))
        else:
            # this will be a new pattern, record
            # this pos as the start of (potential) future repeats
            start_rep = n

        # for next loop
        mri_prev = mri_curr
        n += 1

    if not return_spans:
        return repeats

    # convert to spans
    spans = {}
    for e, s in repeats:
        spans[s] = max(e, spans.get(s, -1))

    return [(s, e + 1) for s, e in spans.items() if (e - s + 1) >= min_len]


@register("custom")
class CustomRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Callable[..., float] | None = None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # Hard code the compute_score function, otherwise some training loops will set a different
        # compute_score, which is undesirable
        self.compute_score = custom_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        # NOTE: @goon - this block is apparently crucial. Seems to prevent us from double computing
        # scores? Not very clear.

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        ground_truth = [x.tolist() if isinstance(x, np.ndarray) else x for x in ground_truth]
        data_sources = data.non_tensor_batch["data_source"]

        assert len(response_str) == len(ground_truth) == len(data_sources)

        scores = []
        batch_size = 1024
        for i in range(0, len(response_str), batch_size):
            cur_response_str = response_str[i : i + batch_size]
            cur_ground_truth = ground_truth[i : i + batch_size]
            cur_data_sources = data_sources[i : i + batch_size]

            cur_scores = parallel_compute_score(
                self.compute_score,
                response_str=cur_response_str,
                ground_truth=cur_ground_truth,
                data_sources=cur_data_sources,
            )

            scores += cur_scores
        assert len(scores) == len(response_str)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward = scores[i]

            if self.overlong_buffer_cfg.enable:
                overlong = self.max_resp_len <= valid_response_length[i].item()
                reward_extra_info["overlong"].append(overlong)
                reward_extra_info["overlong_response_len"].append(valid_response_length[i].item())

                if overlong:
                    # we know that scores are 1 for correct and 0 for wrong
                    # - so we just put a different value
                    reward = self.overlong_buffer_cfg.penalty_factor  # HIJACK THIS

            # if self.overlong_buffer_cfg.rep_tail:
            #     tail = self.overlong_buffer_cfg.rep_tail
            #     threshold = self.overlong_buffer_cfg.rep_threshold
            #     rep_spans = find_repetitions(
            #         response_ids[i][
            #             max(valid_response_length[i] - tail, 0) : valid_response_length[
            #                 i
            #             ]
            #         ].numpy(),
            #         return_spans=True,
            #     )
            #     rep_tokens = sum([e - s for s, e in rep_spans])
            #     rep_ratio = rep_tokens / tail
            #
            #     if rep_ratio > threshold:
            #         reward = self.overlong_buffer_cfg.penalty_factor  # HIJACK THIS
            #
            #     reward_extra_info["overlong_rep_tokens"].append(rep_tokens)
            #     reward_extra_info["overlong_rep_ratio"].append(rep_ratio)
            #     reward_extra_info["overlong_rep_ratio_clip"].append(
            #         rep_ratio > threshold
            #     )
            #     reward_extra_info["overlong_rep_tail"].append(tail)
            #     reward_extra_info["overlong_rep_threshold"].append(threshold)
            #
            reward_tensor[i, valid_response_length[i].item() - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", response_str[i])

        reward_extra_info["acc"] = scores

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
