import inspect
from collections.abc import Callable

from verl import DataProto
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.experimental.reward.reward_loop.registry import register
from verl.utils.reward_score.custom import custom_compute_score


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
class CustomRewardLoopManager(RewardLoopManagerBase):
    """
    Based on DAPORewardLoopManager
    https://github.com/volcengine/verl/blob/5d00c08217696cb87bbaa2f00e7f2a4fe0e2255a/verl/experimental/reward/reward_loop/dapo.py?plain=1#L24
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score: Callable[..., float] | None = None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer)
        # Hard code the compute_score function, otherwise some training loops will set a different
        # compute_score, which is undesirable
        self.compute_score = custom_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        overlong_buffer_cfg = config.reward_model.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward_model.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
