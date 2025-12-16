import warnings

from math_verify import parse, verify
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    NormalizationConfig,
)

normalization_config = NormalizationConfig(
    basic_latex=True,
    units=True,
    malformed_operators=False,
    nits=False,
    boxed="all",
    equations=False,
)

gold_extraction_target = LatexExtractionConfig(
    normalization_config=normalization_config,
)
pred_extraction_target = (
    ExprExtractionConfig(),
    LatexExtractionConfig(
        normalization_config=normalization_config,
    ),
)

# TODO: @goon - restore timeout handling


def math_verify_reward_function(solution_str: str, ground_truth: str) -> float:
    # We need to set parsing_timeout=None, otherwise we hit this exception:
    # "Math-Verify 'parse' function doesn't support threaded environment due to usage of signal.alarm() in timeout mechanism ... "
    # https://github.com/huggingface/Math-Verify/issues/42#issuecomment-2766669510
    # https://github.com/garrett361/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/src/math_verify/parser.py?plain=1#L714
    try:
        math_verify_parsed = parse(solution_str, pred_extraction_target, parsing_timeout=None)
    except Exception as e:
        warnings.warn(
            f"Caught exception {e} while parsing math {solution_str=}, scoring as 0.0",
            stacklevel=1,
        )
        return 0.0

    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0

    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0

    # We now fallback to semantic verification
    try:
        # We need to set parsing_timeout=None, otherwise we hit this exception:
        # "Math-Verify 'parse' function doesn't support threaded environment due to usage of signal.alarm() in timeout mechanism ... "
        # https://github.com/huggingface/Math-Verify/issues/42#issuecomment-2766669510
        # https://github.com/garrett361/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/src/math_verify/parser.py?plain=1#L714
        correct = verify(
            parse(
                f"\\boxed{{{ground_truth}}}",
                gold_extraction_target,
                parsing_timeout=None,
            ),
            math_verify_parsed,
            timeout_seconds=None,
        )
        return 1.0 if correct else 0.0
    except Exception as e:
        warnings.warn(
            f"Caught exception {e} while verifying math {ground_truth=}, {math_verify_parsed=}, scoring as 0.0",
            stacklevel=1,
        )
        return 0.0


def custom_compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str,
    extra_info: dict | None = None,
    search_last_chars: int = 300,
    **extra_reward_kwargs,
) -> float:
    # Take the final post-think string, if needed
    solution_str = solution_str.split("</think>")[-1]
    # if data_source == "math_verify":
    if "math" in data_source:
        if search_last_chars is not None:
            solution_str = solution_str[-search_last_chars:]
        return math_verify_reward_function(solution_str, ground_truth)
    raise ValueError(f"Unexpected {data_source=}")
