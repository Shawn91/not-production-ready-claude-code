import tiktoken


def get_tokenizer(model: str = "cl100k_base"):
    try:
        encoding = tiktoken.get_encoding(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    tokenizer = get_tokenizer(model)
    if tokenizer:
        return len(tokenizer(text))
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def truncate_text(
    text: str,
    max_tokens: int,
    model: str = "cl100k_base",
    suffix: str = "\n... [truncated]",
    preserve_lines: bool = True,
):
    """
    Truncates the given text to fit within the specified token limit.

    Args:
        text (str): The input text to be truncated.
        model (str): The tokenizer model to use for tokenization.
        max_tokens (int): The maximum number of tokens allowed.
        suffix (str): The suffix to append when truncating.
        preserve_lines (bool): Whether to preserve line breaks during truncation.

    Returns:
        str: The truncated text.
    """
    current_tokens = count_tokens(text, model)
    if current_tokens <= max_tokens:
        return text
    suffix_tokens = count_tokens(suffix, model)
    target_token_num = max_tokens - suffix_tokens
    if target_token_num <= 0:
        return suffix.strip()
    if preserve_lines:
        return _truncate_by_lines(text, target_token_num, suffix, model)
    else:
        return _truncate_by_chars(text, target_token_num, suffix, model)


def _truncate_by_lines(text: str, target_tokens: int, suffix: str, model: str) -> str:
    lines = text.split("\n")
    result_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line + "\n", model)
        if current_tokens + line_tokens > target_tokens:
            break
        result_lines.append(line)
        current_tokens += line_tokens

    if not result_lines:
        # Fall back to character truncation if no complete lines fit
        return _truncate_by_chars(text, target_tokens, suffix, model)

    return "\n".join(result_lines) + suffix


def _truncate_by_chars(text: str, target_tokens: int, suffix: str, model: str) -> str:
    # Binary search for the right length
    low, high = 0, len(text)

    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid], model) <= target_tokens:
            low = mid
        else:
            high = mid - 1

    return text[:low] + suffix
