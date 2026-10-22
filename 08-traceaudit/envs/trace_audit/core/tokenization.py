"""Optional model-token counting, outside the standard-library core."""

from __future__ import annotations


def token_counter(encoding_name=None):
    if encoding_name is None:
        return None
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)

    def count(text):
        # Archived control-token spellings are ordinary evidence, not special tokens.
        return len(encoding.encode(text, disallowed_special=()))

    count.encoding_name = encoding_name
    return count
