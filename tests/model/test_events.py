from unmute_core.model import Usage


def test_usage_passes_a_null_count_through_as_before() -> None:
    assert Usage(5, None).output_tokens is None  # ty: ignore[invalid-argument-type]


def test_usage_has_no_cache_write_tokens_by_default() -> None:
    assert Usage(5, 2).cache_write_tokens is None
    assert Usage(5, 2, 3, "r", "q", 1, 8, 4).cache_write_tokens == 4
