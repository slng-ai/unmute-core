from unmute_core.model import Usage


def test_usage_passes_a_null_count_through_as_before() -> None:
    assert Usage(5, None).output_tokens is None  # ty: ignore[invalid-argument-type]
