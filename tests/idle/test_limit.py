"""A call's max duration: it ends the call once, unless the call ends first."""

import asyncio

import pytest

from unmute_core.idle import CallLimit


async def test_the_limit_ends_the_call_once_its_time_is_up():
    ended: list[bool] = []
    limit = CallLimit(0.01, lambda: ended.append(True))
    limit.start()
    limit.start()
    assert limit.running
    async with asyncio.timeout(1):
        while not ended:
            await asyncio.sleep(0)
    assert ended == [True]
    assert not limit.running


async def test_an_async_end_is_awaited():
    ended = asyncio.Event()

    async def end() -> None:
        ended.set()

    CallLimit(0.01, end).start()
    await asyncio.wait_for(ended.wait(), 1)


async def test_a_call_that_ends_first_cancels_the_limit():
    ended: list[bool] = []
    limit = CallLimit(0.02, lambda: ended.append(True))
    limit.start()
    limit.cancel()
    await asyncio.sleep(0.05)  # past the 0.02 s limit, which would end the call
    assert ended == []


@pytest.mark.parametrize("seconds", [0, -5])
def test_a_limit_that_is_not_above_zero_is_refused(seconds):
    with pytest.raises(ValueError, match="Input should be greater than 0"):
        CallLimit(seconds, lambda: None)


@pytest.mark.parametrize(
    "call", [lambda: CallLimit(0, lambda: None), lambda: CallLimit(seconds=0, end=lambda: None)]
)
def test_a_limit_error_names_the_class_and_the_field(call):
    with pytest.raises(ValueError) as caught:
        call()
    assert str(caught.value).splitlines()[:2] == ["1 validation error for CallLimit", "seconds"]
