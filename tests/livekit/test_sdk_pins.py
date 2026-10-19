"""Pin the LiveKit internals the adapter depends on, so an upgrade that moves one fails here.

Each test names the adapter code that relies on it, and says why.
"""

import asyncio
import inspect
import typing
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit.agents import AgentSession, APIConnectOptions, llm, metrics
from livekit.agents.beta.workflows import WarmTransferTask, task_group
from livekit.agents.beta.workflows.utils import WorkflowInstructions
from livekit.agents.metrics import base as metrics_base
from livekit.agents.voice import (
    AgentTask,
    SpeechHandle,
    agent_activity,
    agent_session,
    events,
    generation,
    speech_handle,
)
from livekit.agents.voice.audio_recognition import AudioRecognition

from tests.livekit.fake import Reply, ScriptedLLM
from unmute_core.agent import AgentSpec
from unmute_core.livekit import Backchannels, UnmuteAgent, interruption
from unmute_core.livekit._model import (
    INSTRUCTIONS_MESSAGE_ID,
    RUNNING_PLACEHOLDER_KEY,
)
from unmute_core.livekit._step import _played
from unmute_core.livekit._trace import plugin_voice


def test_the_copied_markers_match_the_installed_livekit() -> None:
    """``_model.request_context`` finds the running call and the prompt by these."""
    assert generation._RUNNING_PLACEHOLDER_KEY == RUNNING_PLACEHOLDER_KEY
    assert generation.INSTRUCTIONS_MESSAGE_ID == INSTRUCTIONS_MESSAGE_ID


async def test_a_session_still_has_its_private_closing_flag() -> None:
    """``UnmuteAgent._await_step`` reads it to tell a closing call from a broken step."""
    assert AgentSession()._closing is False


async def test_a_session_still_has_its_private_started_flag() -> None:
    """``testing._converse_one`` reads it to stop a scenario once the agent ended the call."""
    assert AgentSession()._started is False


async def test_a_session_still_answers_whether_it_is_closing() -> None:
    """``DevObserver`` reads ``_is_closing()`` to tell a hang-up from a tool that failed."""
    assert AgentSession()._is_closing() is False


async def test_a_task_built_with_history_still_drops_calls_to_tools_it_lacks() -> None:
    """Why ``StepTask`` sets its history in ``on_enter`` and not in the constructor."""
    ctx = llm.ChatContext()
    ctx.insert(llm.FunctionCall(call_id="c1", name="elsewhere", arguments="{}"))
    task = AgentTask(instructions="x", chat_ctx=ctx)
    assert [item.type for item in task.chat_ctx.items] == []


def test_task_group_still_adds_its_own_go_back_tool() -> None:
    """Why the adapter runs steps itself: TaskGroup adds ``out_of_scope`` with no switch."""
    assert "out_of_scope" in Path(task_group.__file__).read_text()


async def test_a_calls_item_id_starts_with_its_replys_message_id() -> None:
    """``StepTask.said`` finds the reply that ended a step by this naming."""
    model = ScriptedLLM(Reply("Hello.", [("unknown_tool", {})]))
    async with AgentSession(llm=model) as session:
        await session.start(UnmuteAgent(AgentSpec("x")))
        await session.run(user_input="hi")
    items = session.history.items
    message = next(i for i in items if isinstance(i, llm.ChatMessage) and i.role == "assistant")
    function = next(i for i in items if isinstance(i, llm.FunctionCall))
    assert function.id.rsplit("/fnc_", 1)[0] == message.id


def test_the_away_timer_still_waits_for_running_tools() -> None:
    """Why ``idle_nudges`` exists: a task runs inside its delegate tool, so it never starts."""
    source = Path(agent_session.__file__).read_text()
    assert "if _RunningTasks.get(self):" in source


async def test_a_cut_lines_played_words_are_on_its_speech_when_it_ends() -> None:
    """``StepTask._say_reply`` reads them to count a confirm cut after its values as heard."""
    speech = SpeechHandle.create()
    seen: list[str] = []
    speech.add_done_callback(lambda done: seen.append(_played(done)))
    cut = llm.ChatMessage(role="assistant", content=["Shall I book A-12"], interrupted=True)
    speech._item_added([cut])
    speech._mark_done()
    await asyncio.sleep(0)
    assert seen == ["Shall I book A-12"]


def _held_after_a_backchannel(backchannels: Backchannels) -> list[str]:
    """What LiveKit keeps of a "yes" judged a backchannel, once the line it was said over ends.

    The line plays from 100 s to 105 s. The caller said "yes" from 101.5 s to 102 s and is
    quiet when the line ends, so no speech of theirs is active then.
    """
    boundary = interruption(backchannels).get("backchannel_boundary", (1.0, 1.0))
    held = SimpleNamespace(speech_end_time=102.0, created_at=102.1, text="yes")
    recognition = SimpleNamespace(
        _backchannel_boundary=boundary,
        _agent_speech_started_at=100.0,
        _transcript_buffer=deque([held]),
    )
    AudioRecognition._trim_held_transcripts(
        cast(Any, recognition), resolved_at=105.0, vad_speech_started_at=None
    )
    return [event.text for event in recognition._transcript_buffer]


def test_a_backchannel_over_the_agent_is_kept_only_when_asked() -> None:
    """``interruption("keep")`` relies on this end boundary to keep the caller's words."""
    assert _held_after_a_backchannel("keep") == ["yes"]
    assert _held_after_a_backchannel("drop") == []


def test_an_llm_stream_still_has_the_event_channel_and_run_the_adapter_overrides() -> None:
    """``ModelCallLLM`` sends chunks with ``_event_ch.send_nowait``, and overrides ``_run``.

    ``_ModelCallStream._run`` replaces LiveKit's method of the same name, and LiveKit calls it
    by that name.
    """
    assert callable(llm.llm._LLMEventChannel.send_nowait)
    assert inspect.iscoroutinefunction(llm.LLMStream._run)


def test_a_stream_still_names_its_spans_by_the_class_names_a_chain_stream_sets() -> None:
    """``_ChainStream`` takes these three from LiveKit's ``FallbackLLMStream``."""
    from livekit.agents.llm.fallback_adapter import FallbackLLMStream

    from unmute_core.livekit._model import _ChainStream

    for name in ("_llm_request_span_name", "_llm_attempt_span_name", "_genai_operation_name"):
        assert getattr(_ChainStream, name) == getattr(FallbackLLMStream, name), name


def test_a_tts_plugin_still_keeps_its_voice_in_its_private_options() -> None:
    """``plugin_voice`` reads ``_opts.voice`` for the ``tts`` span. Skipped with no plugin."""
    openai = pytest.importorskip(
        "livekit.plugins.openai", reason="no LiveKit TTS plugin is installed"
    )
    plugin = openai.TTS(api_key="test", voice="alloy")
    assert plugin_voice(plugin) == "alloy"


def test_a_plugin_without_private_options_has_no_voice() -> None:
    """The span has no voice when a plugin keeps it elsewhere."""
    assert plugin_voice(None) is None


def test_connect_options_still_space_their_retries_the_same_way() -> None:
    """``_model._called`` retries a ``model=`` request with LiveKit's own spacing."""
    options = APIConnectOptions(max_retry=3, retry_interval=2.0)
    assert options._interval_for_retry(0) == 0.1
    assert options._interval_for_retry(1) == 2.0


def test_a_tool_round_still_lets_the_listener_cancel_every_tool_reply() -> None:
    """``end_call_now`` cancels the reply to every tool in the round that ends the call."""
    output = llm.FunctionCallOutput(
        call_id="c1", name="a", output="x", is_error=False, reply_required=True
    )
    event = events.FunctionToolsExecutedEvent(
        function_calls=[llm.FunctionCall(call_id="c1", name="a", arguments="{}")],
        function_call_outputs=[output],
    )
    assert event.has_tool_reply
    event.cancel_tool_reply()
    assert not event.has_tool_reply


def test_function_tools_executed_is_still_a_session_event() -> None:
    """``end_call_now`` listens for it with ``session.once``."""
    assert "function_tools_executed" in typing.get_args(events.EventTypes)


def test_a_model_stream_still_reads_its_retry_on_chunk_sent_flag() -> None:
    """``_ModelCallStream`` sets it to False, so a failure after a chunk is not retried."""
    source = inspect.getsource(llm.LLMStream)
    assert "self._retry_on_chunk_sent = " in source
    assert "not self._retry_on_chunk_sent and self._event_ch.output_sent" in source


def test_the_warm_transfer_task_still_takes_the_parameters_the_adapter_passes() -> None:
    """``SipLine._warm`` passes these names."""
    parameters = inspect.signature(WarmTransferTask).parameters
    for name in (
        "sip_call_to",
        "sip_trunk_id",
        "sip_connection",
        "sip_number",
        "chat_ctx",
        "instructions",
        "ringing_timeout",
    ):
        assert name in parameters
    assert set(inspect.signature(WorkflowInstructions).parameters) >= {"persona", "extra"}


def test_metrics_are_still_a_session_event_with_the_metrics_the_adapter_reads() -> None:
    """``SessionTrace`` and ``DevObserver`` listen for ``metrics_collected`` and read ``metrics``."""
    assert "metrics_collected" in typing.get_args(events.EventTypes)
    assert "metrics" in events.MetricsCollectedEvent.model_fields


def test_a_chat_messages_metrics_still_carry_the_keys_the_adapter_reads() -> None:
    """``SessionTrace._end_turn`` and ``DevTimings`` read these keys of ``ChatMessage.metrics``."""
    keys = llm.MetricsReport.__annotations__
    for name in (
        "started_speaking_at",
        "stopped_speaking_at",
        "transcription_delay",
        "end_of_turn_delay",
        "llm_node_ttft",
        "llm_node_ttfs",
        "tts_node_ttfb",
        "e2e_latency",
        "playback_latency",
        "stt_metadata",
    ):
        assert name in keys
    assert llm.ChatMessage.model_fields["metrics"].annotation is llm.MetricsReport


def test_llm_metrics_still_carry_the_fields_the_model_node_sets() -> None:
    """``ModelNode`` emits ``LLMMetrics`` on the agent's LLM, as ``LLMStream`` does."""
    fields = set(metrics.LLMMetrics.model_fields)
    assert {
        "label",
        "request_id",
        "timestamp",
        "duration",
        "ttft",
        "cancelled",
        "completion_tokens",
        "prompt_tokens",
        "prompt_cached_tokens",
        "total_tokens",
        "tokens_per_second",
        "metadata",
    } <= fields
    assert set(metrics_base.Metadata.model_fields) >= {"model_name", "model_provider"}


def test_a_stream_still_reports_its_metrics_from_a_task_it_starts_at_construction() -> None:
    """``ModelCallLLM.emit`` finds the stream through the context its metrics task copies.

    ``_ModelCallStream.answered_metrics`` then corrects the metrics and the request span.
    """
    from livekit.agents.telemetry import trace_types

    init = inspect.getsource(llm.LLMStream.__init__)
    assert "asyncio.create_task(\n            self._metrics_monitor_task(monitor_aiter)" in init
    assert "self._llm_request_span: trace.Span | None = None" in init
    monitor = inspect.getsource(llm.LLMStream._metrics_monitor_task)
    assert 'self._llm.emit("metrics_collected", metrics)' in monitor
    assert "self._llm_request_span.set_attribute(" in monitor
    assert trace_types.ATTR_LLM_METRICS == "lk.llm_metrics"
    assert (
        trace_types.ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK
        == "gen_ai.response.time_to_first_chunk"
    )


def test_the_models_whose_reasoning_markers_livekit_strips_are_the_ones_the_library_knows() -> None:
    """``MODEL_REASONING_MARKERS`` holds the markers LiveKit's inference models strip."""
    from livekit.agents.inference import llm as inference_llm

    from unmute_core.model import MODEL_REASONING_MARKERS

    theirs = {
        model.split("/")[-1]: markers for model, markers in inference_llm._MODEL_THINK_TAGS.items()
    }
    assert theirs == dict(MODEL_REASONING_MARKERS)


def test_an_activity_still_listens_for_metrics_on_its_llm() -> None:
    """``ModelNode`` emits ``metrics_collected`` on the LLM, and ``AgentActivity`` listens."""
    source = inspect.getsource(agent_activity)
    assert 'self.llm.on("metrics_collected", self._on_metrics_collected)' in source


def test_say_and_generate_reply_still_raise_the_not_running_message() -> None:
    """``open_call`` ends quietly on a closed session by this ``RuntimeError`` text."""
    from unmute_core.livekit._call import SESSION_NOT_RUNNING

    for method in (AgentSession.say, AgentSession.generate_reply):
        assert SESSION_NOT_RUNNING in inspect.getsource(method)


def test_the_interruption_timeout_and_the_release_of_running_tools_are_still_there() -> None:
    """``test_interrupted_tools`` shortens the timeout, and reads the placeholder a release writes."""
    assert speech_handle.INTERRUPTION_TIMEOUT == 5.0
    assert inspect.iscoroutinefunction(generation._release_running_tools)
    assert "still in progress" in generation._RUNNING_TOOL_PLACEHOLDER


def test_a_line_whose_voice_failed_is_not_an_interrupted_speech_but_its_message_is() -> None:
    """``_step._heard`` counts a line as heard only when neither says it was cut short."""
    activity = Path(agent_activity.__file__).read_text()
    assert "interrupted=speech_handle.interrupted\n" in activity
    assert "or any(out.tts_failed for out in segment_outputs)" in activity
    assert "out.tts_failed = tts_data is not None and tts_data.error is not None" in (
        Path(generation.__file__).read_text()
    )
