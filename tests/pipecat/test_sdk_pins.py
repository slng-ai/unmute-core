"""Pin the Pipecat internals the adapter depends on, so an upgrade that moves one fails here.

Each test names the adapter code that relies on it, and says why.
"""

import dataclasses
import importlib
import inspect
from typing import Any, cast

import pytest
from pipecat.flows import FlowManager
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    LatencyContribution,
    UserBotLatencyObserver,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallRegistryItem,
    FunctionCallRunnerItem,
    LLMService,
)
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from tests.pipecat.harness import FakeWorker


def test_flows_still_keeps_a_pending_transition() -> None:
    """``UnmuteFlows._leave_flow`` clears ``FlowManager._pending_transition`` of an abandoned run."""
    flow = FlowManager(llm=cast(Any, None), context_aggregator=None, worker=cast(Any, FakeWorker()))
    assert flow._pending_transition is None


@pytest.mark.parametrize(
    "serializer",
    [
        lambda: TwilioFrameSerializer("MZ1", "CA1", "AC1", "secret"),
        lambda: TelnyxFrameSerializer("s1", "PCMU", "PCMU", "v3:abc", "KEY"),
        lambda: PlivoFrameSerializer("s1", "call-1", "MA1", "secret"),
    ],
    ids=["twilio", "telnyx", "plivo"],
)
def test_a_carrier_serializer_still_has_a_hang_up_flag(serializer: Any) -> None:
    """``disarm_hang_up`` reads and sets ``_hangup_attempted`` to stop the hang-up after a transfer."""
    assert serializer()._hangup_attempted is False


def test_the_realtime_service_still_has_its_private_context_and_setup_flag() -> None:
    """``RealtimeVoice.wait_for_caller`` sets ``_context`` and ``_llm_needs_conversation_setup``."""
    llm = OpenAIRealtimeLLMService(api_key="test")
    assert llm._context is None
    assert isinstance(llm._llm_needs_conversation_setup, bool)


def test_the_realtime_service_still_notes_when_its_session_is_ready() -> None:
    """The conformance driver opens a realtime call once ``_api_session_ready`` is set."""
    assert OpenAIRealtimeLLMService(api_key="test")._api_session_ready is False


def test_an_llm_service_still_has_a_base_system_instruction_and_update_settings() -> None:
    """``RequestHooks`` reads ``_base_system_instruction`` and calls ``_update_settings`` to lend a prompt."""
    assert inspect.iscoroutinefunction(LLMService._update_settings)
    llm = OpenAILLMService(api_key="test")
    assert hasattr(llm, "_base_system_instruction")


def test_the_public_settings_property_is_the_services_own_settings_object() -> None:
    """Every ``service.settings`` read in the adapter relies on it being the object ``_settings`` is."""
    llm = OpenAILLMService(api_key="test")
    assert llm.settings is llm._settings


def test_a_google_service_still_has_its_client_and_request_builders() -> None:
    """``SideRequests.google_ask`` builds a request as the service does, and sends it on its client."""
    pytest.importorskip("google.genai")
    google = importlib.import_module("pipecat.services.google.llm")
    settings = google.GoogleLLMService.Settings(model="gemini-x")
    llm = google.GoogleLLMService(api_key="test", settings=settings)
    assert llm._client is not None
    assert inspect.iscoroutinefunction(llm._client.aio.models.generate_content)
    assert isinstance(llm._should_inject_trailing_user_message(), bool)
    tool_config = {"function_calling_config": {"mode": "ANY"}}
    assert llm._build_generation_params(tool_config=tool_config)["tool_config"] == tool_config


def test_an_openai_service_still_has_its_client() -> None:
    """``UnmuteFlows._native_ask`` sends a side request through ``_client.chat.completions``."""
    llm = OpenAILLMService(api_key="test")
    assert callable(llm._client.chat.completions.create)


def test_an_openai_service_still_has_the_hooks_of_a_failed_request() -> None:
    """``ModelCallLLMService._process_context`` wraps ``_process_context``, fires ``_call_event_handler``
    and the ``on_completion_timeout`` event."""
    assert inspect.iscoroutinefunction(BaseOpenAILLMService._process_context)
    assert inspect.iscoroutinefunction(BaseOpenAILLMService._call_event_handler)
    llm = OpenAILLMService(api_key="test")
    assert "on_completion_timeout" in llm._event_handlers


def test_a_parallel_pipeline_still_tracks_the_frames_it_has_seen() -> None:
    """``VoiceSwitcher.process_frame`` discards the flush probe's id from ``_seen_ids``."""
    assert isinstance(ParallelPipeline([FrameProcessor()], [FrameProcessor()])._seen_ids, set)


def test_function_call_params_still_has_the_fields_the_adapter_builds() -> None:
    """``UnmuteFlows._transfer`` builds ``FunctionCallParams`` for a group's transfer target."""
    names = {field.name for field in dataclasses.fields(FunctionCallParams)}
    wanted = {
        "function_name",
        "tool_call_id",
        "arguments",
        "llm",
        "pipeline_worker",
        "context",
        "result_callback",
        "app_resources",
    }
    assert wanted <= names


def test_an_llm_service_still_runs_function_calls() -> None:
    """``UnmuteFlows`` hands a tool call back to the service through ``run_function_calls``."""
    assert inspect.iscoroutinefunction(LLMService.run_function_calls)


def test_the_latency_observer_still_has_its_events_and_fields() -> None:
    """``DevObserver`` listens to three latency events and reads the breakdown's fields."""
    observer = UserBotLatencyObserver()
    for event in ("on_latency_measured", "on_first_bot_speech_latency", "on_latency_breakdown"):
        assert event in observer._event_handlers
    assert {"contributions", "measured_from", "total_secs", "user_turn_secs"} <= set(
        LatencyBreakdown.model_fields
    )
    assert {"key", "label", "owner", "owner_kind", "start_time", "duration_secs"} <= set(
        LatencyContribution.model_fields
    )


def test_a_processor_still_has_the_frame_events_the_adapter_follows() -> None:
    """``RequestHooks``, ``attach_call``, ``UnmuteFlows`` and the realtime replies add handlers by name."""
    llm = OpenAILLMService(api_key="test")
    for event in (
        "on_before_process_frame",
        "on_after_process_frame",
        "on_before_push_frame",
        "on_after_push_frame",
    ):
        assert event in llm._event_handlers


def test_a_worker_still_has_the_end_events_the_adapter_follows() -> None:
    """``attach_call`` and ``attach_realtime`` add handlers to ``on_pipeline_finished`` and ``on_pipeline_error``."""
    worker = PipelineWorker(cast(Any, FrameProcessor()))
    for event in ("on_pipeline_finished", "on_pipeline_error"):
        assert event in worker._event_handlers


def test_a_realtime_service_still_skips_the_results_it_has_marked_completed() -> None:
    """``withhold_result`` adds a call id to ``_completed_tool_calls``, which the service never sends."""
    llm = OpenAIRealtimeLLMService(api_key="test")
    assert isinstance(llm._completed_tool_calls, set)
    source = inspect.getsource(OpenAIRealtimeLLMService._process_completed_function_calls)
    assert "not in self._completed_tool_calls" in source


def test_a_realtime_service_still_sends_one_tool_result_without_a_reply() -> None:
    """``send_quietly`` calls ``_send_tool_result``, which creates the item and no response."""
    source = inspect.getsource(OpenAIRealtimeLLMService._send_tool_result)
    assert "ConversationItemCreateEvent" in source
    assert "ResponseCreateEvent" not in source


def test_a_live_service_still_answers_only_the_calls_it_holds_open() -> None:
    """``withhold_result`` pops a call from ``_open_function_calls``, so the live service sends no output."""
    from pipecat.services.openai.live.llm import OpenAILiveLLMService

    llm = OpenAILiveLLMService(api_key="test")
    assert isinstance(llm._open_function_calls, dict)
    source = inspect.getsource(OpenAILiveLLMService._handle_function_call_result)
    assert "not in self._open_function_calls" in source


def test_an_llm_service_still_lists_its_running_calls_with_their_handlers() -> None:
    """``report`` reads ``_function_call_tasks`` for the other calls that can end the call."""
    llm = OpenAILLMService(api_key="test")
    assert llm._function_call_tasks == {}
    fields = {field.name for field in dataclasses.fields(FunctionCallRunnerItem)}
    assert {"tool_call_id", "registry_item"} <= fields
    assert "handler" in {field.name for field in dataclasses.fields(FunctionCallRegistryItem)}


def test_an_llm_service_still_composes_the_async_tool_text_from_has_async_tools() -> None:
    """``hide_async_guidance`` sets ``_has_async_tools`` on the instance and calls ``_compose_system_instruction``."""
    llm = OpenAILLMService(
        api_key="test", settings=OpenAILLMService.Settings(system_instruction="Hi.")
    )
    llm.register_function("slow", cast(Any, None), cancel_on_interruption=False)
    llm._compose_system_instruction()
    assert "ASYNC TOOLS" in str(cast(Any, llm.settings).system_instruction)
    llm._has_async_tools = lambda: False  # ty: ignore[invalid-assignment]
    llm._compose_system_instruction()
    assert cast(Any, llm.settings).system_instruction == "Hi."


def test_a_service_still_registers_the_handler_it_is_given_with_its_call_options() -> None:
    """``hide_async_guidance`` reads ``_functions`` for ``handler`` and ``cancel_on_interruption``."""
    llm = OpenAILLMService(api_key="test")

    async def handler(params: Any) -> None:
        """Do nothing."""

    llm.register_function("slow", handler, cancel_on_interruption=False)
    assert llm._functions["slow"].handler is handler
    assert llm._functions["slow"].cancel_on_interruption is False


async def test_flows_still_wraps_a_handler_in_a_function_that_closes_over_it() -> None:
    """``hide_async_guidance`` finds a step tool's handler in its Flows transition function's closure."""
    flow = FlowManager(llm=cast(Any, None), context_aggregator=None, worker=cast(Any, FakeWorker()))

    async def handler(args: Any, manager: Any) -> None:
        """Do nothing."""

    wrapped = await flow._create_transition_func("slow", handler)
    assert handler in [cell.cell_contents for cell in cast(Any, wrapped).__closure__ or ()]


def test_flows_still_builds_a_node_function_schema_in_a_method_we_extend() -> None:
    """``ParametersFlowManager`` extends ``_create_function_schema``, which drops the parameters."""
    parameters = inspect.signature(FlowManager._create_function_schema).parameters
    assert list(parameters) == ["self", "tool"]
