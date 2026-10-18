"""The call's TTS and STT services on Pipecat: one of each runs at a time, a handoff switches."""

from pipecat.frames.frames import Frame, PipelineFlushFrame
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyManual
from pipecat.processors.frame_processor import FrameDirection


class _FlushSwitcher(ServiceSwitcher[ServiceSwitcherStrategyManual]):
    """A manual service switcher that Pipecat's flush probe crosses every time.

    A plain ``ServiceSwitcher`` lets each frame out once (``ParallelPipeline``
    ``_parallel_push_frame``). Pipecat's flush probe crosses it three times, so it is lost
    on its way back, and every ``flush_pipeline`` waits out its timeout: a handoff or a task
    would hang for seconds. Here the probe crosses each time. Only the running service's
    branch lets it out, so it crosses once per pass.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Let the flush probe cross again, then route the frame as a switcher does."""
        if isinstance(frame, PipelineFlushFrame):
            self._seen_ids.discard(frame.id)
        await super().process_frame(frame, direction)


class VoiceSwitcher(_FlushSwitcher):
    """The TTS services of a call's agents, in the pipeline where the TTS goes.

    Build it with every agent's voice, the first agent's first, and give it to
    ``UnmuteFlows(tts=...)``. ``hand_off(tts=...)`` makes another one speak.
    """


class STTSwitcher(_FlushSwitcher):
    """The STT services of a call's agents, in the pipeline where the STT goes.

    Build it with every agent's STT service, the first agent's first, and give it to
    ``UnmuteFlows(stt=...)``. ``hand_off(stt=...)`` makes another one listen.
    """
