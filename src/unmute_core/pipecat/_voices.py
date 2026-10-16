"""The call's voices on Pipecat: one TTS service speaks at a time, and a handoff switches."""

from pipecat.frames.frames import Frame, PipelineFlushFrame
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyManual
from pipecat.processors.frame_processor import FrameDirection


class VoiceSwitcher(ServiceSwitcher[ServiceSwitcherStrategyManual]):
    """The TTS services of a call's agents, in the pipeline where the TTS goes.

    Build it with every agent's voice, the first agent's first, and give it to
    ``UnmuteFlows(tts=...)``. ``hand_off(tts=...)`` makes another one speak.

    A plain ``ServiceSwitcher`` lets each frame out once (``ParallelPipeline``
    ``_parallel_push_frame``). Pipecat's flush probe crosses it three times, so it is lost
    on its way back, and every ``flush_pipeline`` waits out its timeout: a handoff or a task
    would hang for seconds. Here the probe crosses each time. Only the speaking voice's
    branch lets it out, so it crosses once per pass.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Let the flush probe cross again, then route the frame as a switcher does."""
        if isinstance(frame, PipelineFlushFrame):
            self._seen_ids.discard(frame.id)
        await super().process_frame(frame, direction)
