"""What one agent of a call is, the same on LiveKit and Pipecat."""

from collections.abc import Mapping
from typing import Self

from pydantic import Field, SkipValidation, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.model import ModelCall
from unmute_core.tasks import TaskPlan
from unmute_core.tasks._spec import AgentName
from unmute_core.tools import ToolRegistry


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class AgentSpec:
    """One agent of a call: its prompt, tools, tasks and models, on any framework.

    ``UnmuteAgent(spec)`` runs it on LiveKit and ``UnmuteFlows(spec, ...)`` on Pipecat. The
    call-wide objects, the state, the hooks and the telemetry, go to the adapter, and so do
    the framework's own tools, voices and services. Build it per call, after the state, since
    a model like ``RouterModelCall(state.call)`` reads the call.

    Attributes:
        instructions: The main prompt. It may hold ``{{name}}`` placeholders.
        name: The agent's name, which a ``handoff`` event shows. None uses the call's
            ``agent_id``. An agent the call is handed to sets its own.
        tasks: The checked tasks and groups. The plan holds every tool, task-only ones
            included.
        tools: The agent's tools, for an agent without tasks.
        model: The agent's model, which its steps run on too. On LiveKit a plugin goes in as
            ``LiveKitModel(plugin)``. None uses the framework's own LLM.
        summarizer: Writes the summary a ``summary`` history asks for, for a task step or a
            handoff without its own. None asks the agent's model.
        task_models: The models a task's ``model`` names, by name.
        language: The call's language, a BCP-47 tag. Confirm and reply lines say dates and
            times in it.
        timezone: Where the business is, an IANA name. Confirm and reply lines then name a
            day from today: "today, October 8".
        save_values: End each main-agent request with the current text of the State values,
            and give the agent the ``save_values`` tool for the values no task saves and the
            call does not start with.
        handoffs: The names of the agent's handoff tools, among its framework tools. The
            first reply after a handoff gives it the call leaves them out, so it cannot hand
            the call straight back. So does a greeting the model words.
    """

    instructions: str
    name: AgentName | None = None
    tasks: TaskPlan | None = None
    tools: ToolRegistry | None = None
    model: SkipValidation[ModelCall | None] = None
    summarizer: SkipValidation[ModelCall | None] = None
    task_models: SkipValidation[Mapping[str, ModelCall]] = Field(default_factory=dict)
    language: str = "en"
    timezone: str | None = None
    save_values: bool = False
    handoffs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _fits_the_tasks(self) -> Self:
        """Refuse tools beside tasks, and a task model ``task_models`` does not name.

        Raises:
            ValueError: If ``tools`` comes with ``tasks``, or a task names a model
                ``task_models`` lacks.
        """
        if self.tasks is not None and self.tools is not None:
            raise ValueError("Pass tools= or tasks=, not both: the task plan holds the tools.")
        if self.tasks is not None and (
            unknown := [
                (task.name, task.model)
                for task in self.tasks.tasks.values()
                if task.model is not None and task.model not in self.task_models
            ]
        ):
            task, model_name = unknown[0]
            raise ValueError(
                f"task '{task}' runs on the model '{model_name}', which task_models does not "
                "name. Pass it in task_models."
            )
        return self
