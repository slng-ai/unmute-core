"""Tasks and task groups, the same on every framework.

``TasksSpec`` declares an agent's tasks. ``TaskPlan`` checks it once against the State and
the tools. Per call, ``CallTasks`` gives the main agent its delegate tools and starts a
``DelegateRun`` for each one it calls. The run hands the adapter one ``StepRun`` at a time,
and the step answers each of its tool calls. Adapters only translate these into LiveKit or
Pipecat objects.
"""

from unmute_core.model import STEP_START_CUE
from unmute_core.tasks._finish import AfterReply, FinishCheck, decided_finish
from unmute_core.tasks._opening import Opening
from unmute_core.tasks._plan import Delegate, TaskPlan, entry_history
from unmute_core.tasks._progress import GroupProgress
from unmute_core.tasks._record import RunRecord, ServedRequests, TaskEventSink
from unmute_core.tasks._run import CallTasks, DelegateRun, StepRun, caller_turns
from unmute_core.tasks._spec import (
    FINISH_TOOL_NAME,
    GO_BACK_TOOL_NAME,
    Assign,
    FinishRule,
    OpeningTool,
    TaskGroupSpec,
    TaskInput,
    TaskSpec,
    TasksSpec,
    ToolReply,
)
from unmute_core.tasks._texts import (
    EMPTY_REPLY_ATTEMPTS,
    EMPTY_REPLY_GAVE_UP,
    EMPTY_REPLY_GAVE_UP_AFTER_TOOL,
    NOT_YET_TOOL_NAME,
    REFUSED_BUSY,
    CallStartResults,
    Refused,
    ToolDeclaration,
    ToolStatus,
    empty_reply_note,
)
from unmute_core.tasks._values import SaveOutcome, TaskValueRefused
from unmute_core.tools._spec import tool_succeeded
from unmute_core.tools._spoken import spoken

__all__ = [
    "EMPTY_REPLY_ATTEMPTS",
    "EMPTY_REPLY_GAVE_UP",
    "EMPTY_REPLY_GAVE_UP_AFTER_TOOL",
    "FINISH_TOOL_NAME",
    "GO_BACK_TOOL_NAME",
    "NOT_YET_TOOL_NAME",
    "REFUSED_BUSY",
    "STEP_START_CUE",
    "AfterReply",
    "Assign",
    "CallStartResults",
    "CallTasks",
    "Delegate",
    "DelegateRun",
    "FinishCheck",
    "FinishRule",
    "GroupProgress",
    "Opening",
    "OpeningTool",
    "Refused",
    "RunRecord",
    "SaveOutcome",
    "ServedRequests",
    "StepRun",
    "TaskEventSink",
    "TaskGroupSpec",
    "TaskInput",
    "TaskPlan",
    "TaskSpec",
    "TaskValueRefused",
    "TasksSpec",
    "ToolDeclaration",
    "ToolReply",
    "ToolStatus",
    "caller_turns",
    "decided_finish",
    "empty_reply_note",
    "entry_history",
    "spoken",
    "tool_succeeded",
]
