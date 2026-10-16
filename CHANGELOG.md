# Changelog

Each release lists what changed since the release before it. "Changes that can break you" comes first: read it before you upgrade.

Before 1.0, the minor version is the breaking part. `unmute_core.require("0.13.0")` refuses 0.14.0 on purpose. Call `require` with the exact version your code was built for.

## 0.14.0

This is the first release since 0.10.0. Versions 0.11, 0.12 and 0.13 were never released, so their changes are listed here too.

It is the first release on PyPI: `uv add "unmute-core[livekit]"`. You no longer need a git source or a GitHub token.

### Changes that can break you

Dependencies:
- LiveKit Agents `>=1.8.7,<1.9`, Pipecat `>=1.12.0,<1.13`. Python `>=3.12`.
- The core needs pydantic `>=2.11`, pydantic-settings `>=2.8`, OpenTelemetry `>=1.40` and phonenumbers `>=8.13,<10`. These floors are lower than in 0.10.0, so an app's own pins fit more often.
- The `slng` extra needs openai `>=1.109,<4`. LiveKit 1.8.7 itself needs openai 3, so a LiveKit app gets it anyway. A Pipecat app may keep openai 1 or 2. openai 3 runs on `httpx2`, not `httpx`: pass `openai.Timeout` as a model call's `timeout`, and a client of the package your openai runs on as `http_client`.

One agent description on both frameworks, `AgentSpec`:
- `unmute_core.agent.AgentSpec` holds what makes an agent itself: `instructions`, `name`, `tasks` or `tools`, `model`, `summarizer`, `task_models`, `language`, `timezone`, `save_values` and `handoffs`. `UnmuteAgent(spec)` runs it on LiveKit and `UnmuteFlows(spec, ...)` on Pipecat. A handoff takes the next agent's spec on both.
- The adapters keep what every agent of a call shares (`state`, `call`, `telemetry`) and the framework's own objects (`native_tools`, `then_targets`, `tts`, Pipecat's `llm`, `context` and `aggregators`, LiveKit's `Agent` options). These have the same names on both.
- Build the spec per call, after the state, since a model like `RouterModelCall(state.call)` reads the call.
- A spec's `model`, `summarizer` and `task_models` are `ModelCall`s. On LiveKit a plugin goes in as `LiveKitModel(plugin)`.
- An agent's name is its spec's `name`, else the call's `agent_id`. It replaces LiveKit's `id=`, which came from the class name when left out, and Pipecat's `name="main"` and the handoff tool's name.
- On Pipecat, `UnmuteFlows` builds the call's `CallTasks` itself, as `UnmuteAgent` does. It uses `call.state` when no state is given, and gives the steps the call-start results. Read them with `flows.tasks`, and hear every agent's task events with `flows.listen(sink)`.
- On Pipecat, `save_values` and `summarizer` now belong to each agent, as on LiveKit. The agent a call is handed to no longer gets the first agent's.
- `AgentSpec` refuses `tools=` beside `tasks=`, and a task model `task_models` does not name, on both frameworks.
- Error texts that named an old argument name the new one: `AgentSpec(handoffs=...)`, `hand_off(spec) with a model`, and "An agent with tasks needs state=CallState(...) or call=CallHooks(...)."

| Before | Now |
|---|---|
| `UnmuteAgent(instructions=, tasks=, model=, id=, save_values=, handoffs=, state=, call=, ...)` | `UnmuteAgent(AgentSpec(instructions=, tasks=, model=, name=, save_values=, handoffs=), state=, call=, ...)` |
| `UnmuteAgent(model=openai.LLM(...))` | `AgentSpec(model=LiveKitModel(openai.LLM(...)))` |
| `UnmuteFlows(CallTasks(plan, state, language=), instructions=, name=, llm=, ...)` | `UnmuteFlows(AgentSpec(instructions=, tasks=plan, name=, language=), state=state, llm=, ...)` |
| `UnmuteFlows(CallTasks.empty(state, tools=registry), ...)` | `UnmuteFlows(AgentSpec(instructions=, tools=registry), state=state, ...)` |
| `flows.hand_off(params, tasks, instructions=, name=, model=, handoffs=, native_tools=, tts=, ...)` | `flows.hand_off(params, AgentSpec(instructions=, tasks=, name=, model=, handoffs=), native_tools=, tts=, ...)` |
| `tasks.listen(recorder.task_event)` on Pipecat | `flows.listen(recorder.task_event)` |

One call state. The call facts are gone:
- Every `CallState` holds `call`, a `CallInfo`: what the call carries. Read `state.call.calling_number` in Python and `{{call.calling_number}}` in prompts, tasks, greetings, announce lines and locked arguments. Only the library writes it. `state.save({"call": ...})` is refused.
- `call` is a reserved name. A State, a `StateSchema` variable, an alias, a `confirm`, `dependencies` or `call_start` rule, a task's assign or a prefetch assign that names `call` fails when it is built. Rename the value.
- `session_id` is one id per call. The library makes it when the state is built, the same way on LiveKit and Pipecat. It stays the same on every turn, task and handoff. A dispatch may set it.
- `agent_id` is required. Pass it to `CallInfo`, `call_info`, `sip_call_info` or `telephony_call_info`, or set `UNMUTE_AGENT_ID`. With neither, `CallState(State)` raises `SettingsError`. Add `AgentSettings` to your `check_settings` call to fail the deploy instead.
- Both ids are 1 to 128 printable ASCII characters with no comma, whitespace, `{`, `}` or `|`. A bad id fails when the call is built.
- `RouterModelCall` takes the call, not `RouterHeaders`: `RouterModelCall(state.call, ...)`. Pass `llm_config_id=` on its own. Anything other than a `CallInfo` raises `TypeError`.
- `unmute_core.slng.tracking_ids(state.call)` gives SLNG speech-to-text and text-to-speech the same two ids: `slng.STT(..., **tracking_ids(state.call))` on LiveKit, `SlngSTTService(..., **tracking_ids(state.call))` on Pipecat. Pipecat needs `pipecat-slng` 0.7.0 or newer.
- `call_facts=` is gone from `UnmuteAgent`, `CallTasks`, `CallTasks.empty`, `function_tools`, `tools_schema` and `CallHooks`. Pass the state instead. `Prefetch.run(state)` and `run_background(state)` take no facts. `UnmuteAgent(state=None)` takes the state of its `CallHooks`. `CallHooks(state=None)` builds `CallState()`.
- `CallState(model)` defaults to `NoValues`, so `CallState()` serves an agent that keeps no values. `NoValues` moved to `unmute_core.state`.
- `{{@name}}` is gone. It fails when the tool is built, and the error names the new spelling.
- `Prefetch(seed_env=...)` is gone. Simulate a caller with `UNMUTE_CALL_START='{"call": {"direction": "inbound", "calling_number": "+34612345678"}}'`, or a `call` object in the dispatch's `call_start`. On a web call it applies. On a phone call the line wins, and the dispatch fills only what the line left empty.
- The webhook envelope's `context` holds `state.call` with the new names, and `trigger_event` and `transcript` on a tool on an event.
- A number from the line must be a valid phone number. One that is not, or a withheld caller ID, reads as None.
- The dev record's state origin `fact` now means the library saving `state.call`. A `CallValue` copy is a `prefetch`.

| Before | Now |
|---|---|
| `prefetch.call_facts(direction, ...)` | `state.call_info(direction, ...)`, a `CallInfo` |
| `livekit.sip_call_facts(..., room_name=)` | `livekit.sip_call_info(..., agent_id=, session_id=)`. The room is not kept |
| `pipecat.telephony_call_facts(call_data)` | `pipecat.telephony_call_info(call_data, agent_id=, session_id=)` |
| `prefetch.call_phone_region(facts)` | `state.call_phone_region(state.call)` |
| `CallFact("caller", "from_number", [...])` | `CallValue("caller", "calling_number", [...])` |
| `from_number`, `to_number` | `calling_number`, `called_number` |
| `{{@phone_number}}` | `{{call.remote_number}}` |
| `{{@call_id}}` | `{{call.carrier_call_id}}`, or `{{call.session_id}}` |
| `{{@call_end_reason}}` | `{{call.end_reason}}` |
| `{{@first_user_message}}` | `{{call.first_user_message}}` |
| `{{@transcript}}` | `Transcript()` |
| `{{@trigger_event}}` | the webhook envelope's `trigger_event` |
| `stream_id` | not kept. Pipecat's `runner_args` has it |

Errors and input checks:
- Value objects, configs and specs are Pydantic classes now. A wrong value for an argument raises `pydantic.ValidationError`, which is a `ValueError`. Some of these raised `TypeError`, `KeyError` or `AttributeError` before, for example `CallLimit("x")`, `ToolSpec(5, ...)` and `RouterHeaders(agent_id=5)`. Catch `ValueError`.
- Pydantic's lax mode applies. Some text is now read as its type: `"5"` as 5, `"no"` as False, `"yes"` as True. A few values that were accepted are now refused, like `Transcript(max_messages=2.5)`.
- A built-in constraint error names the public class and the field it checked.
- A tool schema's root must be a plain object. `nullable` is refused.
- A tool call with an argument a closed schema does not name is refused, and the refusal names the argument.
- A webhook URL is checked when the tool is built. A GET argument that reuses a URL query name is refused, and so is an `Authorization` header next to `auth`.
- Pipecat refuses two tools with one name on one agent.
- A task input of kind `tool` that names no prefetch `Lookup` fails at build, when the agent has the call's hooks.
- `RouterHeaders` refuses an id that is not printable ASCII when it is built. Before, every request failed.
- `env_value(name, type)` reads the variable with exactly that name. A lowercase variable no longer answers an uppercase name.
- The `require()` error names the version range to require in `pyproject.toml`, not a git tag.
- A constraint error also names the field for an argument given by position: `PromptSuffix(5)` names `text`, not `0`.
- `CallState(model)` raises `TypeError` when `model` is not a Pydantic model class.
- `ToolSpec` refuses an executor with no async `run(arguments, state)` when it is built.
- `transfer_tool` refuses a `PhoneLine` that is not hashable and weak-referenceable, like a `@dataclass(slots=True)` class.
- `collect()` raises `TypeError` for an event that is not `TextDelta`, `ToolCall` or `Usage`. Before, it was read as usage.
- `OpenAIModelCall` and `RouterModelCall` check `model`, `base_url` (http or https) and `timeout` (seconds above zero, or an `openai.Timeout`). `OpenAIModelCall(timeout=0)` used the default timeout. It is refused now.
- `FallbackModelCall` refuses an empty `calls`, a call with no `stream`, and a wait below zero. `Otlp(endpoint=)` must be an http or https URL.
- `InterruptionRule` checks `backchannels` against `keep`, `drop` and `stop`.
- LiveKit refuses two tools with one name on one agent at build, as Pipecat does. A task named like another tool of the agent, such as `end_call`, is refused there.

What the model reads:
- A whole float reads as an integer: `3`, not `3.0`. This holds in prompts, tool results, webhook URL placeholders and `CallState.snapshot()`. A whole `"{{x}}"` locked argument still sends the float.
- Text values are trimmed: `str`, `Id`, `Literal` choices and shape text. A declared choice with spaces, like `" small"`, is saved as declared.
- The notes header for tool results has a new sentence: use them, but never read them aloud or share the raw data.
- `OpenAIModelCall` folds every system and developer message into one at the front, for a Gemini model.
- On Pipecat, a step whose task has an announce line gets the step cue.
- On LiveKit 1.8.7, a running tool is never cancelled. When the reply that called it is cut off for long, the model reads that the tool is still running, and the result comes in a later reply.
- A value refusal no longer shows a regex. NaN and infinity are refused.
- A saved value and a prefetch value reach a native model whole. Only `RouterModelCall` cuts a template variable to the router's 4000 characters, with a warning. Before, every path cut a saved value to 4000 and a prefetch value to 512. `unmute_core.prefetch.VALUE_MAX` is gone.
- A placeholder key longer than 64 characters is refused at build only next to a model call that names `variable_key_max`, like `RouterModelCall`. A `FallbackModelCall` takes the smallest limit of its calls. A native model takes any length.
- `{{rtv_<name>}}` no longer reads the value `<name>`. It reads a value named `rtv_<name>`, like any placeholder. `unmute_core.model.RUNTIME_KEY_PREFIX` is gone. Rename the placeholders, or rewrite them in a `prompt` hook.
- Pipecat: a worded greeting and a model-worded idle nudge go to the model as developer messages, not system messages.

Calls and timing:
- A TTS error no longer ends the call, on both frameworks. The library logs it and the call goes on. `call.failed` observers and `wrap_failure` hooks no longer run for `tts`. `call.tts_failed` observers hear it.
- A `wrap_failure` hook that returns without calling its handler keeps the call going. The call is no longer failed, `call.recovered` observers hear it, and its end reason is the usual one. Before, the call stayed failed and ended as `error`.
- A tool result with `"error": null` or an empty `error` is a success. Before, any `error` key made it a failure.
- `FallbackModelCall` stops the chain on the error codes each call names in `stop_codes`. `STOP_CODES` is empty. `RouterModelCall` names `project_mode_forbidden`, so a chain with a router stops on it as before. A chain of other calls no longer stops on that code.
- LiveKit `sip_call_info` reads the direction from `direction=`, then the metadata's `direction`, then the SIP participant: one with a dispatch rule (`sip.ruleID`) is inbound. Otherwise `phone_number` in the metadata means outbound. A metadata `direction` that contradicts `phone_number` is no longer refused.
- Pipecat `end_call_tool` has no `language`. With no `goodbye`, it says `GOODBYE_LINE`, in English. Pass your own `goodbye` in the call's language.
- End reasons are `caller_hung_up`, `agent_ended`, `transferred`, `max_duration`, `idle`, `error` and `ended`, the same on both frameworks.
- A model error after the reply started ends that reply. The call goes on.
- A failure after `call.end_requested(...)` or `call.caller_left()` plays no failure line.
- LiveKit: a greeting holds the caller's audio for at most `GREETING_HOLD_SECONDS` (20 s). Set it with `open_call(max_hold_s=)`.
- A worded greeting on a realtime model waits for the prefetch lookups and then for the instructions to be sent again, at most `SETTLE_WAIT_SECONDS` (5 s), before it speaks.
- LiveKit no longer retries a `FallbackModelCall` on top of its own chain.
- `Telemetry.close()` waits 10 s by default.
- `telephony.E164` reads a number of 2 to 15 digits after the plus as a number.
- No model warm-up. `ModelCall` has only `stream`. `ModelCall.warm`, `RouterModelCall.warm` and `FallbackModelCall.warm` are gone, and the adapters no longer warm a model call when the session starts. The client closed the warm connection before the caller's first turn, so the warm-up saved no time. A `warm` method on your own class is now ignored.

Telemetry:
- Pipecat: `gen_ai.request.model` on an `llm` span is the model the model call names, not `TelemetryObserver(model=)`.
- Each `llm` span names the model the request went to. `FallbackModelCall(sticky=True)` spans carry the model that answered as `gen_ai.response.model`.

Telemetry and tasks:
- `DEFAULT_ATTRIBUTES` holds `unmute.call.framework` and `unmute.call.origin`, so the default `MetadataOnly` filter exports them.
- `CallTasks.emit` takes fixed keywords: `task`, `group`, `run_id`, `step` and `saved`.
- `Langfuse.from_env`, `Logfire.from_env`, `Coval.from_env` and their `from_settings` take no extra options.
- `TURN_WINDOWS` is a read-only mapping.
- `PromptSuffix` is frozen.
- The task, Literal and State caps of 0.10.0 are gone: a plan may hold any number of tasks, choices and values. The reserved task names are now only `finish`, `go_back_to_step` and `summary`.

Typed tool outputs:
- Every tool names an output model. `local_tool` reads it from the function's return annotation (`-> Order`). A function that returns a dict passes `output=Order`. A tool with neither raises `TypeError` when it is built.
- `ToolSpec` takes the output model after `parameters`: `ToolSpec(name, description, parameters, output, executor, ...)`.
- A result that succeeded and does not fit its output model fails the call. The model reads which field is wrong and "Do not claim success". A result with `error`, `refused` or `"ok": false` is not checked.
- The model reads the checked result, `output.model_dump(mode="json")`. Defaults are filled in, and fields the model does not declare are dropped unless it sets `extra="allow"`. The built-in phone tool now sends `call_type` and `reason` as `null` when they have no value, and a transfer sends `ended` and `reason` too.
- Every `local_tool` parameter needs a type hint. `*args` and `**kwargs` are refused. `local_tool(parameters=)` is removed.
- A function whose only parameter is a Pydantic model takes that model as its whole input. Before, the model was one argument.
- `webhook_tool(name, description, url, *, input, output, ...)` takes the input and output models. Its `parameters` argument is removed. A webhook that does not wait for its answer must use `output=Accepted`.
- `hosted_tool`, `HostedHandler`, `typed_tool` and `TypedHandler` are removed: `local_tool` covers them. A function whose only parameter is a Pydantic model takes it as its whole input.

Spoken lines:
- A `{{result.<field>}}` in a step's `FinishRule.say`, `ToolReply` or `OpeningTool.say` is now said by the output field's JSON Schema `format`, in the call's language, as confirm lines already were. A `date` reads "tomorrow, October 16". A `Literal` or `Enum` value is said as words.
- `TaskPlan` checks each result field a step's lines, `when` and `success` read against the tool's output model. A field the tool does not return fails when the plan is built.

- `spoken` lives in `unmute_core.tools`. `unmute_core.tasks.spoken` still works.

Test doubles:
- A LiveKit end tool calls `session.once`. A Pipecat ending tool reads `params.tool_call_id`, `params.llm` and `params.pipeline_worker`. A fake session or params object needs them.

Step tools:
- A step's tool that does not end the step now saves its registry `assign` rows as soon as it returns, as it does on the main agent. Before, a step ignored them and saved only its own `assign`, when it ended. Leave a row off the registry when a step must not save it early.

Tool replies:
- A `when` or `success` value the field can never hold now fails at build time. Before, the rule silently never matched.
- `ToolReply.tool` is optional. A `ToolReply` on a `ToolRegistry` or a `TaskSpec` without it is refused.
- A required tool argument the model sends as blank text is missing now, like an absent one. Before, a free-text argument sent as `""` passed.

### Added

- `DevObserver` on LiveKit, for dev records.
- `UnmuteAgent(model=, tts=)` and Pipecat `hand_off(..., model=, tts=)`: each agent can have its own model and voice. `handoffs=` names an agent's handoff tools.
- `TaskGroupSpec(then="transfer", then_target=...)`.
- Gemini thought signatures on replayed tool calls, on both adapters.
- `FallbackModelCall(stop_statuses=, stop_codes=, same_model_retries=, first_event_timeout_s=, sticky=)` and `on_fail_over`.
- `SipLine.dial(participant_name=, display_name=, metadata=, attributes=, krisp=, audio_wait_s=)`.
- Pipecat: `interruption("stop", min_speech_s=)`, `GreetingHold`, `attach_call(..., transport=)`, `open_call(..., transport=)`, `dial_out(..., call=)` and `voicemail_tool(...)`.
- `unmute_core.call.GOODBYE_LINE`: the goodbye `end_call_tool` says when the reply has no words.
- `filled_llm` on LiveKit and `fill_prompt` on Pipecat: a plain agent's prompt is filled from the call state, and fresh after each save, with no `Telemetry`.
- A call whose `prompt`, `before_model`, `wrap_model` or `after_model` hooks would not run logs one warning that names the fix.
- `shape_chat_ctx` on LiveKit: a plain agent's handoff history, as `UnmuteAgent.hand_off` shapes it.
- `UnmuteAgent(**agent_options)`: LiveKit's own `Agent` options pass through.
- `CallTasks(gave_up_line=, gave_up_after_tool_line=)`, `IdleNudges(nudge_instruction=)`, `Telemetry(gateways=)` and `save_call_start(env=)`.
- `unmute_core.turn.word_count`. Pipecat's `keep` interruption counts Chinese, Japanese and Thai words with it.
- `variable_key_max` and `stop_codes` are optional members of a model call. `variable_key_max(*calls)` and `FallbackModelCall` read them.
- `check_tool_names`, the duplicate tool check both adapters run.
- Pipecat: `UnmuteFlows(summarizer=)` also answers the finish check when the call's LLM cannot be asked. `TelemetryObserver` warns when metrics are off.
- LiveKit: `interruption("keep")` and `("drop")` warn when LiveKit's adaptive detector has no key.
- `call_type` (inbound or outbound) in the caller-number tool's result.
- State value names with a leading underscore or a `model_` prefix build. Prompts and saves keep the real name.
- `RouterConfig` takes every inline router setting, `slice_assistant_max_chars` and `prewarm_cache_enabled` among them.
- Coval: `correlate_coval` on both adapters.
- LiveKit: a request that a `ModelCall` answers through `UnmuteAgent` emits `LLMMetrics`, so `metrics_collected` and `session.usage` carry its tokens.
- `save_values=True` shows the values block whenever State has values. It offers the save tool only when the agent can save a value.
- `ToolRegistry(tools, state=..., assign=..., replies=...)`: an agent without tasks saves a tool's result into State with `Assign` rows, and speaks a `ToolReply` line in place of a model reply. `function_tools`, `tools_schema`, `UnmuteAgent` and `UnmuteFlows` use them. The registry checks every row when it is built.
- `ToolReply.state`: a reply can depend on State values. `ToolReply.when` is optional, so a reply with neither is the fixed reply.
- `Assign(variable, path=())` saves the whole result into a State value typed as the output model.
- `language` and `timezone` on `function_tools` and `tools_schema`, for the reply lines. `UnmuteAgent` keeps them without tasks too.
- `SpokenLines`: the per-call lines code speaks, in the call's language.
- `Done` and `Accepted`: the output models of a tool that only runs, and of a webhook that does not wait.
- The `date-time` spoken form: the day, then the time, in the order CLDR gives.
- Spoken forms follow `$ref` and `anyOf`, so a nested model, an `Enum` class or an optional field is said by its format, in confirm lines too.
- `StateSchema.shape(name)` returns a declared shape's model, so a tool defined as data can name it as its output.
- The tool event `tool.save_refused`, when the State refuses a registry save.
- Spoken lines (`confirm`, `say`, `ToolReply`) read a field of a shape in the State by its path, like `{{account.amount_words}}`. A path that does not exist fails when the agent is built.
- Two value types in `unmute_core.state` with their own spoken form: `Spelled`, a code said a character at a time in groups ("K P, 4 8 2, 9 1 3"), and `BirthDate`, a date said with its year and no weekday. Their JSON Schema formats are `spelled` and `birth-date`.
- A tool's `hidden` fields, like `local_tool(..., hidden=("read_back",))`, are left out of what the model reads of a result that succeeded. `assign` rows and spoken lines still read them, so code can say a line from a result the model never sees.
- `Greeting.check(prefetch, call_start)` accepts the values a call starts with.
- `tracking_ids` returns a `TrackingIds` typed dict, so `**tracking_ids(state.call)` type-checks against the SLNG speech plugins.
- A tool carries its own replies: `local_tool(fn, replies=[ToolReply(when=..., say=...)])`. They play wherever the tool runs, after the place's own replies. A tool's own reply leaves out `tool`.
- A `when` value on a `Literal`, `bool` or `Enum` result field must be one the field holds. A typo fails when the registry or the plan is built, with the values it can hold. This holds for `ToolReply.when` and `FinishRule.success`.
- Code that speaks a reply line tells the sinks and hooks `tool.replied`.
- A tool can check a business rule before it runs: `local_tool(fn, precheck=check, replies=[...])`. `check` is a plain function that takes some of the tool's parameters and returns an instance of the output model, or None to go ahead. An outcome is the tool's result: the tool does not run, no announce plays, and its replies say it. On a closing tool with `confirm`, the precheck runs before the read-back, so a wrong value is never read back and the refusal costs no model request.
- `hand_off(opens_with=OpeningTool(...))` on LiveKit and Pipecat runs a tool of the next agent in code as it takes the call. The registry saves what the result names. When a reply line fills, that line is the agent's first reply, with no model request. When none fills, the model replies as before. A realtime model refuses it.
- `ArgumentsMissing`, a `ToolRefused`: a required argument the model left out or sent blank. The model reads which ones are missing and asks the caller. `replies_to(result, spoke=...)` in `unmute_core.tools` holds the rule: no model reply to that refusal when the reply that made the call already spoke.

### Fixed

- A locked transcript keeps the newest turns within 32 KiB, counted as a request sends them.
- A blank tool description falls back to the tool name.
- LiveKit: a tool beside an end tool in one reply gets no second reply. The same holds on a Pipecat realtime model.
- LiveKit: a tool line the tool waits for is not interruptible.
- The idle sequence goes on after a nudge the model words.
- A request field the SLNG Context Router drops is sent with one warning, not refused. `extra_body` and `extra_query` are not dropped.
- Pipecat hides its async-tool text only for the library's own tools that run on. Your own tools keep it.
- LiveKit: `open_call` returns None when the caller leaves during the `before_call` hooks or the prefetch.
- LiveKit retries a model call that timed out or lost its connection on openai 3, as it did on openai 2.
- `Telemetry.close()` reports its flush on OpenTelemetry releases before 1.42.
- A knowledge base with fewer passages than `top_k` answers lookups.
- CI runs the tests with every dependency at its lowest allowed version, with and without LiveKit.
- `function_tools` and `tools_schema` drain their webhooks at the end of the call, as `UnmuteAgent` and `UnmuteFlows` do.
- `import unmute_core.knowledge` without the extra names the extra to install.
- A read-back asks again when a tool's locked values changed since the caller heard it. Before, a yes ran the tool with the new locked values, like a changed report, that the caller never heard.
- A looked-up value in `CallState(dependencies=...)` that no step confirms no longer makes every prompt raise `KeyError`. It reads empty everywhere until its sources are confirmed.
- A step that opens on a tool with `confirm` lines and no `arguments` now says its read-back. Before, it opened with a model reply.
- On Pipecat, a line code says right after a tool call joins the context as soon as it has played. Before, Pipecat's TTS held it until the caller's next words, so the model read the caller's answer before the question it answered.
- A tool the model calls gets the transcript in a `Transcript` locked argument, as an event tool does. Before, it got an empty list. `CallHooks.transcript()` reads it.
- On Pipecat with a realtime model, a tool result waits for its reply to end, so a later call of the same reply can still end the call. Before, the model was asked for another reply first.
- On Pipecat, a line played as the call ends, like a failure's apology, has its `tts` span in the last turn. Before, it opened a turn that never closed.
- The model sees a plainer copy of each tool schema. An optional value whose default is `null` has no null branch, `format` and `pattern` are words in the description, and local `$ref`s are inlined. Gemini 3.5 Flash-Lite answered the old schemas with `MALFORMED_FUNCTION_CALL`, an empty reply, about three times as often. The tool still checks its arguments against the full schema, so a model that sends `null` is still accepted.
- A model call with a blank required argument no longer costs a second model request when its reply already asked the caller. Before, the caller could hear the same question twice.
- Pipecat: a task step in `record` mode runs its finish check, and a `summary` history gets its summary, when the call's LLM is a `GoogleLLMService`. Before, both were skipped, unlike on LiveKit.
- Pipecat realtime: a tool that ends the call waits for the reply to end first. Before, the session could close before the reply's later tool calls arrived, and they were lost.
- Pipecat dev records: a caller who speaks as the agent's line ends keeps one input, and the reply links to it. Before, the dev page showed an empty second caller line and linked the reply to that.

### Examples

- kestrel: a future incident day is the tool's rule, not the argument's. `note_facts` returns `status: "future_day"`, and its own reply asks again. The turn costs one model request instead of up to three.
- No example validator carries caller text now. kestrel's "KP and six digits", faro's "Spanish numbers only" and wasla's ten-day window are prechecks of `lookup_policy`, `check_number`, `record_promise` and `schedule_callback`. Each outcome is a `Literal` value its tool's reply says, before the read-back.
