# minimal, Pipecat

One piece of unmute-core at a time, on a stock Pipecat runner app. Each file is a small, complete bot with its own `bot()` function. Speech runs on SLNG: Soniox hears the caller and Fish Audio speaks. The LLM is an OpenAI service, except in `router.py`, which uses the SLNG Context Router.

Where a file runs `UnmuteFlows`, the agent is an `AgentSpec`: its prompt, tools, tasks and models. The same spec runs on LiveKit with `UnmuteAgent(spec)`. The call's state, hooks and telemetry, and Pipecat's own services, go to `UnmuteFlows`.

| File | What it shows |
|---|---|
| `tools.py` | A `ToolRegistry` with a local tool, the `current_datetime_tool`, and a `values_tool` that saves State values, given to the `LLMContext` with `tools_schema`. The local tool says an announce line with a call input in it and waits for the line to play. `UnmuteFlows(AgentSpec(..., save_values=True))` adds the values to the prompt for you. |
| `router.py` | The SLNG Context Router as the bot's LLM, with `ModelCallLLMService`. |
| `telemetry.py` | Traces to Langfuse with `TelemetryObserver`. |
| `greeting.py` | A fixed opening line queued with `open_call` and held caller audio with `GreetingHold`, a `CallHooks` tool that runs as the call starts, and an observer told how the call ended. |
| `hooks.py` | One `Hooks` object for the call, on a plain pipeline with `attach_call(llm=...)` and `tools_schema(hooks=call)`: a `PromptSuffix`, a `before_model` hook that sends `temperature=0`, a `before_tool` hook that refuses a refund, and observers of the tool and call-end events. |
| `idle.py` | Nudges for a quiet caller, then a hang-up, with `IdleNudgeObserver`. Nudge only and hang-up only are one setting each. |
| `handoff.py` | Two `AgentSpec`s: `UnmuteFlows` runs the first, and a native handoff tool calls `flows.hand_off` with the second. Also `end_call_tool`. |
| `tasks.py` | One task whose step saves a value, on `UnmuteFlows`. |
| `realtime.py` | `UnmuteFlows` on OpenAI's realtime service, with no STT or TTS: `open_call(llm=...)` has the model say the fixed greeting, a tool, and `save_values` in its instructions. |

## Run one

```bash
uv sync
export SLNG_API_KEY=... SLNG_WORLD_PART=eu-north OPENAI_API_KEY=...
export UNMUTE_AGENT_ID=minimal
uv run python tools.py -t webrtc
```

Then open http://localhost:7860 and talk. `router.py` needs `OPENAI_API_KEY`, `SLNG_API_KEY` and `SLNG_WORLD_PART` (or `SLNG_ROUTER_URL`). `telemetry.py` needs `OPENAI_API_KEY` and the `LANGFUSE_*` keys.

The router needs an `SLNG_API_KEY` from a project on the Execution Layer. `SLNG_WORLD_PART` must be that project's region, or the router answers 401. With any other key, the router answers 403 `project_mode_forbidden`. Each request carries your own router config, which sends the model to OpenAI on your `OPENAI_API_KEY` (BYOK, bring your own key). `router.py` never uses a config stored on the router.
