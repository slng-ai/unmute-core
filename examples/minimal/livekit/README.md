# minimal, LiveKit

One piece of unmute-core at a time, on a stock LiveKit Agents app. Each file is a small, complete agent with its own `AgentServer`. Speech runs on SLNG: Soniox hears the caller and Fish Audio speaks. The LLM is an OpenAI plugin, except in `router.py`, which uses the SLNG Context Router.

Where a file runs `UnmuteAgent`, the agent is an `AgentSpec`: its prompt, tools, tasks and models. The same spec runs on Pipecat with `UnmuteFlows(spec, ...)`. The call's state, hooks and telemetry, and LiveKit's own options, go to `UnmuteAgent`.

| File | What it shows |
|---|---|
| `tools.py` | A `ToolRegistry` with a local tool, the `current_datetime_tool`, and a `values_tool` that saves State values, given to `Agent(tools=function_tools(...))`. The local tool says an announce line with a call input in it and waits for the line to play. On LiveKit, a listener keeps the prompt showing the saved values. |
| `router.py` | The SLNG Context Router as the agent's LLM, with `ModelCallLLM`. |
| `telemetry.py` | Traces to Langfuse with `trace_session`, `traced_llm` in the agent's `llm_node` and `traced_tts`. The LLM is LiveKit's OpenAI plugin. |
| `greeting.py` | A fixed opening line said at once with `open_call`, a `CallHooks` tool that runs as the call starts, and an observer told how the call ended. |
| `hooks.py` | One `Hooks` object for the call, on `UnmuteAgent(call=...)`: a `PromptSuffix`, a `before_model` hook that sends `temperature=0`, a `before_tool` hook that refuses a refund, and observers of the tool and call-end events. |
| `idle.py` | Nudges for a quiet caller, then a hang-up, with `idle_nudges`. Nudge only and hang-up only are one setting each. |
| `handoff.py` | Two `UnmuteAgent`s and `hand_off` with an announce line and a history rule. |
| `tasks.py` | One task whose step saves a value, on `UnmuteAgent`. |
| `phone.py` | A SIP call answered or dialled with `SipLine`, and a `transfer_tool` that puts the caller through to a person. |
| `realtime.py` | `UnmuteAgent` on OpenAI's realtime model, with no STT, LLM or TTS: a fixed greeting the model says itself, a tool, and `save_values` in its instructions. |

## Run one

```bash
uv sync
export SLNG_API_KEY=... SLNG_WORLD_PART=eu-north OPENAI_API_KEY=...
export UNMUTE_AGENT_ID=minimal
uv run python tools.py console
```

`console` talks to your microphone. `router.py` needs `OPENAI_API_KEY`, `SLNG_API_KEY` and `SLNG_WORLD_PART` (or `SLNG_ROUTER_URL`). `telemetry.py` needs `OPENAI_API_KEY` and the `LANGFUSE_*` keys. `phone.py` needs a SIP trunk: `SIP_ADDRESS`, `SIP_USERNAME`, `SIP_PASSWORD`, `FROM_NUMBER` and `TRANSFER_NUMBER`. Use `dev` in place of `console` to connect to a LiveKit server.

The router needs an `SLNG_API_KEY` from a project on the Execution Layer. `SLNG_WORLD_PART` must be that project's region, or the router answers 401. With any other key, the router answers 403 `project_mode_forbidden`. Each request carries your own router config, which sends the model to OpenAI on your `OPENAI_API_KEY` (BYOK, bring your own key). `router.py` never uses a config stored on the router.
