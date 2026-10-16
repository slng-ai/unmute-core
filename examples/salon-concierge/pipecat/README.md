# salon-concierge

Generated Pipecat voice agent, compiled by `unmute` from a declarative spec. Do
not edit by hand — change the source spec and recompile.

## Quickstart

```sh
cp .env.example .env                      # then fill in your keys

uv run bot.py -t webrtc                   # web: open the URL it prints

```

Pipecat logs default to `INFO`. Set `UNMUTE_LOG_LEVEL=DEBUG` only in a
controlled environment because logs can contain caller speech, tool data, and
phone details.

## Streaming conversation and measurements

`unmute dev` shows words and data as their sources provide them. Final caller
words become normal text immediately. Agent words are labelled generated;
visible text does not mean every word has played as audio. Ending a call keeps
the conversation and marks unfinished work incomplete. Latest returns to the
live end after you scroll back.

Each observed SDK model call has a number, with its first-response and full
request duration visible beside it. TTS first-audio time and tool duration also
stay visible. Debug details holds secondary timings and source metadata. See
[Reading the latency numbers](https://unmute.ai/optimization/latency) for definitions. They can overlap and must not be added to obtain reply
latency. Reply latency ends at the first runtime audio and excludes delivery to
the browser. A tool's returned state does not prove business success. A call into a task, or a
handoff to another agent, gets a row of its own and no duration, because it
hands control over rather than returning a result; that row is what accounts
for the model call following it.

Only captured measurements are shown. Missing and pending values have no
placeholder. A real zero is `0ms`, a
positive value below one millisecond is `<1ms`, and other values use
milliseconds or seconds. Counts describe observed SDK calls, including explicit
task retries, rather than hidden provider retries. Missing source coverage or
lost event history labels the count as observed. Values with no proven owner remain
unassigned; they are never attached to the newest reply by timing or name.

Pipecat reports service first-response and processing durations separately.
Its native audio-latency summaries have no proven response ID and remain
unassigned. Call first speech uses the native observer session-start boundary.

**Where a reply's time went.** Beside that unassigned reply latency the page
shows the reply broken into the parts that make it up, in time order, each with
what it cost and who owns it. The parts account for the whole wait, so they add
up to the total, and the owner is one of four kinds:

- a **service** you bound, such as the transcriber or the model;
- a **setting** you wrote, such as the silence window before a turn ends;
- the **bot** itself, meaning this project's own code between the frameworks;
- the **pipeline**, meaning time inside the framework that belongs to nobody
  above, which is what makes a gap visible instead of silently missing.

The line above the parts says what the interval was measured from: the caller
falling silent, or the client connecting, which is how a greeting is kept from
being compared with a reply. This breakdown carries no reply identity, so it
sits with the unassigned values rather than being matched to a reply by timing.

**A tool that produced no result says why.** Its row reads `failed` with the
kind of error the handler raised, or `timed_out` when it ran past its deadline,
and a call that ends badly names the service that stopped it. The messages
themselves stay in `build/pipecat/dev.log`: an error's text quotes whatever
was being worked on, and this feed carries no prompts, tool arguments or
results.

The event feed reconnects separately from audio. Retained events can replay;
a gap stays marked until the next call. Switching Conversation and Logs keeps
the same microphone connection.

The generated `utils/dev_metrics.py` is inert unless `UNMUTE_DEV_METRICS` is enabled,
as `unmute dev` does. Its identified v2 snapshots use `UNMUTE_METRIC ` framing.
Raw values use seconds. To inspect measurements:

```sh
grep '"kind":"measurement"' build/pipecat/dev.log
```

The local `dev.log` also contains transcript snapshots. The normalized dev
records exclude audio, reasoning, prompts and tool arguments/results. Ordinary
SDK output can contain other data. This path adds no exporter; configured
provider calls and optional tracing keep their own existing behavior.

## Agent handoffs

A handoff with `announce:` gives the exact sentence spoken once by
the active agent, after any brief the handoff carries has been validated, so a
handoff refused for a bad brief never speaks it. Pipecat waits for that speech
to stop before it activates the receiving worker. Without `announce`, the
handoff stays silent.

The opening greeting belongs to call start only. A later handoff back to the
entry agent continues from the shared context and never repeats it.

## Facts resolved before the greeting

This agent resolves some values once per call, before it speaks, so the model
never spends a turn discovering them. Each entry below runs in the order
`agent.yaml` lists it.

| Entry | Reads | Lands in |
|---|---|---|
| `today` | the clock, in `Europe/Madrid` | `booking_date, booking_weekday, salon_local_time` |
| `caller` | the call's own `from_number`, on an inbound call only | `customer_phone` |

**The budget.** The whole block has 2.0 seconds, which you will find in the
code as `_PREFETCH_BUDGET_S`. Past that it gives up, the
values keep their declared defaults, and the call greets on time. Nothing here can
fail a call: a lookup that times out or raises is logged and stepped over.

**An entry can do nothing, and that is normal.** An entry whose inputs are empty
is skipped. A call that carries no caller ID skips every entry that reads one,
and every entry downstream of those, so the agent behaves exactly as it would
have without this block at all. Compiling for a target whose route supplies no
call facts prints a warning naming each entry it will skip there.

**A caller's number is best effort, even where the route supplies one.** `caller` can arrive empty on a
perfectly ordinary call. A caller withholding their number is ordinary rather than
exceptional, and it does not arrive as nothing: it arrives as the word `anonymous`,
or as digits that look like a number and are not one. All three are treated as
absent, the entry is skipped, and the log says so.

The other way it arrives empty is configuration. On a route where the number
rides markup you pasted into your carrier's console, a console object written
before this agent needed the number simply does not carry it. Nothing here
checks that for you: reading your console back would need carrier credentials
this build does not ask for. The telephony section of this file says exactly
what to paste.

**The clock is read once, at session start, in `Europe/Madrid`.** Not the
container's clock, which is UTC and would name the wrong day for anybody who is
not on it. Each entry above names its own zone.

**A call that crosses midnight keeps the day it started on.** The value is read
once and never refreshed, so a call beginning at 23:55 still says that day at
00:10. This is deliberate: a value that changed underneath a conversation would
be worse, because the caller and the agent would stop agreeing about what
"tomorrow" means halfway through. If a call really can outlast midnight and the
date has to follow, read it in a tool instead of pre-fetching it.

**A pre-fetched value can need confirming.** `customer_phone` (confirmed by `verify_customer`) arrive
filled but not yet agreed to. Until the naming step has heard the caller agree,
such a value appears in no prompt except that step's own, and any tool that
reads it is refused. So the agent reads the value back and asks for a yes
rather than acting on it, which is what stops somebody ringing from a friend's
phone being treated as the account holder.

**What you see in the log.** One line per entry, naming the entry and the values:

```text
prefetch today: resolved booking_date, booking_weekday, salon_local_time
prefetch caller: skipped, ...
```

A value is never written to the log, only its name, which matters when it is
something like a phone number.


## Tool-call failures

The generated direct-tool boundary keeps every call terminal. If the provider
sends an argument outside the declared input schema, the tool does not run and
the model receives one corrective result naming the allowed fields. If a
handler fails before returning, the full exception stays in the log and the
model receives one safe failure result. It must not claim success from either.

## What the caller cannot talk over

`call.py` passes `user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()]` to the context aggregator.

The greeting is protected because this is a phone route, and a phone leg has no
echo cancellation. A caller on speakerphone sends your agent's own greeting back
through their microphone; the transcriber reports it as caller speech; the agent
interrupts itself about a second into the call. Worse than the truncated
greeting, that garbled turn stays in the model's context for the rest of the
conversation, so later answers are built on top of it. A real call before this
default existed opened with the model's first user turn reading
`hi you've reached`, which is the agent's own line.

Barge-in is still on for every turn after the opening line. To change that, set
`conversation.interruption.protect` in the package: name `tool_calls` as well to
keep a cough from abandoning a booking mid-write, or `[]` to ask for no
protection at all and accept the echo.


## Tools that speak first

A tool with `announce:` speaks that exact sentence as it starts, so the caller
is not waiting in silence while the call is in flight. The line is queued, not
awaited: the tool's own work starts straight away and the caller hears the
answer no later than they would without the line. The argument check runs
first, so a call the model got wrong stays silent. If the caller speaks over
the line, the tool's `interruption:` value decides what happens.

You will see the line in two shapes in `agents.py`, because a tool listed on an
agent and the same tool listed on a task are emitted differently. An agent tool
holds `FunctionCallParams` and pushes the frame through `params.llm`; a task tool
is a flows handler and queues it through `flow_manager.worker`. Same frame, same
contract, and nothing waits for playout in either.

## Turn taking

Two numbers decide how long a caller waits. **Both are spelled `stop_secs` in
`call.py` and they are different fields**, so grep finds two and only this tells
you which is which.

**The floor is 0.2s**, in `VADParams(stop_secs=0.2)`. The VAD waits that
much silence before speech counts as stopped. It comes from the
package's `endpointing_delay`.

**The ceiling is 1.2s**, in `SmartTurnParams(stop_secs=1.2)` on the
end-of-turn analyzer. That is the longest the turn can run before closing
regardless, and it comes from `pace: snappy`.

The wait does not adapt to the caller on this target. Unlike the LiveKit driver,
which has a `dynamic` endpointing mode, this is the same fixed window on every
turn.

**Lowering the floor alone does not shorten a turn**, and on this target
widening it makes some turns *slower*. pipecat-slng asks the bridge to finalise
when the VAD reports the caller stopped; a final that has already arrived by then
finds no request outstanding, so the frame goes out unfinalized and Pipecat waits
out a flat 1.0s safety net instead. Measured over eight turns at 0.4s: every turn
whose transcript beat the window paid a flat 1.000s, and observed transcripts
arrive from 0.27s. That is why the floor here stays at 0.2s for every pace while
LiveKit's moves, and why the inconsistency is deliberate.

The end-of-turn analyzer (`LocalSmartTurnAnalyzerV3`) runs on every call. It is
constructed explicitly only so the ceiling is reachable; left implicit it sits at
its own 3s. A transcriber that detects turns itself only *proposes* an ending
while this pair is bound; `turn: provider: listen` hands it the decision.


## Task returns

When a task returns, its internal turns stay out of the owner's conversation.
The completed task call stays in that conversation with the typed result,
so the same caller request does not start the task again.

A task may list a handoff in its own `handoffs:`. Calling it ends the active
task and any remaining task-group steps, then hands the caller directly to the
receiving agent. It does not return through the task owner. A task definition
has `tools:` and `handoffs:` and no other list, so a task or an escalation on a
task is unwritable rather than rejected: there is no key to put one in.

### Steps that end on their own tool

`manage_booking` ends when `save_booking` returns a result the package calls a
success. The step saves its `assign:` from that result and moves on with no
model request in between, and the result never reaches the model. Anything that
is not a success goes back to the model as an ordinary result.

`verify_customer` ends when `find_or_create_customer` returns a result the package calls a
success. The step saves its `assign:` from that result and moves on with no
model request in between, and the result never reaches the model. Anything that
is not a success goes back to the model as an ordinary result.

After one of those tools succeeds, none of that step's own mutations runs again
in the same invocation: a repair cannot book a second appointment. The model's
`finish` is still there for a request the step cannot serve, for values it
already holds, and for recording a result whose save was refused; that repair
keeps the values the tool returned. The step says nothing after its tool: the
next step opens, or the owner speaks once when the flow returns.

### Group steps that can be skipped

`book` skips `verify_customer` when `customer_phone` is confirmed as the group
starts, and runs it otherwise. Every entry of `verify_customer`, standalone included,
withdraws that confirmation until it saves the value again.

A group also stops when a step ends unserved: the later steps do not run and the
owner is handed the unserved status.

A step is refused, rather than run, when the caller has not spoken since it last
returned. A completed result with no new caller turn above it is the model
reading its own finished work as though it were a fresh request, and running the
step again answers nothing while costing the caller a turn. The refusal arrives
before the step's `announce:`, so a refused call speaks nothing at all.

A task that lists no handoff still has a way out. Every node's
`role_message` here ends with one generated rule: when the caller asks for
something none of the step's tools can do, and no handoff on the step covers it,
call that node's `finish_<task>_<step>` function right away with the closest
result, naming that request in `unserved_request`. Every finish schema here
declares that field and never requires it, and a filled one rides the results
handback that tells the owning agent to take the request next. The rule does not
excuse skipping the step's own work, and a handoff the step declares wins over
it.

In a group with `context_scope: shared`, each exact typed result enters the
shared context before the next task starts. An isolated group carries no
results between steps. The final `merge: results` map is keyed by task name.

## Context a step or handoff starts with

A task or a handoff can set `context.history` to something other than
`full`. That changes what the step, or the agent taking the handoff, starts
with.

`history: full` needs no line here: one `LLMContext` is shared for the whole
call, so a step with nothing authored already has the running transcript,
tool records included.

`history: messages` keeps what the caller and the agent said out loud. Every
tool record is dropped, and a call leaves with its result: on this target the
call is a field on an assistant message and only the reply is its own entry, so
keeping one without the other is a request the provider rejects mid-call. A turn
that both spoke and called a tool keeps what it said.

`history: last_n` keeps the newest `max_messages` entries, tool records
kept. The cut lands at the front, so a leading tool result whose matching
call fell outside the window is dropped with it: a tool message with no
matching call is a request the provider rejects mid-call.

`history: reset` gives the step only its own instructions and its declared
variables. That is the floor: a reset step does not receive the call that
triggered it, the caller's own words included. Write a reset step to be
fully described by its declared values, like confirming a number or taking a
payment. A step that has to interpret what was just asked does not fit
`reset`, and there is no compiler error for getting this wrong.

`history: summary` is refused on this target. It exists only on LiveKit.

The ceiling is the same at every value: no `history:` setting is a privacy
control. Shortening what a step sees does not unsay what the caller already
said out loud.

`uv` installs the pinned dependencies on first run. The Pipecat dev runner
serves a WebRTC test client you can open in the browser.

That command is for talking to the agent right now, with no phone involved. For a
real phone call through your own number, see **Telephony setup** below: deploy,
then a few actions in your carrier's console, and nothing running on your side
at all.

While iterating on the source spec, `unmute dev <source-dir>` recompiles this
project and runs it for you behind a dev client.
## Knowledge bases

The agent answers from these documents rather than from what the model
remembers. Each folder was copied into this project by `unmute compile`.

- `refunds`, from `knowledge/refunds/`, embedded with `text-embedding-3-small` (needs `OPENAI_API_KEY`). `hybrid` search over passages of 90 tokens overlapping 20, returning 3 per lookup
- `services`, from `knowledge/services/`, embedded with `text-embedding-3-small` (needs `OPENAI_API_KEY`). `hybrid` search over passages of 220 tokens overlapping 40, returning 3 per lookup

Four things to expect at run time:

- **Content is fixed until the next compile.** The documents in `knowledge/` are
  what this project searches. Editing the originals in the source package changes
  nothing here until you compile and deploy again.
- **Indexing runs once per container, at import, not per session.** Pipecat Cloud
  imports `bot.py` and then calls `bot()` for each session, and it keeps a warm
  container for about five minutes after one ends, so the cost is amortised across
  every session that lands on it. A failure raises during import, so the container
  never reaches ready and the platform refuses the start. That is the intent: an
  agent that cannot read its documents should not take calls.
- **Bake the index into the image to make that import cheap.** Without it every
  cold container embeds the corpus before it can accept a session, and image
  warm-up is already the thing that decides how quickly a scale-up starts serving.
  Baked, that import is a disk read instead.

  ```sh
  docker build --build-arg KNOWLEDGE_BAKE=1 \
    --secret id=OPENAI_API_KEY,env=OPENAI_API_KEY .
  ```

  The credential arrives as a build secret, so it is never written into a layer.
  Leave the flags off and the image still works, and the log says at startup that
  it is embedding instead. A lookup still embeds the caller's question, so the run
  time needs the same credential either way: baking removes the corpus cost, not
  the query cost.
- **A document with no text layer is named and skipped at startup.** A scanned PDF
  is an image of a page, and nothing can extract text from it without OCR. If some
  documents are readable the agent still starts and the log says how many of how
  many were indexed. If none are, the import fails.
- **A lookup that fails does not end the call.** The agent is told the lookup is
  unavailable and says so. The reason goes to this container's log and never into
  the model's context, because a model handed a provider error can read it out
  loud.

**Importing `bot.py` builds the indexes, so a plain import needs a working
embedding credential.** That is deliberate: the platform imports this module and
its readiness probe does not pass until the import finishes, which is what keeps
a caller from waiting for indexing. It does mean a tool that imports the module
just to inspect it — a type checker, an editor, a script — reaches the embedding
service too. Give it a valid key, or stub `knowledge._embed_<base>` before the
import.

Both `utils/` and `knowledge/` have explicit `COPY` lines in the Dockerfile.
That is not decoration: this image copies named paths only, because `/app` belongs
to the base image, and leaving either one out gives a container that starts,
raises, and never reaches ready while its own log looks healthy.

Nothing here writes outside the container. The index is in memory for the life of
the process and is not persisted anywhere.

## Input variables

This agent expects values supplied when the call starts:

- `appointment` (Appointment | None)
- `booking_date` (str (date) | None)
- `booking_weekday` (str)
- `customer_phone` (str (phone) | None)
- `customer_verified` (str)
- `salon_local_time` (str (time) | None)

Each value is checked against its field's type in `state.py` before the greeting.
A value that does not fit stops the call start, and the error names the field.
A model or a list is passed as JSON.

Locally, pass them with `unmute dev` on the source package:

```sh
unmute dev <source-dir> --var appointment=... --var booking_date=... --var booking_weekday=... --var customer_phone=... --var customer_verified=... --var salon_local_time=...
```

In production the same values ride the runner's call-start payload as one flat
JSON object, and the bot also reads them from the `UNMUTE_CALL_START`
environment variable when the payload carries none.

## Declared state

The call's state is the `State` class you wrote in `state.py`, copied here as
you wrote it. `call_state.py` is generated. Its `CallState` subclasses your
`State` and is the one way a value is saved. It also tracks which values still
await a confirmation and renders a value for a prompt. Each task step gets its
own result model, built from your field types when the module loads.

A task's `assign` derives its finish fields from the fields it saves into. A
step must return every whole value it assigns, except one its `assign` marks
with `?`. A failed or unserved finish saves nothing.

A prompt reads only its own `{{state.variable}}` or
`{{state.variable.field}}` references. Missing values render as
`none recorded yet.`; false, zero and empty lists keep their values.
Referencing an object shares the whole object. History is a separate choice and
does not clear saved values.

A field's type is the check every save goes through: a step's result, a
pre-fetched value, a dispatched one and a carrier fact. The values one save
brings are checked together, and nothing is recorded unless all of them fit.
The previous contents then stay as they were, and the model is told which field
and what was allowed, which is what lets it correct itself on the next turn.

A format or pattern rule in your types reaches the model as words in the field's
description. The schema carries no `pattern` or `format` keyword, because a
strict provider rejects one. The rule is still checked where the value is saved.

`state.py` imports from `pydantic_extra_types`, so this project depends on it
with the extras those types need.

## Required environment

Set these before running (values are never written into the project):

- `GOOGLE_API_KEY`
- `LANGFUSE_BASE_URL`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `MANAGER_PHONE_NUMBER`
- `OPENAI_API_KEY`
- `SLNG_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`

One group, because one thing reads them: the deployed agent. There is no second
side on this route, so every name above belongs in the platform secret set.

### If every provider call fails to connect

The OpenAI SDK this project resolves verifies TLS against the operating system's
own certificate store rather than a bundle shipped inside the package. On an image
with no system certificates every provider call fails at the handshake, and the
error names TLS rather than the provider.

The base image this project builds on is the framework vendor's own and normally
carries them. If you swap the base for a minimal image, install `ca-certificates`
in it, or point at a bundle you supply:

```sh
SSL_CERT_FILE=/path/to/ca-bundle.crt
```

`REQUESTS_CA_BUNDLE` is read by some libraries and not by this one, so set
`SSL_CERT_FILE` even if you already set the other.

## Deploy to Pipecat Cloud

Authenticate once first with `pipecat cloud auth login`. After that, one command
does the first deploy and every later one: `pipecat cloud deploy` builds the image
in the cloud from this project's `Dockerfile` and deploys it. That is why the
manifest names no image; naming one switches cloud builds off.

Create the secret set first, because the manifest already references it and a
deploy cannot start without it:

```sh
cp .env.example .env            # then fill in the values
pipecat cloud secrets set salon-concierge-pipecat-secrets --file .env --region eu-central
pipecat cloud secrets list salon-concierge-pipecat-secrets
```

The listing shows names and readiness, without values. Wait for `ready`, then deploy:

```sh
pipecat cloud deploy
pipecat cloud agent status salon-concierge-pipecat
```

If that set name is already taken, create yours under another name and pass
`--secrets <other-name>` on every deploy, which overrides the manifest.

### Update secrets without rebuilding

Run the same `secrets set` command after changing `.env`. It adds new names and
replaces supplied values; omitted names keep their values. Remove an obsolete
name explicitly with `pipecat cloud secrets unset salon-concierge-pipecat-secrets OLD_API_KEY`.
Use the same organization and region as the agent.

Wait for `pipecat cloud secrets list salon-concierge-pipecat-secrets` to report `ready`.
The set does not refresh running agents, so roll out each deployment using it.
Read the current build ID from deployment history and replace the placeholder below:

```sh
pipecat cloud agent deployments salon-concierge-pipecat
pipecat cloud deploy --build-id <current-build-id> --force
pipecat cloud agent status salon-concierge-pipecat
```

This reuses the current image without rebuilding. `--force` replaces running
instances and can interrupt active sessions. Verify a fresh interaction afterward.
Updating a stored value does not create or revoke its provider key.
See [secret updates](https://docs.pipecat.ai/pipecat-cloud/fundamentals/secrets)
and [reusing a build](https://docs.pipecat.ai/pipecat-cloud/guides/cloud-builds#redeploying-a-previous-build).

Deploying the same agent name again updates the existing agent rather than making
a second one. Sessions already running finish on the old image, and a deploy that
fails leaves the previous ready version serving traffic. `status` reporting
`ready` is the deploy actually being usable.

### This agent's name

**salon-concierge-pipecat** is `name:` in your `agent.yaml` joined to the `pipecat`
target it was compiled for, and it is `agent_name` in `pcc-deploy.toml`. Agent
names are unique across your organisation and the deploy above updates whichever
agent matches, so this name says which package this is rather than which
platform it runs on. The secret set follows the same name,
because a secret set is per-organisation in the same way.

Renaming it in the source package does not move this deployment. The next deploy
creates a second agent and the old one keeps running and keeps billing, so
delete it with `pipecat cloud agent delete salon-concierge-pipecat` and
create the secret set again under the new name.

The manifest holds **1** instance ready, from `warm_instances` on
this target in `targets.yaml`. That is a standing cost and it is what removes the
cold start. Change the number in the source package and recompile: this file is
rewritten on every compile, so the declaration belongs there and reaches every
later deploy on its own.

### Reading this agent's log

```sh
pipecat cloud agent logs salon-concierge-pipecat -n 250
pipecat cloud agent logs salon-concierge-pipecat -s <session-id>
```

**Do not pass `-l DEBUG`.** It filters to *only* that level rather than setting a
minimum, so it hides the `ERROR` line that says why a session died. Run with no
`-l` at all.

The CLI wraps output at 80 columns, which corrupts its JSON output. For machine
reading, widen it first:

```sh
COLUMNS=100000 pipecat cloud --output json agent logs salon-concierge-pipecat -n 2000 -s <session-id>
```

`pipecat cloud agent sessions salon-concierge-pipecat` lists the sessions themselves, which is
how you tell a session that never started from one that failed inside. A session
marked `Complete` with a blank `Bot Start Seconds` never reached `bot()` at all.

### Region

This agent deploys to `eu-central`, from the manifest. Its secret set
carries the same region, because a set is region-scoped and an agent can only
read one in its own region.

A second region is a second agent, because agent names are globally unique
across regions:

```sh
pipecat cloud secrets set salon-concierge-pipecat-<region>-secrets --file .env --region <region>
pipecat cloud deploy salon-concierge-pipecat-<region> --region <region> --secrets salon-concierge-pipecat-<region>-secrets
```

`pipecat cloud regions list` prints the regions available to you.

## Telephony setup

**Nothing runs on your side and nothing is hosted by you.** Your carrier streams
the call's audio straight to Pipecat Cloud, which starts this agent.

The whole setup is **four steps**: three in your carrier's console, plus one
lookup command you can run anywhere. In production there is nothing of yours to
run, host, or keep awake.

### Who does what

Two parties, and neither of them is you.

- **Twilio** owns the phone number and holds the small piece of call markup that
  sends a call's audio to the platform.
- **Pipecat Cloud** runs this agent and receives that audio directly.

The carrier part is yours to do, because nobody but you can sign in to your
carrier account, so every step is written out.

### At your carrier (Twilio)

Do these in the [Twilio Console](https://console.twilio.com), in this order.

1. **A voice-capable number you own in this Twilio account.** Phone Numbers,
   Manage, Buy a number, with the Voice capability. Skip this if you already own
   the number you want. If another target in this repository already uses a
   Twilio number, the same account and the same number serve this one.

   This same number is also what the person you call sees, and there is nothing
   else to enable on it: voice capability is the whole requirement. See
   **What the recipient sees** below for the two account-level things that can
   still refuse a call.

   Take the number off any SIP trunk first. A number attached to a trunk ignores
   its voice configuration completely, and it does so silently, so the markup you
   create in step 3 would never be consulted. See **One number serves one target
   at a time** below.

2. **Your organization name**, the one value the compiler cannot know:

   ```sh
   pipecat cloud organizations list
   ```

   **You want the machine slug, not your display name.** The output has more than
   one column and their headings differ between CLI versions, so go by the shape
   of the value rather than by the heading:

   ```text
   Organization Name   Organization ID           Active
   ─────────────────────────────────────────────────────
   your-display-name   three-random-words-12345  active
                       ^^^^^^^^^^^^^^^^^^^^^^^^ this one
   ```

   The value you want looks like `three-random-words-12345`: lowercase words
   joined by hyphens, ending in a number, never a space in it. Depending on your
   CLI version that column is headed **Organization ID** or just **Name**. It is
   never the human-readable name you chose, and never carries the `(active)`
   marker.

   Getting this wrong is the most common way this route fails, and it fails in the
   least helpful way: the caller hears the spoken line, then silence, and **your
   agent's log stays completely empty**, because a connection the platform refuses
   never reaches your agent at all.

3. **Create a TwiML Bin** holding exactly this, with your organization name in
   place of `YOUR_ORGANIZATION`. Console path: TwiML, TwiML Bins, then the plus
   button. Name it anything; the content is what matters.

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
     <Say>Connecting you now.</Say>
     <Connect>
       <Stream url="wss://eu-central.api.pipecat.daily.co/ws/twilio">
         <Parameter name="_pipecatCloudServiceHost" value="salon-concierge-pipecat.YOUR_ORGANIZATION"/>
         <Parameter name="from_number" value="{{From}}"/>
       </Stream>
     </Connect>
   </Response>
   ```

   Five things in there are deliberate:

   - **The spoken line comes first.** Starting this agent takes a few seconds
     from cold, and a caller who hears nothing hangs up. Delete the `<Say>` line
     once you keep a warm instance (this deployment holds 1 ready).
   - **The address is already the right one**, including the region: this file
     was generated for this deployment, so paste it as it is rather than copying
     one from a guide.
   - **The agent name is already filled in.** `YOUR_ORGANIZATION` is the only
     substitution you make.
   - **`{{From}}` is the caller's number**, and Twilio substitutes it as it
     builds the Bin's response. That is the only place this agent can learn it:
     the platform terminates the carrier's stream itself, so nothing else in this
     package ever sees the call's own details. It is what `source: from_number`
     reads.
   - **There is nothing else in it.** A `<Parameter>` reaches this agent only if
     the emitted code reads one by that name, and a parameter nothing reads is
     markup you paste and then trust. Only the two above are read here.

   **If you already have a Bin, add the `from_number` line to it.** An older Bin
   without it is not an error and does not fail the build: nothing checks your
   Bin, because reading it back would need carrier credentials this command does
   not ask for. What happens instead is that the number arrives empty, every
   entry reading it is skipped, and the agent's log says so by name:

   ```text
   prefetch caller: skipped, the call carries no from_number
   ```

   That is also what a caller who withholds their number produces, which is the
   point: the agent behaves the same either way, and asks.

4. **Point the number at that Bin.** Phone Numbers, Manage, Active Numbers,
   your number, Voice Configuration, "A call comes in", choose TwiML Bin, pick
   the one you just made, Save.

   Read the state back rather than testing by ear:

   ```sh
   twilio api core incoming-phone-numbers list \
     --properties phoneNumber,trunkSid,voiceUrl
   ```

   An empty `trunkSid` is what you want. A `trunkSid` with a value means the
   trunk still owns the call and your Bin is never reached.

### One region, three places

This agent's region is `eu-central`, declared once in the source
package's `targets.yaml` as `deployment_region`. Three things are derived from that
one declaration, and the platform requires them to agree:

| What | Where it lands | Why it has to match |
|---|---|---|
| where this agent runs | `region` in `pcc-deploy.toml` | it is the deployment |
| where its secrets live | the `--region` on the `secrets set` command | an agent can only read a secret set from its own region |
| where your carrier streams to | the `wss://` address in the markup above | a regional stream endpoint routes **only** to agents deployed in that region |

Following this file keeps all three in step, because all three are rendered from
the same declaration. To move region, change that one line in `targets.yaml`,
recompile, and re-paste the address into your Bin. `pipecat cloud regions list`
prints what is available to you.

Two things bite when you move an agent that already exists:

- **Agent names are globally unique across regions**, so `salon-concierge-pipecat` cannot exist
  in two regions at once. Either delete the old deployment first, or deploy the new
  region under another name and change the agent half of the service host in your
  Bin to match it.
- **A secret set is region-scoped and its name is global.** Create one for the new
  region under a new name and pass `--secrets <that name>` on the deploy, or delete
  the old set first.

Where these steps come from, if you want the upstream reading:

- The working example this route is modelled on:
  <https://github.com/pipecat-ai/pipecat-examples/tree/main/twilio-chatbot>
- The wire protocol between your carrier and the platform:
  <https://www.twilio.com/docs/voice/media-streams/websocket-messages>
- What a TwiML Bin is and how to create one:
  <https://help.twilio.com/articles/360043489573-Getting-started-with-TwiML-Bins>
- Regions, and the rule that the stream endpoint must match the deployment:
  <https://docs.pipecat.ai/pipecat-cloud/guides/regions>

### On this side

**Nothing, in production.** That is the whole section, and the absence is the
feature. Deploy per **Deploy to Pipecat Cloud** above, wait for `ready`, and the
number works. (This package places or redirects calls, so fill the
names under **Required environment** and push the secret set before deploying.)

### Hear it before the phone

`unmute dev <source-dir>` opens a browser session with this same agent: the same
prompt, the same tools, the same models, and no phone involved. Get the
conversation right there.

The phone leg itself starts where that stops, and it starts at a deployed agent.
Deploy, do the **Wire up Twilio** steps above, then call your number. A carrier
reaches you over the public internet, so nothing running on your laptop can
stand in for that.


### What a transfer does, and what it cannot do

When the agent hands the caller to a person, one request replaces the live call's
markup at your carrier: a spoken line, then the destination dialled with ringback
audible, then a final hangup that ends the original call after that destination
leg finishes. The
destination is read from the environment at call time, so no phone number is
written into this project.

This is deliberately a cold transfer. If the person hangs up first, the original
call ends. If the destination declines or never answers, the original call also
ends after the dial timeout. No fresh agent starts with an empty conversation.

Pipecat `cloud-websocket` requires explicit `on_unavailable: hangup`; it cannot
reconnect the original media stream. Omitting the field resolves to
`return_to_caller`, so validation refuses it on this route. A successful carrier
REST update means the transfer has started, not that the destination answered;
the tool result is `transfer_started`.

**Warm transfer is not supported on any Pipecat target in Unmute.** Use the
LiveKit SIP route when the agent must brief the person before connecting the
caller. See the telephony and transfer overviews in the Unmute documentation
site.

### Security, in one paragraph

This route's deploy manifest sets `websocket_auth = "none"`, because a static
piece of markup in a carrier console cannot fetch a token first. What limits who
can start a session, then, is knowing the `salon-concierge-pipecat.<your organization>`
string your markup carries. Treat that string like a capability: it is not a
secret in the cryptographic sense, but anyone holding it can open sessions you
pay for. This is the platform's documented model for carriers that connect
directly.

### Every step, in order, from a source change to a phone call

Run it from the **source package**, not from this directory: this directory is
generated output and recompiling overwrites it.

**1. Recompile after any source change.**

```sh
unmute validate <source-dir>
unmute compile <source-dir>
```

**2. Move into the build directory.**

```sh
cd <source-dir>/build/pipecat
```

**3. Fill the environment and push it.** `.env` is not regenerated over, so this
survives recompiles:

```sh
cp .env.example .env      # then fill in the values
pipecat cloud secrets set salon-concierge-pipecat-secrets --file .env --region eu-central
```

Wait for `pipecat cloud secrets list salon-concierge-pipecat-secrets` to report `ready`.
For a later value change, follow **Update secrets without rebuilding** above.

**4. Deploy, and wait for ready.**

```sh
pipecat cloud deploy
pipecat cloud agent status salon-concierge-pipecat
```

Do not place or expect a call until `status` reports `ready`.

**5. Do the carrier part once**, per **At your carrier** above. It only changes
again if you change region or agent name.

**6. Test an incoming call.** Ring your own number. Expected: the spoken line
within a couple of seconds, then this agent.

**7. Test a transfer, both ways.** On a call, ask for the person this agent can
hand you to. Expected: the accepted update ends the agent's stream, then the
carrier speaks the line and rings the destination. Let the person **hang up
first**: expected is the original call ending, with no new agent greeting. Then
run it a third time with
the destination **declining or not answering**: expected is the original call
ending after the dial finishes. Those last two runs cover the ordinary and
failure paths.

One log, and only one: `pipecat cloud agent logs salon-concierge-pipecat`. There is no
second place to look on this route, because there is no second thing running.

### If the call feels slow

The platform writes the numbers into that log, so read them before changing
anything. Two lines answer almost every latency question:

```text
[pcc-observability] First bot speech | latency=2.297s
[pcc-observability] Latency breakdown | events: ["...STT: TTFB 17.651s", "...LLM: TTFB 2.215s", ...]
```

- **First bot speech** counts from inside the process only. If it reads about two
  seconds and your caller waited much longer, the difference was the container
  starting: see the cold-start row above.
- **The breakdown** names every part of the turn, not only the services: a part
  owned by a service far above the others is that provider, not this project and
  not the platform, so take it up with them or bind a different provider for that
  role in the source package and recompile. A part owned by a **setting** is
  yours to change in the package, a part owned by the **bot** is this project's
  own code, and a part owned by the **pipeline** is time inside the framework.
  `unmute dev` shows the same parts per reply while you are still on the browser
  loop, which is the cheaper place to read them.
- **End-of-turn detection sits between the two.** The pipeline waits to be sure
  the caller has finished before it answers, which costs a second or two per turn
  on purpose. It is the difference between an agent that interrupts and one that
  feels polite.

### One number serves one target at a time

A number's Voice configuration is **either** a webhook or a TwiML Bin or a SIP
trunk, never two at once, so there is never a question of which answers:
whichever the number points at is the one that gets the call. Both of your agents
can stay deployed the whole time, and taking one down is not how you switch
between them.

Moving a number off a trunk and onto a Bin is two changes, and the order matters:
**take it off the trunk first, then set the Bin.** Setting the Bin while the
number is still on a trunk does not move it, and that failure is silent. Read
where a number points with the command in step 4 rather than wondering which
agent answered. If two of your agents share a greeting and a voice they are
indistinguishable by ear.

### When something does not work

| What happens | Where to look |
|---|---|
| A working agent answers, but not this one | The number is still on a SIP trunk or an old webhook, so your markup was never consulted. Read the number's configuration with the command above. |
| The call connects, the spoken line plays, then silence and a drop | Either the service host in your markup is wrong or this agent is not `ready`. **The tell is that this agent's log is empty**: a connection the platform refuses never reaches it. Check `pipecat cloud agent status salon-concierge-pipecat` first. Then re-read the service host: the agent name is `salon-concierge-pipecat`, and the organization is the **slug** from `pipecat cloud organizations list`, the one shaped like `three-random-words-12345`, never your display name (carrier action 2). Twilio's own call log names the stream error outright, under Monitor, Logs, Calls. |
| The spoken line plays, a long silence, then the agent | A **cold start**, and it is the single biggest thing you can fix. With no warm instance the platform has to start a container after the call arrives: schedule it, import pipecat, load the voice-activity and end-of-turn models. That is the silence, and it happens on the first call after every idle period, not just after a deploy. This deployment already holds 1 instance ready, so if you are reading this row the pool is not the cause: check `pipecat cloud agent status salon-concierge-pipecat` for a deploy that has not gone ready. Your agent's own log tells you which half you are looking at: `[pcc-observability] First bot speech | latency=...` measures only the part inside the process, so if that says two seconds and the caller waited twelve, the other ten were the container starting. |
| The spoken line plays and the agent never arrives at all | The same cold start, past the point where the session gives up. A session that expires before the container is listening leaves an **empty session**: `pipecat cloud agent sessions <name>` shows it as `Complete` with a blank `Bot Start Seconds`, and the agent log has no `pipecat.workers.runner` line for it, because `bot()` was never called. Measured 2026-08-26: a session created at T+0 ended at T+10 while the server bound at T+12. So a warm instance is not only a latency nicety here, it is what makes the call answerable. This project has a knowledge base, which makes it much easier to hit: the corpus is embedded at import, before the server binds, so read the "Baking the index" section above. |
| Silence or degraded audio, everything else correct | Your markup names a different region than this deployment. The address in this file is the right one; re-paste it. |
| Nothing in the agent log at all | The call never reached the platform. That is the first row, or the number is not the one you think. |
| The transfer never rings the destination | The per-call check refuses a missing destination by name at session start, so if the session started, read the agent log for the update request's response. |


## Trace with Langfuse

Tracing is configured. Set `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and
`LANGFUSE_BASE_URL` in `.env`; all three are required. The trace name is
`concierge-salon-concierge-pipecat`; one trace contains the full conversation. The
Pipecat runner session ID is also the Pipecat conversation ID and Langfuse
session ID.
Pipecat tracing owns the process OpenTelemetry provider and startup fails if another SDK provider is installed first.


Starting the worker or exporting a synthetic span proves connectivity only.
Complete at least one user turn before reviewing the trace. A traced request
contains `conversation` and `turn` spans plus `llm` and `tts` generation
observations.

Traces follow the Langfuse v4 data model. One call is one trace. Open it and
the root observation holds the whole conversation, so the top of the trace reads
as the call rather than as an envelope. Inside it each exchange is a `turn` span
you can open on its own: what the caller said, what the agent replied, and every
model and tool call in between.

The session ID and the trace name are on every observation, not only on the top
one, so filtering by session or adding up what a call cost covers all of it.
Langfuse trace input and output are deprecated and this project writes neither.

Traces can contain caller speech, model input and output, and tool arguments and results.
Use only fake identities and fake customer data for release tests. Use a separate
Langfuse project for those tests, and do not send real customer data until its
access and retention rules are approved.


## Carrier setup you have to do yourself

Going live on this route needs real carrier configuration. Every step below is
dictated by the route and none of them is done for you. Until they are done, the
number does not reach this agent.

1. select a Voice-capable number you own in this Twilio account (or reuse the one another target already uses; a number serves one target at a time), and take it off any SIP trunk first, because a number on a trunk ignores its voice configuration silently
1. run `pipecat cloud organizations list` and note the organization name; it is the one value the compiler cannot know
1. create a TwiML Bin holding the exact markup the generated README dictates, with the organization name pasted in (Twilio Console, TwiML Bins, create)
1. point the number's "A call comes in" at that TwiML Bin (Phone Numbers, Manage, Active Numbers, your number, Voice Configuration)

## Agents

- `complaint_specialist`
- `concierge`
Entry agent: `concierge`.
