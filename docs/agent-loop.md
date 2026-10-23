# Agent loop

One `Agent` is one conversation. All state lives on the entity; the systems in
`agent.rs` drive it. This note describes one turn from input to the next turn.

## History model

`llm_history_model::history::Entry` is the only transcript representation:

- `SystemPrompt { content }` — exactly once, at index 0
- `SystemMessage { content }` — in-conversation system updates
- `UserMessage { content: { text, image } }`
- `Response { blocks }` — assistant text/thinking (committed reply)
- `ResponseToolCall { tool_call }` — one batch of tool calls
- `ToolCallResult { tool_call }` — results for the preceding batch, in order
- `Draft { blocks }` — the in-flight response while streaming

`llm_api::validation` enforces the role/ordering invariants before serialization
(system prompt placement, user-first, no orphan tool results, unique ids, valid
tool arguments). A validation failure becomes a request-build error for the
agent, not a panic.

## Turn lifecycle

1. **Input** — `submit_input` reads the focused text input and pushes
   `QueuedMessage::User`; `deliver_pending_messages` flushes it at the next
   legal boundary and writes `StartAgentTurn`.
2. **Start** — `start_agent_turn`:
   - skips if the agent already has an active turn (`AgentRuntime::active`)
   - builds `ModelConfig` with the advertised tool snapshot
   - builds the provider-specific request body (`transport::request_body`)
   - creates a cancellation `watch` and a `Decoder`, inserts `ActiveTurn`
   - spawns `stream_transport` on Tokio with a cloned `TaskContext`
3. **Stream** — `stream_transport` writes `StreamChunk` messages; on normal end
   `StreamEnded`, on failure `StreamFailed`.
4. **Decode** — `decode_stream_chunks` feeds bytes into the provider decoder,
   producing `DraftBlock`s and `AgentStreamEvent`s (thinking/text/tool-call
   deltas, usage, completion).
5. **Live UI** — `handle_stream_events` updates the assistant card, thinking
   collapsible, status line, usage readout, and scroll pin.
6. **Commit** — `AgentTurnFinished { entries, usage, stop_reason }` is handled by
   `handle_turn_finished`:
   - verifies the turn id is still current
   - appends the committed entries (`Response`/`ResponseToolCall`)
   - records usage and writes the history dump
   - if the response contains tool calls, writes `RunToolCalls`
   - otherwise marks the agent idle and ends the turn
7. **Tools** — `execute_tool_calls` (see `docs/tool-calls.md`) dispatches; when
   all results are available, `ToolRoundFinished` is handled by
   `apply_tool_results`, which appends `ToolCallResult` and writes
   `StartAgentTurn` for the next model turn (unless the agent called
   `end_agent`).
8. **Next turn** — the loop repeats until the model replies without tool calls.

## Pending messages

Messages that arrive while a turn is active wait for a legal boundary (after a
reply or after a completed tool round):

```rust
enum QueuedMessage {
    User(String),
    Agent { from: String, text: String },     // send_message_to_agent
    Task(String),                             // spawn_agent initial prompt
    BackgroundResult { task_id, result, is_error },
}
```

`flush_pending_messages` inserts as many as fit at one boundary;
`deliver_pending_messages` starts a turn when the agent is idle and has queued
input.

Cross-agent and background messages are wrapped as `SystemMessage`-style user
envelopes so the provider sees them in order without breaking role alternation.
`spawn_agent` prompts are delivered verbatim.

## Retry, cancel, failure

- **Cancel** (`CancelAgentRequest`): signals the transport watch, rolls the
  pending user turn back out of history, drops the active turn and tool
  bookkeeping for that agent (`ToolRounds::cancel_rounds`), and leaves the
  agent idle.
- **Retry**: resends the last unanswered user message (`retry_message`), after a
  failure or cancellation.
- **Failure** (`StreamFailed` / decode error): `handle_agent_failed` records the
  error, resets turn state, and offers Retry; a request-build error rolls the
  turn back and shows the cause.
- **Removal** (`end_agent` or Kill): queues `RemoveAgentRequest`; the round still
  commits, then `remove_agents` despawns the agent, its UI, JS runtime, rounds,
  detached/nested calls, and budgets.

## Limits and diagnostics

- `MAX_AGENTS = 64`.
- `history_dumps.jsonl` in the working directory receives JSON dumps on turn
  commit, request-build errors, prompt-cache regressions, and rollbacks; it is
  gitignored and meant for forensics.
- `previous_history` + `last_history_change` hold one step of rollback context
  and are included in dumps.
- Usage is merged per turn (`UsageProvider` labels DeepSeek/Z.ai reporting in
  the OpenAI-compatible adapter).
- The footer shows token usage and `Currently running toolcalls: N`
  (`ToolRounds::running_for` counts unresolved round slots, nested calls, and
  detached tasks).

## Where to look

| Concern | File |
| --- | --- |
| Turn start/commit | `src/agent.rs` (`start_agent_turn`, `handle_turn_finished`) |
| Streaming/decoding | `src/agent.rs` (`decode_stream_chunks`), `src/transport.rs` |
| Tool round application | `src/agent.rs` (`apply_tool_results`) |
| History model/validation | `crates/llm_history_model`, `crates/llm_api/src/validation.rs` |
| UI rendering | `src/transcript.rs` |
| Tool-call presentation (`execute` unwrapping) | `src/tool_display.rs` |
| Provider request bodies | `src/transport.rs` + `crates/llm_api/src/{openai_responses,openai_completions,anthropic}` |
