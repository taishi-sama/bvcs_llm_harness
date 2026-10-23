# Providers and the `llm_api` crate

`crates/llm_api` is provider plumbing with no Bevy dependency. The harness
selects one wire format and one streaming decoder per request.

## Providers

`Provider` (`src/config.rs`) chooses the wire API. Defaults:

| `PROVIDER` | Endpoint | Model kind |
| --- | --- | --- |
| `responses` (default) | `http://127.0.0.1:8080/v1/responses` | OpenAI-compatible |
| `completions` | `http://127.0.0.1:8080/v1/chat/completions` | OpenAI-compatible |
| `anthropic` | `http://127.0.0.1:8080/v1/messages` | Anthropic-compatible |

`ENDPOINT` overrides the URL. `API_TOKEN` adds `Authorization: Bearer …` when
non-empty. `SESSION_ID` becomes `x-opencode-session` and the prompt-cache key.

Both OpenAI paths are treated as compatible endpoints (llama.cpp, OpenCode Go,
DeepSeek, Z.ai). Anthropic is best-effort: unsigned thinking blocks are accepted
(`allow_unsigned_thinking`) and system messages are serialized as user messages
inside a `<system_update>` envelope by default (`SystemMessagePolicy::Reject` is
available if a provider cannot represent them at all).

## Request side

`transport::request_body` dispatches to the matching adapter:

- `openai_responses` — Responses API (typed input items, tools, reasoning)
- `openai_completions` — Chat Completions (role messages, tool calls)
- `anthropic` — Messages API (content blocks, thinking, tools)

Shared pieces live in `wire_common`:

- `parameters_schema`: sorted properties; `required` contains only parameters
  marked `required` (omitted entirely when none); `additionalProperties: false`;
  `type` is `string` / `number` / `boolean` / `object`.
- MIME strings and thinking-block extraction.

`ModelConfig` carries `kind`, the tool snapshot, and `max_tokens`.

**Strict mode caveat:** OpenAI strict function calling requires every property
in `required`, so optional parameters are only safe with non-strict or
compatible endpoints. Schema generation does not pretend otherwise.

## Stream side

Each adapter owns a `StreamDecoder` that consumes raw bytes (`RawStreamBytes`)
and produces normalized `StreamEvent`s:

```rust
ThinkingDelta, SignatureDelta, TextDelta,
ToolCallDelta { index, id, name, arguments },
Usage(Usage),
Completed { stop_reason }
```

- SSE framing is in `sse.rs`; quirks are adapter-local.
- Tool-call arguments arrive as string deltas and are assembled per index.
- Usage is merged into one snapshot; `UsageProvider` labels DeepSeek/Z.ai
  reports that share the Chat Completions shape.
- `Decoder` in `src/agent.rs` wraps all three and feeds `DraftBlock`s.

`stream.rs` also defines `StreamError` (invalid JSON, provider error, missing
tool-call id/name, not-a-draft) and the draft/finalize helpers.

## Validation

`validation.rs` runs before serialization and rejects history that a provider
cannot represent:

- `SystemPrompt` exactly once at index 0
- conversation starts with a user turn
- legal transitions / role alternation
- non-empty user/assistant content
- tool arguments are a JSON **object**
- each `tool_use_id` is called in the immediately preceding batch, once, and
  has a matching result

A violation is a request-build error for the agent, shown in the status line
and written to `history_dumps.jsonl`.

## Model config and ids

`ModelConfig`/`ModelKind` live in `model_config.rs`. `generate_session_id`
produces the `ses_…` shape (12 hex + 14 base62) used for routing/caching when
`SESSION_ID` is absent. One id per process keeps caching effective.

## Connector example

`crates/llm_connector_example` is a minimal CLI that sends one prompt through
the selected provider and prints streamed output. It is the fastest way to
check a new endpoint without the Bevy UI.
