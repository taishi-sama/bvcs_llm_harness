# Scheduling: rounds, budgets, detach, cancellation

Dispatch is described in `docs/tool-calls.md`. This note focuses on the
bookkeeping that makes async calls safe.

## State

`ToolRounds` (resource, `src/tools/schedule.rs`):

| Map | Key | Meaning |
| --- | --- | --- |
| `active` | round id | one model tool batch; results indexed by call position |
| `pending` | request id | async call that still owns a round slot |
| `detached` | request id | call detached from its round; result becomes a message |
| `nested` | request id | `execute()` called by JS/script; caller is blocked |
| `budgets` | root id | remaining nested calls + reference count |

Request ids:

- interpreter: one shared `AtomicU64` across all agent contexts
- script: `1 << 63 | counter`

The namespaces cannot collide.

## Root identity and budget

- A model batch entry gets `root = (round_id << 32) | call_index`.
- Opening the root inserts `Budget { remaining: 2000, refs: 1 }`.
- Every nested call (gateway target, `execute` from JS/scripts) spends one unit
  **before** submission and refunds if submission fails.
- `refs` starts at 1 for the round slot. `detach` adds one per detached task.
  `finish_round_if_ready` releases the slot; `route_async_result` releases a
  detached task when its result arrives. The budget disappears at `refs == 0`.
- Consequence: background work keeps its budget across turns, which is what
  makes “detach a long eval, then keep using tools from it” work.

## Depth

- root call = 0
- a context created for a call at depth `d` makes calls at depth `d + 1`
- limit `MAX_TOOL_DEPTH = 50`, checked before dispatch/submission

## Detach

Detach is **not cancellation**. Only root calls can detach:

1. resolve the grace: per-call (`run_in_background` / `background_after`) →
   per-tool `auto_background` → `AutoBackgroundPolicy` (`JS_AUTO_BACKGROUND_MS`,
   default 5 s)
2. `Some(0)` → detach at dispatch
3. `Some(grace)` → `auto_background_long_calls` detaches after the grace has
   elapsed since `Started` (not since submission, so queued calls are not
   detached before they run)
4. detaching fills the round slot with a “Task N running in background” result,
   increments the root budget reference, and the real result later becomes a
   `TaskResult` → queued message

Nested calls cannot detach; `background_after` on nested `execute` is rejected.

## Completion

`route_async_result` is the single router for both interpreter and script
completions:

```
detached → release budget ref → TaskResult (message)
nested   → send_nested_reply (agent host token or script channel)
else     → complete_tool_request → fill slot → finish_round_if_ready
```

`finish_round_if_ready` emits `ToolRoundFinished` when `remaining == 0`; the
agent then appends results and starts the next model turn.

## Cancellation and removal

- `cancel_rounds(agent)`: drops that agent's rounds and pending calls, and its
  budgets. Late completions find no slot and are ignored.
- `remove_agent(agent)`: also drops detached and nested calls.
- Host requests from a removed agent are rejected before any tool runs.
- Work already executing on a Boa thread cannot be interrupted; it finishes and
  its result is discarded.

## Observability

`running_for(agent)` = unresolved round slots + nested calls + detached tasks.
The footer renders it through `update_tool_activity` as
`Currently running toolcalls: N`.

## Invariants

- results are written by `call_index`; never append
- a request id is completed exactly once (maps are removed on completion)
- spend before submit, refund on submit failure
- a detached task holds a budget reference until its result is routed
- do not detach a nested call; root calls only
