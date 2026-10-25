//! Tool-call dispatch, async completion routing, and nested host calls.
//!
//! Extracted from `agent.rs`; these systems all consume `ToolRounds` and the
//! tool registry, so they change together.

use super::*;

/// Dispatch a finished assistant turn's tool calls.
///
/// World tools run immediately. JavaScript tools become [`JsTask`]s on the
/// owning context thread, so this system never blocks on a Boa heap. Results
/// are collected by [`poll_js_events`] and committed by [`apply_tool_results`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_tool_calls(
    mut requests: MessageReader<RunToolCalls>,
    mut world: ToolWorld,
    js_handles: Query<&JsRuntimeHandle>,
    script: Res<ScriptRuntime>,
    sources: Res<ToolSourceCache>,
    policy: Res<AutoBackgroundPolicy>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
) {
    for request in requests.read() {
        let Ok(agent) = world.agents.get(request.agent) else {
            continue;
        };
        if agent.current_turn != Some(request.turn_id) || request.tool_call.is_empty() {
            continue;
        }
        let source_name = agent.name().to_owned();

        let round_id = rounds.next_round;
        rounds.next_round += 1;

        let mut results: Vec<Option<ToolCallResult>> =
            (0..request.tool_call.len()).map(|_| None).collect();
        let mut remaining = request.tool_call.len();

        for (call_index, call) in request.tool_call.iter().enumerate() {
            let raw_arguments = call.toolcall_params.trim();
            let arguments = if raw_arguments.is_empty() {
                Value::Null
            } else {
                match serde_json::from_str::<Value>(raw_arguments) {
                    Ok(value) => value,
                    Err(err) => {
                        results[call_index] = Some(ToolCallResult {
                            result: format!(
                                "invalid JSON arguments for `{}`: {err}",
                                call.tool_name
                            ),
                            is_error: true,
                            tool_use_id: call.tool_use_id.clone(),
                        });
                        remaining -= 1;
                        continue;
                    }
                }
            };
            let root = (round_id << 32) | call_index as u64;
            rounds.open_budget(root, request.agent);
            let context = CallContext {
                source_entity: request.agent,
                source_agent: &source_name,
                depth: 0,
                root,
            };

            let found = find(&world, &call.tool_name);
            if let Some(reason) = found.and_then(|found| found.disabled) {
                results[call_index] = Some(ToolCallResult {
                    result: format!("Tool `{}` is disabled: {reason}", call.tool_name),
                    is_error: true,
                    tool_use_id: call.tool_use_id.clone(),
                });
                remaining -= 1;
                continue;
            }
            match found {
                Some(found) if matches!(found.execution, Execution::Interpreter) => {
                    let auto = resolve_auto_background(
                        background_from_flag(arguments.get("run_in_background")),
                        found.auto_background,
                        policy.auto_after,
                    );
                    if !rounds.spend_budget(root) {
                        results[call_index] = Some(ToolCallResult {
                            result: budget_exhausted_message(),
                            is_error: true,
                            tool_use_id: call.tool_use_id.clone(),
                        });
                        remaining -= 1;
                        continue;
                    }
                    let submitted =
                        interpreter_task(&call.tool_name, &arguments, &world.fs, 1, root).and_then(
                            |task| {
                                js_handles
                                    .get(request.agent)
                                    .map_err(|_| {
                                        format!("agent {} has no JavaScript runtime", request.agent)
                                    })?
                                    .submit(task)
                            },
                        );
                    match submitted {
                        Ok(request_id) if auto == Some(Duration::ZERO) => {
                            let task_id = rounds.detach(request_id, request.agent, root);
                            results[call_index] = Some(ToolCallResult {
                                result: background_started_message(task_id),
                                is_error: false,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                        Ok(request_id) => rounds.track(
                            request_id,
                            PendingCallFields {
                                round_id,
                                call_index,
                                auto_background: auto,
                            },
                        ),
                        Err(err) => {
                            rounds.refund_budget(root);
                            results[call_index] = Some(ToolCallResult {
                                result: err,
                                is_error: true,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                    }
                }
                Some(found) if matches!(found.execution, Execution::Gateway) => {
                    match dispatch_gateway(
                        &mut world,
                        &js_handles,
                        &script,
                        &sources,
                        &policy,
                        &mut rounds,
                        request.agent,
                        &source_name,
                        &arguments,
                        root,
                    ) {
                        GatewayDispatch::Done { message, is_error } => {
                            results[call_index] = Some(ToolCallResult {
                                result: message,
                                is_error,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                        GatewayDispatch::Pending {
                            request: target,
                            auto_background,
                        } => rounds.track(
                            target,
                            PendingCallFields {
                                round_id,
                                call_index,
                                auto_background,
                            },
                        ),
                    }
                }
                Some(found) if matches!(found.execution, Execution::Script) => {
                    if let Err(err) = validate_script_arguments(&world, &call.tool_name, &arguments)
                    {
                        results[call_index] = Some(ToolCallResult {
                            result: err,
                            is_error: true,
                            tool_use_id: call.tool_use_id.clone(),
                        });
                        remaining -= 1;
                        continue;
                    }
                    let auto = resolve_auto_background(
                        AutoBackground::Default,
                        found.auto_background,
                        policy.auto_after,
                    );
                    let spec = world.scripts.get(found.entity).cloned();
                    match spec {
                        Ok(spec) => {
                            if !rounds.spend_budget(root) {
                                results[call_index] = Some(ToolCallResult {
                                    result: budget_exhausted_message(),
                                    is_error: true,
                                    tool_use_id: call.tool_use_id.clone(),
                                });
                                remaining -= 1;
                                continue;
                            }
                            let submitted = script.submit(
                                request.agent,
                                ScriptTask {
                                    sources: sources.0.clone(),
                                    entry: spec.entry,
                                    export: spec.export,
                                    arguments: arguments.clone(),
                                    depth: 1,
                                    root,
                                },
                            );
                            match submitted {
                                Ok(request_id) if auto == Some(Duration::ZERO) => {
                                    let task_id = rounds.detach(request_id, request.agent, root);
                                    results[call_index] = Some(ToolCallResult {
                                        result: background_started_message(task_id),
                                        is_error: false,
                                        tool_use_id: call.tool_use_id.clone(),
                                    });
                                    remaining -= 1;
                                }
                                Ok(request_id) => rounds.track(
                                    request_id,
                                    PendingCallFields {
                                        round_id,
                                        call_index,
                                        auto_background: auto,
                                    },
                                ),
                                Err(err) => {
                                    rounds.refund_budget(root);
                                    results[call_index] = Some(ToolCallResult {
                                        result: err,
                                        is_error: true,
                                        tool_use_id: call.tool_use_id.clone(),
                                    });
                                    remaining -= 1;
                                }
                            }
                        }
                        Err(_) => {
                            results[call_index] = Some(ToolCallResult {
                                result: format!(
                                    "script tool `{}` is missing its execution data",
                                    call.tool_name
                                ),
                                is_error: true,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                    }
                }
                _ => {
                    let (result, is_error) =
                        execute_tool(&mut world, &call.tool_name, &arguments, &context);
                    results[call_index] = Some(ToolCallResult {
                        result,
                        is_error,
                        tool_use_id: call.tool_use_id.clone(),
                    });
                    remaining -= 1;
                }
            }
        }

        rounds.insert_round(
            round_id,
            PendingToolRound {
                agent: request.agent,
                turn_id: request.turn_id,
                calls: request.tool_call.clone(),
                results,
                remaining,
            },
        );
        finish_round_if_ready(&mut rounds, round_id, &mut finished);
    }
}

/// Drain completions from all context threads and finish any round that now
/// has every result.
pub(crate) fn poll_js_events(
    channels: Res<JsChannels>,
    js_handles: Query<&JsRuntimeHandle>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
    mut task_results: MessageWriter<TaskResult>,
) {
    for event in channels.drain_events() {
        let (agent, request, result) = match event {
            JsEvent::Started { request, at } => {
                rounds.mark_started(request, at);
                continue;
            }
            JsEvent::Done {
                agent,
                request,
                result,
            } => (agent, request, result),
        };

        route_async_result(
            &mut rounds,
            request,
            agent,
            result,
            &js_handles,
            &mut task_results,
            &mut finished,
        );
    }
}

/// Route one completed async request: detached task, blocked nested caller, or
/// a slot in an active round.
fn route_async_result(
    rounds: &mut ToolRounds,
    request: u64,
    agent: Entity,
    result: Result<String, String>,
    js_handles: &Query<&JsRuntimeHandle>,
    task_results: &mut MessageWriter<TaskResult>,
    finished: &mut MessageWriter<ToolRoundFinished>,
) {
    if let Some(task) = rounds.take_detached(request) {
        rounds.release_budget(task.root);
        let (text, is_error) = match result {
            Ok(text) => (text, false),
            Err(err) => (err, true),
        };
        task_results.write(TaskResult {
            agent: task.agent,
            task_id: task.task_id,
            result: text,
            is_error,
        });
        return;
    }

    if let Some(mut nested) = rounds.take_nested(request) {
        if let Some(reply) = nested.take_reply() {
            send_nested_reply(js_handles, reply, result);
        }
        return;
    }

    complete_tool_request(rounds, request, agent, result, finished);
}

/// Drain completions from script workers.
pub(crate) fn poll_script_events(
    runtime: Res<ScriptRuntime>,
    js_handles: Query<&JsRuntimeHandle>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
    mut task_results: MessageWriter<TaskResult>,
) {
    for event in runtime.drain_events() {
        match event {
            crate::tools::script::ScriptEvent::Started { request, at } => {
                rounds.mark_started(request, at);
            }
            crate::tools::script::ScriptEvent::Done {
                agent,
                request,
                result,
            } => route_async_result(
                &mut rounds,
                request,
                agent,
                result,
                &js_handles,
                &mut task_results,
                &mut finished,
            ),
        }
    }
}

/// Result of calling the model-facing `execute` gateway.
enum GatewayDispatch {
    Done {
        message: String,
        is_error: bool,
    },
    /// An async target was submitted; its result fills the gateway's round slot.
    Pending {
        request: u64,
        auto_background: Option<Duration>,
    },
}

fn gateway_error(message: impl Into<String>) -> GatewayDispatch {
    GatewayDispatch::Done {
        message: message.into(),
        is_error: true,
    }
}

/// Parse `execute(tool_name, args, background_after)`.
pub(super) fn parse_gateway_arguments(
    arguments: &Value,
) -> Result<(String, Value, AutoBackground), String> {
    let Some(target) = arguments.get("tool_name").and_then(Value::as_str) else {
        return Err("`execute` requires a `tool_name` string".to_owned());
    };
    let target_arguments = arguments.get("args").cloned().unwrap_or(Value::Null);
    let background = match arguments.get("background_after") {
        None | Some(Value::Null) => AutoBackground::Default,
        Some(value) => {
            let Some(seconds) = value.as_f64() else {
                return Err("`background_after` must be a number of seconds".to_owned());
            };
            parse_background_after(Some(seconds))?
        }
    };
    Ok((target.to_owned(), target_arguments, background))
}

/// Call another tool by name on behalf of the model.
///
/// This is the model-facing twin of the JavaScript `execute()` global: same
/// argument shape, same background semantics, same depth/budget rules.
#[allow(clippy::too_many_arguments)]
fn dispatch_gateway(
    world: &mut ToolWorld<'_, '_>,
    js_handles: &Query<&JsRuntimeHandle>,
    script: &ScriptRuntime,
    sources: &ToolSourceCache,
    policy: &AutoBackgroundPolicy,
    rounds: &mut ToolRounds,
    agent: Entity,
    source_name: &str,
    arguments: &Value,
    root: u64,
) -> GatewayDispatch {
    let (target_name, target_arguments, background) = match parse_gateway_arguments(arguments) {
        Ok(parsed) => parsed,
        Err(err) => return gateway_error(err),
    };

    let context = CallContext {
        source_entity: agent,
        source_agent: source_name,
        depth: 1,
        root,
    };

    let Some(found) = find(world, &target_name) else {
        return gateway_error(format!("Unknown tool: {target_name}"));
    };
    if let Some(reason) = found.disabled {
        return gateway_error(format!("Tool `{target_name}` is disabled: {reason}"));
    }
    if matches!(found.execution, Execution::Gateway) {
        return gateway_error("`execute` cannot call `execute`");
    }
    if !rounds.spend_budget(root) {
        return gateway_error(budget_exhausted_message());
    }

    let auto = resolve_auto_background(background, found.auto_background, policy.auto_after);
    match found.execution {
        Execution::Inline(_) => {
            if !matches!(background, AutoBackground::Default | AutoBackground::Never) {
                rounds.refund_budget(root);
                return gateway_error(format!("tool `{target_name}` cannot run in the background"));
            }
            let (message, is_error) =
                execute_tool(world, &target_name, &target_arguments, &context);
            GatewayDispatch::Done { message, is_error }
        }
        Execution::Interpreter => {
            let submitted = interpreter_task(&target_name, &target_arguments, &world.fs, 2, root)
                .and_then(|task| {
                    js_handles
                        .get(agent)
                        .map_err(|_| format!("agent {agent} has no JavaScript runtime"))?
                        .submit(task)
                });
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    gateway_error(err)
                }
                Ok(request) if auto == Some(Duration::ZERO) => {
                    let task_id = rounds.detach(request, agent, root);
                    GatewayDispatch::Done {
                        message: background_started_message(task_id),
                        is_error: false,
                    }
                }
                Ok(request) => GatewayDispatch::Pending {
                    request,
                    auto_background: auto,
                },
            }
        }
        Execution::Script => {
            if let Err(err) = validate_script_arguments(world, &target_name, &target_arguments) {
                rounds.refund_budget(root);
                return gateway_error(err);
            }
            let Ok(spec) = world.scripts.get(found.entity).cloned() else {
                rounds.refund_budget(root);
                return gateway_error(format!(
                    "script tool `{target_name}` is missing its execution data"
                ));
            };
            let submitted = script.submit(
                agent,
                ScriptTask {
                    sources: sources.0.clone(),
                    entry: spec.entry,
                    export: spec.export,
                    arguments: target_arguments,
                    depth: 2,
                    root,
                },
            );
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    gateway_error(err)
                }
                Ok(request) if auto == Some(Duration::ZERO) => {
                    let task_id = rounds.detach(request, agent, root);
                    GatewayDispatch::Done {
                        message: background_started_message(task_id),
                        is_error: false,
                    }
                }
                Ok(request) => GatewayDispatch::Pending {
                    request,
                    auto_background: auto,
                },
            }
        }
        Execution::Gateway => unreachable!("checked above"),
        Execution::Missing => {
            rounds.refund_budget(root);
            gateway_error(format!("tool `{target_name}` has no registered handler"))
        }
    }
}

/// Which context is asking the main loop to run a tool.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostOrigin {
    /// The agent's persistent interpreter thread.
    AgentInterpreter,
    /// A script worker with a throwaway context.
    Script,
}

/// Outcome of a host tool call.
enum HostOutcome {
    /// The reply was already sent.
    Immediate,
    /// Registered as a nested/detached call; the reply is sent on completion.
    Pending,
}

/// Send an immediate host reply to whichever caller is waiting.
fn reply_now(
    js_handles: &Query<&JsRuntimeHandle>,
    reply: NestedReply,
    result: Result<String, String>,
) -> HostOutcome {
    send_nested_reply(js_handles, reply, result);
    HostOutcome::Immediate
}

/// Shared handling for `execute(...)` from JavaScript and script workers.
#[allow(clippy::too_many_arguments)]
fn handle_host_tool(
    world: &mut ToolWorld<'_, '_>,
    rounds: &mut ToolRounds,
    js_handles: &Query<&JsRuntimeHandle>,
    script: &ScriptRuntime,
    sources: &ToolSourceCache,
    policy: &AutoBackgroundPolicy,
    agent: Entity,
    name: &str,
    arguments: &Value,
    depth: u32,
    root: u64,
    background: Option<f64>,
    origin: HostOrigin,
    reply: NestedReply,
) -> HostOutcome {
    if depth > MAX_TOOL_DEPTH {
        return reply_now(
            js_handles,
            reply,
            Err(format!(
                "tool call nesting limit ({MAX_TOOL_DEPTH}) reached"
            )),
        );
    }
    let request = match parse_background_after(background) {
        Ok(request) => request,
        Err(err) => return reply_now(js_handles, reply, Err(err)),
    };
    if !matches!(request, AutoBackground::Default | AutoBackground::Never) {
        return reply_now(
            js_handles,
            reply,
            Err("only a model-level tool call can detach into the background".to_owned()),
        );
    }
    let source_name = world
        .agents
        .get(agent)
        .map(|agent| agent.name().to_owned())
        .unwrap_or_else(|_| "agent".to_owned());
    let context = CallContext {
        source_entity: agent,
        source_agent: &source_name,
        depth,
        root,
    };

    let Some(found) = find(world, name) else {
        return reply_now(js_handles, reply, Err(format!("Unknown tool: {name}")));
    };
    if let Some(disabled) = found.disabled {
        return reply_now(
            js_handles,
            reply,
            Err(format!("Tool `{name}` is disabled: {disabled}")),
        );
    }
    if found.model_only {
        return reply_now(
            js_handles,
            reply,
            Err(format!(
                "tool `{name}` can only be called by the model, not from JavaScript or scripts"
            )),
        );
    }

    // Nested calls always run synchronously from the caller's point of view.
    let _ = policy;
    match found.execution {
        Execution::Inline(_) => {
            if !matches!(request, AutoBackground::Default | AutoBackground::Never) {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!("tool `{name}` cannot run in the background")),
                );
            }
            if !rounds.spend_budget(root) {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!(
                        "tool call budget for this root call is exhausted ({})",
                        crate::tools::schedule::MAX_CALLS_PER_ROOT
                    )),
                );
            }
            let (text, is_error) = execute_tool(world, name, arguments, &context);
            let result = if is_error { Err(text) } else { Ok(text) };
            reply_now(js_handles, reply, result)
        }
        Execution::Interpreter => {
            // The agent has exactly one interpreter thread. Submitting to it
            // while that thread is blocked in `execute()` would never run.
            if origin == HostOrigin::AgentInterpreter {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!(
                        "JavaScript cannot call another JavaScript tool (`{name}`)"
                    )),
                );
            }
            if !rounds.spend_budget(root) {
                return reply_now(js_handles, reply, Err(budget_exhausted_message()));
            }
            let submitted =
                interpreter_task(name, arguments, &world.fs, depth + 1, root).and_then(|task| {
                    js_handles
                        .get(agent)
                        .map_err(|_| format!("agent {agent} has no JavaScript runtime"))?
                        .submit(task)
                });
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    reply_now(js_handles, reply, Err(err))
                }
                Ok(request_id) => {
                    rounds.add_nested(NestedCall::new(agent, request_id, reply));
                    HostOutcome::Pending
                }
            }
        }
        Execution::Gateway => reply_now(
            js_handles,
            reply,
            Err(format!("tool `{name}` can only be called by the model")),
        ),
        Execution::Missing => reply_now(
            js_handles,
            reply,
            Err(format!("tool `{name}` has no registered handler")),
        ),
        Execution::Script => {
            if let Err(err) = validate_script_arguments(world, name, arguments) {
                return reply_now(js_handles, reply, Err(err));
            }
            let Ok(spec) = world.scripts.get(found.entity).cloned() else {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!(
                        "script tool `{name}` is missing its execution data"
                    )),
                );
            };
            if !rounds.spend_budget(root) {
                return reply_now(js_handles, reply, Err(budget_exhausted_message()));
            }
            let submitted = script.submit(
                agent,
                ScriptTask {
                    sources: sources.0.clone(),
                    entry: spec.entry,
                    export: spec.export,
                    arguments: arguments.clone(),
                    depth: depth + 1,
                    root,
                },
            );
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    reply_now(js_handles, reply, Err(err))
                }
                Ok(request_id) => {
                    rounds.add_nested(NestedCall::new(agent, request_id, reply));
                    HostOutcome::Pending
                }
            }
        }
    }
}

/// Handle `execute(...)` and module reads coming from script workers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn poll_script_host_requests(
    runtime: Res<ScriptRuntime>,
    mut world: ToolWorld,
    js_handles: Query<&JsRuntimeHandle>,
    sources: Res<ToolSourceCache>,
    policy: Res<AutoBackgroundPolicy>,
    mut rounds: ResMut<ToolRounds>,
) {
    for request in runtime.drain_host_requests() {
        let ScriptHostRequest {
            agent,
            command,
            reply,
        } = request;

        // Work from a removed agent must not keep mutating the world.
        if world.agents.get(agent).is_err() {
            let _ = reply.send(Err(format!("agent entity {agent} no longer exists")));
            continue;
        }

        match command {
            HostCommand::Tool {
                name,
                arguments,
                depth,
                root,
                background,
            } => {
                let _ = handle_host_tool(
                    &mut world,
                    &mut rounds,
                    &js_handles,
                    &runtime,
                    &sources,
                    &policy,
                    agent,
                    &name,
                    &arguments,
                    depth,
                    root,
                    background,
                    HostOrigin::Script,
                    NestedReply::Channel(reply),
                );
            }
            HostCommand::ReadFile { path } => {
                let _ = reply.send(world.fs.read(&path));
            }
        }
    }
}

/// Execute host commands coming from context threads: JavaScript `execute`
/// tool calls and virtual-filesystem reads for module imports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn poll_js_host_requests(
    channels: Res<JsChannels>,
    mut world: ToolWorld,
    js_handles: Query<&JsRuntimeHandle>,
    script: Res<ScriptRuntime>,
    sources: Res<ToolSourceCache>,
    policy: Res<AutoBackgroundPolicy>,
    mut rounds: ResMut<ToolRounds>,
) {
    for request in channels.drain_host_requests() {
        let HostRequest {
            agent,
            token,
            command,
        } = request;

        if world.agents.get(agent).is_err() {
            if let Ok(handle) = js_handles.get(agent) {
                handle.respond(token, Err(format!("agent entity {agent} no longer exists")));
            }
            continue;
        }

        match command {
            HostCommand::Tool {
                name,
                arguments,
                depth,
                root,
                background,
            } => {
                // The reply is delivered now or when the nested call completes;
                // the agent's interpreter stays blocked on `token`.
                let _ = handle_host_tool(
                    &mut world,
                    &mut rounds,
                    &js_handles,
                    &script,
                    &sources,
                    &policy,
                    agent,
                    &name,
                    &arguments,
                    depth,
                    root,
                    background,
                    HostOrigin::AgentInterpreter,
                    NestedReply::Agent { agent, token },
                );
            }
            HostCommand::ReadFile { path } => {
                if let Ok(handle) = js_handles.get(agent) {
                    handle.respond(token, world.fs.read(&path));
                }
            }
        }
    }
}

/// Commit a fully-resolved tool round to history and continue the agent loop.
pub(crate) fn apply_tool_results(
    mut finished: MessageReader<ToolRoundFinished>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut texts: Query<&mut Text>,
    mut start_turn: MessageWriter<StartAgentTurn>,
) {
    for message in finished.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(message.agent) else {
            continue;
        };
        if agent.current_turn != Some(message.turn_id) {
            continue;
        }

        // `end_agent` queues a removal; do not commit a new turn for an agent
        // that is about to be despawned.
        if message
            .calls
            .iter()
            .any(|call| call.tool_name == "end_agent")
        {
            continue;
        }

        agent.push_history(
            "tool_results",
            Entry::ToolCallResult {
                tool_call: message.results.clone(),
            },
        );

        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Tools finished; continuing...");

        // Force waiting user and cross-agent messages into the history now:
        // a completed tool round is a legal boundary, and leaving them queued
        // would delay them until the whole agent loop finishes.
        let inserted = flush_pending_messages(&mut agent, &mut view, &mut texts, true);
        if inserted > 0 {
            set_text(
                &mut texts,
                view.status,
                format!("Inserted {inserted} pending message(s); continuing..."),
            );
        }

        agent.current_turn = None;
        start_turn.write(StartAgentTurn {
            agent: message.agent,
        });
    }
}

fn budget_exhausted_message() -> String {
    format!(
        "tool call budget for this root call is exhausted ({})",
        crate::tools::schedule::MAX_CALLS_PER_ROOT
    )
}
