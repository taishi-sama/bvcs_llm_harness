//! HTTP streaming transport and provider request bodies.

use bevy::prelude::Entity;
use bevy_tokio_tasks::TaskContext;
use futures_util::StreamExt as _;
use llm_api::{anthropic, model_config::ModelConfig, openai_completions, openai_responses};
use llm_history_model::history::Entry;
use tokio::sync::watch;

use crate::{
    config::Provider,
    messages::{StreamChunk, StreamEnded, StreamFailed},
};

// ---------------------------------------------------------------------------
// Transport task
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) async fn stream_transport(
    mut ctx: TaskContext,
    mut cancel: watch::Receiver<bool>,
    agent: Entity,
    turn_id: u64,
    endpoint: String,
    token: Option<String>,
    session_id: String,
    body: String,
) {
    if let Err(err) = stream_request(
        &mut ctx,
        &mut cancel,
        agent,
        turn_id,
        endpoint,
        token,
        session_id,
        body,
    )
    .await
    {
        let message = format!("{err:#}");
        ctx.run_on_main_thread(move |ctx| {
            ctx.world.write_message(StreamFailed {
                agent,
                turn_id,
                message,
            });
        })
        .await;
    }
}

#[allow(clippy::too_many_arguments)]
async fn stream_request(
    ctx: &mut TaskContext,
    cancel: &mut watch::Receiver<bool>,
    agent: Entity,
    turn_id: u64,
    endpoint: String,
    token: Option<String>,
    session_id: String,
    body: String,
) -> Result<(), anyhow::Error> {
    let client = reqwest::Client::new();
    let mut request = client
        .post(&endpoint)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::USER_AGENT, "bvcs_llm_harness/0.1")
        .header("x-opencode-session", session_id)
        .body(body);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let res = request.send().await?;
    if !res.status().is_success() {
        anyhow::bail!("HTTP {}: {}", res.status(), res.text().await?);
    }

    let mut stream = res.bytes_stream();
    loop {
        if *cancel.borrow() {
            return Ok(());
        }

        let next = tokio::select! {
            next = stream.next() => next,
            changed = cancel.changed() => {
                let _ = changed;
                return Ok(());
            }
        };

        let Some(chunk) = next else {
            break;
        };
        let bytes = chunk?.to_vec();
        ctx.run_on_main_thread(move |ctx| {
            ctx.world.write_message(StreamChunk {
                agent,
                turn_id,
                bytes,
            });
        })
        .await;

        if *cancel.borrow() {
            return Ok(());
        }
    }

    if *cancel.borrow() {
        return Ok(());
    }

    ctx.run_on_main_thread(move |ctx| {
        ctx.world.write_message(StreamEnded { agent, turn_id });
    })
    .await;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn request_body(
    provider: Provider,
    history: &[Entry],
    config: &ModelConfig,
) -> Result<String, anyhow::Error> {
    Ok(match provider {
        Provider::Anthropic => serde_json::to_string(
            &anthropic::to_anthropic_messages(history, config)
                .map_err(|errors| anyhow::anyhow!("history validation failed: {errors:?}"))?,
        )?,
        Provider::Completions => serde_json::to_string(
            &openai_completions::to_completions_input(history, config)
                .map_err(|errors| anyhow::anyhow!("history validation failed: {errors:?}"))?,
        )?,
        Provider::Responses => serde_json::to_string(
            &openai_responses::to_responses_input(history, config)
                .map_err(|errors| anyhow::anyhow!("history validation failed: {errors:?}"))?,
        )?,
    })
}
