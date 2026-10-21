//! Provider-neutral token accounting.
//!
//! Every provider adapter normalizes its wire usage payload into [`Usage`].
//! Accounting is deliberately kept out of [`llm_history_model::history`]:
//! usage describes one API call, while history entries describe replayable
//! conversation state.
//!
//! Normalization rules:
//! - `input_tokens` is the **total** prompt token count, including cached and
//!   cache-creation tokens.
//! - `cached_input_tokens` and `reasoning_tokens` are **subsets** of input and
//!   output respectively, never additive.
//! - `cache_creation_input_tokens` is additional billed input (Anthropic only).
//! - `total_tokens` prefers the provider-reported value and otherwise derives
//!   from `input_tokens + output_tokens`.
//! - provider-specific numeric counters live in [`Usage::extra`].

use crate::anthropic::response::AnthropicUsage;
use crate::openai_completions::response::ChatUsage;
use crate::openai_responses::response::ResponsesUsage;
use llm_history_model::history::Entry;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Provider label attached to normalized usage reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageProvider {
    /// Unspecified or a compatible server we do not classify further.
    #[default]
    Other,
    OpenAI,
    Anthropic,
    DeepSeek,
    Zai,
}

/// Normalized usage for one provider request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Provider that produced this usage report.
    #[serde(default)]
    pub provider: UsageProvider,

    /// Total input/prompt tokens, including cached input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,

    /// Total output/completion tokens, including reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,

    /// Provider-reported total, or derived from input + output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,

    /// Subset of input served from cache (OpenAI/DeepSeek/Z.ai cache hits,
    /// Anthropic cache reads).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,

    /// Anthropic cache-write tokens. These are additional input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,

    /// Subset of output used for reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,

    /// Provider-specific numeric counters not covered above (for example
    /// DeepSeek cache hit/miss or Anthropic ephemeral-cache split).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, u64>,
}

/// Assistant-side history entries plus the usage of the call that produced
/// them. Usage intentionally lives outside [`Entry`] so history replay is not
/// coupled to billing data.
#[derive(Debug, Clone, Default)]
pub struct ParsedResponse {
    pub entries: Vec<Entry>,
    pub usage: Option<Usage>,
}

/// Accumulates usage across multiple requests (for example one session).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageTotals {
    /// Number of usage reports folded in.
    #[serde(default)]
    pub requests: u64,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub cached_input_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, u64>,
}

impl Usage {
    /// Provider-reported total, falling back to `input_tokens + output_tokens`
    /// when only one side is known.
    pub fn total(&self) -> Option<u64> {
        self.total_tokens
            .or_else(|| match (self.input_tokens, self.output_tokens) {
                (Some(input), Some(output)) => Some(input.saturating_add(output)),
                _ => None,
            })
    }

    /// Whether the report carries no counters at all.
    pub fn is_empty(&self) -> bool {
        self.input_tokens.is_none()
            && self.output_tokens.is_none()
            && self.total_tokens.is_none()
            && self.cached_input_tokens.is_none()
            && self.cache_creation_input_tokens.is_none()
            && self.reasoning_tokens.is_none()
            && self.extra.is_empty()
    }

    /// Tags this report with a provider label.
    pub fn with_provider(mut self, provider: UsageProvider) -> Self {
        self.provider = provider;
        self
    }

    /// Merges a partial (typically streaming) report into this one.
    ///
    /// Input-side counters fill missing values; output/total counters are
    /// overwritten because providers send their authoritative final counts in
    /// later stream events. `extra` entries are merged key by key.
    pub fn merge(&mut self, other: &Usage) {
        if self.provider == UsageProvider::Other && other.provider != UsageProvider::Other {
            self.provider = other.provider;
        }
        merge_replace(&mut self.output_tokens, other.output_tokens);
        merge_replace(&mut self.reasoning_tokens, other.reasoning_tokens);
        merge_fill(&mut self.input_tokens, other.input_tokens);
        merge_fill(&mut self.cached_input_tokens, other.cached_input_tokens);
        merge_fill(
            &mut self.cache_creation_input_tokens,
            other.cache_creation_input_tokens,
        );
        // A later stream event usually carries the final output count, so the
        // derived total must be recomputed unless the provider sent an
        // authoritative total in this report.
        if other.total_tokens.is_some() {
            self.total_tokens = other.total_tokens;
        } else if let (Some(input), Some(output)) = (self.input_tokens, self.output_tokens) {
            self.total_tokens = Some(input.saturating_add(output));
        }
        self.extra
            .extend(other.extra.iter().map(|(k, v)| (k.clone(), *v)));
    }

    /// Normalizes OpenAI Chat Completions usage (also used by DeepSeek and
    /// Z.ai, which share the wire format). Defaults to the [`UsageProvider::OpenAI`]
    /// label.
    pub fn from_chat(u: &ChatUsage) -> Option<Usage> {
        Self::from_chat_for(u, UsageProvider::OpenAI)
    }

    /// Normalizes Chat Completions usage with an explicit provider label.
    pub fn from_chat_for(u: &ChatUsage, provider: UsageProvider) -> Option<Usage> {
        let input_tokens = u.prompt_tokens;
        let output_tokens = u.completion_tokens;
        let total_tokens = u
            .total_tokens
            .or_else(|| derive_total(input_tokens, output_tokens));
        let cached_input_tokens = u
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .or(u.prompt_cache_hit_tokens);
        let reasoning_tokens = u
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens);

        let mut extra = numeric_extra(&u.extra);
        if let Some(hit) = u.prompt_cache_hit_tokens {
            extra.insert("prompt_cache_hit_tokens".to_owned(), hit);
        }
        if let Some(miss) = u.prompt_cache_miss_tokens {
            extra.insert("prompt_cache_miss_tokens".to_owned(), miss);
        }

        finish(
            provider,
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            None,
            reasoning_tokens,
            extra,
        )
    }

    /// Normalizes OpenAI Responses API usage. Defaults to
    /// [`UsageProvider::OpenAI`].
    pub fn from_responses(u: &ResponsesUsage) -> Option<Usage> {
        Self::from_responses_for(u, UsageProvider::OpenAI)
    }

    /// Normalizes Responses API usage with an explicit provider label.
    pub fn from_responses_for(u: &ResponsesUsage, provider: UsageProvider) -> Option<Usage> {
        let input_tokens = u.input_tokens;
        let output_tokens = u.output_tokens;
        let total_tokens = u
            .total_tokens
            .or_else(|| derive_total(input_tokens, output_tokens));
        let cached_input_tokens = u
            .input_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens);
        let reasoning_tokens = u
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens);

        finish(
            provider,
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            None,
            reasoning_tokens,
            numeric_extra(&u.extra),
        )
    }

    /// Normalizes Anthropic Messages API usage.
    ///
    /// Anthropic's `input_tokens` excludes cache reads and writes, so the
    /// normalized input is the sum of all three. The raw values are preserved
    /// in [`Usage::extra`].
    pub fn from_anthropic(u: &AnthropicUsage) -> Option<Usage> {
        let cache_creation_input_tokens = u.cache_creation_input_tokens;
        let cached_input_tokens = u.cache_read_input_tokens;
        let input_tokens = match (
            u.input_tokens,
            cache_creation_input_tokens,
            cached_input_tokens,
        ) {
            (None, None, None) => None,
            (input, creation, read) => Some(
                input
                    .unwrap_or(0)
                    .saturating_add(creation.unwrap_or(0))
                    .saturating_add(read.unwrap_or(0)),
            ),
        };
        let output_tokens = u.output_tokens;
        let total_tokens = derive_total(input_tokens, output_tokens);

        let mut extra = numeric_extra(&u.extra);
        if let Some(raw) = u.input_tokens {
            extra.insert("raw_input_tokens".to_owned(), raw);
        }
        if let Some(creation) = &u.cache_creation {
            if let Some(tokens) = creation.ephemeral_5m_input_tokens {
                extra.insert("ephemeral_5m_input_tokens".to_owned(), tokens);
            }
            if let Some(tokens) = creation.ephemeral_1h_input_tokens {
                extra.insert("ephemeral_1h_input_tokens".to_owned(), tokens);
            }
        }

        finish(
            UsageProvider::Anthropic,
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            None,
            extra,
        )
    }
}

impl UsageTotals {
    /// Folds one report into the running totals.
    pub fn add(&mut self, usage: &Usage) {
        self.requests = self.requests.saturating_add(1);
        self.input_tokens = self
            .input_tokens
            .saturating_add(usage.input_tokens.unwrap_or(0));
        self.output_tokens = self
            .output_tokens
            .saturating_add(usage.output_tokens.unwrap_or(0));
        self.total_tokens = self.total_tokens.saturating_add(usage.total().unwrap_or(0));
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(usage.cached_input_tokens.unwrap_or(0));
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(usage.cache_creation_input_tokens.unwrap_or(0));
        self.reasoning_tokens = self
            .reasoning_tokens
            .saturating_add(usage.reasoning_tokens.unwrap_or(0));
        for (key, value) in &usage.extra {
            let entry = self.extra.entry(key.clone()).or_insert(0);
            *entry = entry.saturating_add(*value);
        }
    }
}

fn derive_total(input: Option<u64>, output: Option<u64>) -> Option<u64> {
    match (input, output) {
        (Some(input), Some(output)) => Some(input.saturating_add(output)),
        _ => None,
    }
}

fn merge_fill(slot: &mut Option<u64>, value: Option<u64>) {
    if slot.is_none() {
        *slot = value;
    }
}

fn merge_replace(slot: &mut Option<u64>, value: Option<u64>) {
    if value.is_some() {
        *slot = value;
    }
}

fn numeric_extra(extra: &BTreeMap<String, Value>) -> BTreeMap<String, u64> {
    extra
        .iter()
        .filter_map(|(key, value)| value.as_u64().map(|value| (key.clone(), value)))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn finish(
    provider: UsageProvider,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cached_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
    extra: BTreeMap<String, u64>,
) -> Option<Usage> {
    let usage = Usage {
        provider,
        input_tokens,
        output_tokens,
        total_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
        reasoning_tokens,
        extra,
    };
    (!usage.is_empty()).then_some(usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(value: Value) -> ChatUsage {
        serde_json::from_value(value).expect("valid chat usage")
    }

    fn responses(value: Value) -> ResponsesUsage {
        serde_json::from_value(value).expect("valid responses usage")
    }

    fn anthropic(value: Value) -> AnthropicUsage {
        serde_json::from_value(value).expect("valid anthropic usage")
    }

    #[test]
    fn chat_openai_basic() {
        let usage = Usage::from_chat(&chat(serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14
        })))
        .expect("usage");

        assert_eq!(usage.provider, UsageProvider::OpenAI);
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(4));
        assert_eq!(usage.total(), Some(14));
        assert!(usage.cached_input_tokens.is_none());
    }

    #[test]
    fn chat_openai_cached_and_reasoning_details() {
        let usage = Usage::from_chat(&chat(serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 20}
        })))
        .expect("usage");

        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.cached_input_tokens, Some(40));
        assert_eq!(usage.reasoning_tokens, Some(20));
        assert_eq!(usage.total(), Some(150));
    }

    #[test]
    fn chat_deepseek_cache_hit_and_miss() {
        let usage = Usage::from_chat_for(
            &chat(serde_json::json!({
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 16
            })),
            UsageProvider::DeepSeek,
        )
        .expect("usage");

        assert_eq!(usage.provider, UsageProvider::DeepSeek);
        assert_eq!(usage.input_tokens, Some(80));
        assert_eq!(usage.cached_input_tokens, Some(64));
        assert_eq!(usage.extra.get("prompt_cache_miss_tokens"), Some(&16));
        assert_eq!(usage.total(), Some(100));
    }

    #[test]
    fn chat_zai_cached_tokens() {
        let usage = Usage::from_chat_for(
            &chat(serde_json::json!({
                "prompt_tokens": 30,
                "completion_tokens": 7,
                "total_tokens": 37,
                "prompt_tokens_details": {"cached_tokens": 12}
            })),
            UsageProvider::Zai,
        )
        .expect("usage");

        assert_eq!(usage.provider, UsageProvider::Zai);
        assert_eq!(usage.cached_input_tokens, Some(12));
    }

    #[test]
    fn unknown_numeric_chat_details_land_in_extra() {
        let usage = Usage::from_chat(&chat(serde_json::json!({
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "audio_tokens": 3,
            "note": "ignored"
        })))
        .expect("usage");

        assert_eq!(usage.extra.get("audio_tokens"), Some(&3));
        assert!(!usage.extra.contains_key("note"));
    }

    #[test]
    fn responses_basic_with_details() {
        let usage = Usage::from_responses(&responses(serde_json::json!({
            "input_tokens": 21,
            "output_tokens": 247,
            "total_tokens": 268,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 100}
        })))
        .expect("usage");

        assert_eq!(usage.input_tokens, Some(21));
        assert_eq!(usage.output_tokens, Some(247));
        assert_eq!(usage.cached_input_tokens, Some(5));
        assert_eq!(usage.reasoning_tokens, Some(100));
        assert_eq!(usage.total(), Some(268));
    }

    #[test]
    fn anthropic_sums_cache_inputs() {
        let usage = Usage::from_anthropic(&anthropic(serde_json::json!({
            "input_tokens": 17,
            "output_tokens": 186,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 3,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 7,
                "ephemeral_1h_input_tokens": 3
            }
        })))
        .expect("usage");

        assert_eq!(usage.provider, UsageProvider::Anthropic);
        assert_eq!(usage.input_tokens, Some(30));
        assert_eq!(usage.output_tokens, Some(186));
        assert_eq!(usage.cached_input_tokens, Some(3));
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
        assert_eq!(usage.total(), Some(216));
        assert_eq!(usage.extra.get("raw_input_tokens"), Some(&17));
        assert_eq!(usage.extra.get("ephemeral_5m_input_tokens"), Some(&7));
        assert_eq!(usage.extra.get("ephemeral_1h_input_tokens"), Some(&3));
    }

    #[test]
    fn total_derives_when_absent() {
        let usage = Usage::from_chat(&chat(serde_json::json!({
            "prompt_tokens": 3,
            "completion_tokens": 4
        })))
        .expect("usage");
        assert_eq!(usage.total(), Some(7));
    }

    #[test]
    fn empty_payload_normalizes_to_none() {
        assert!(Usage::from_chat(&chat(serde_json::json!({}))).is_none());
        assert!(Usage::from_responses(&responses(serde_json::json!({}))).is_none());
        assert!(Usage::from_anthropic(&anthropic(serde_json::json!({}))).is_none());
    }

    #[test]
    fn merge_fills_inputs_and_overwrites_outputs() {
        let mut usage = Usage::from_anthropic(&anthropic(serde_json::json!({
            "input_tokens": 17,
            "cache_read_input_tokens": 2,
            "output_tokens": 0
        })))
        .expect("start usage");
        usage.merge(
            &Usage::from_anthropic(&anthropic(serde_json::json!({
                "output_tokens": 186
            })))
            .expect("delta usage"),
        );

        assert_eq!(usage.input_tokens, Some(19));
        assert_eq!(usage.cached_input_tokens, Some(2));
        assert_eq!(usage.output_tokens, Some(186));
        assert_eq!(usage.total(), Some(205));
    }

    #[test]
    fn totals_accumulate() {
        let mut totals = UsageTotals::default();
        totals.add(
            &Usage::from_chat(&chat(serde_json::json!({
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            })))
            .unwrap(),
        );
        totals.add(
            &Usage::from_chat(&chat(serde_json::json!({
                "prompt_tokens": 3,
                "completion_tokens": 2
            })))
            .unwrap(),
        );

        assert_eq!(totals.requests, 2);
        assert_eq!(totals.input_tokens, 13);
        assert_eq!(totals.output_tokens, 7);
        assert_eq!(totals.total_tokens, 20);
    }
}
