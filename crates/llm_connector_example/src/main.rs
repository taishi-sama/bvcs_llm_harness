//! Exercises server-side streaming against a local llama.cpp server.
//!
//! Select the API with `PROVIDER`:
//!
//! ```text
//! PROVIDER=anthropic    cargo run -p llm_connector_example  # /v1/messages
//! PROVIDER=completions  cargo run -p llm_connector_example  # /v1/chat/completions
//! PROVIDER=responses    cargo run -p llm_connector_example  # /v1/responses
//! ```

use std::{
    cell::RefCell,
    collections::HashMap,
    fs,
    io::{self, Write},
};

use boa_engine::{Context, JsResult, JsValue, NativeFunction, Source};
use http_body_util::BodyExt as _;
use hyper::{Request, client::conn::http1};
use hyper_util::rt::TokioIo;
use llm_api::{
    anthropic,
    generic_tool::{ParameterDescription, ParameterType, Tool},
    model_config::{GlobalConfig, ModelConfig, ModelKind},
    openai_completions, openai_responses,
    stream::{StreamError, StreamEvent, finish_draft_for},
};
use llm_history_model::history::{DraftBlock, Entry, MessageContent, ToolCallResult};
use serde_json::Value;
use tokio::net::TcpStream;

/// Which server-side streaming API to exercise.
#[derive(Clone, Copy, Debug)]
enum Provider {
    Anthropic,
    Completions,
    Responses,
}

impl Provider {
    fn from_env() -> Self {
        match std::env::var("PROVIDER").as_deref() {
            Ok("completions") => Self::Completions,
            Ok("responses") => Self::Responses,
            _ => Self::Anthropic,
        }
    }

    fn endpoint(self) -> &'static str {
        match self {
            Provider::Anthropic => "http://127.0.0.1:8080/v1/messages",
            Provider::Completions => "http://127.0.0.1:8080/v1/chat/completions",
            Provider::Responses => "http://127.0.0.1:8080/v1/responses",
        }
    }

    fn kind(self, model_name: &str) -> ModelKind {
        match self {
            Provider::Anthropic => ModelKind::AnthropicCompatible {
                model_name: model_name.into(),
                streaming: true,
                thinking: None,
                // llama.cpp returns thinking blocks with `signature: ""`.
                allow_unsigned_thinking: true,
            },
            Provider::Completions | Provider::Responses => ModelKind::OpenAICompatible {
                model_name: model_name.into(),
                streaming: true,
                prompt_cache_key: String::new(),
            },
        }
    }
}

/// Uniform wrapper over the three provider stream decoders.
enum Decoder {
    Anthropic(anthropic::stream::StreamDecoder),
    Completions(openai_completions::stream::StreamDecoder),
    Responses(openai_responses::stream::StreamDecoder),
}

impl Decoder {
    fn new(provider: Provider) -> Self {
        match provider {
            Provider::Anthropic => Self::Anthropic(anthropic::stream::StreamDecoder::new()),
            Provider::Completions => {
                Self::Completions(openai_completions::stream::StreamDecoder::new())
            }
            Provider::Responses => Self::Responses(openai_responses::stream::StreamDecoder::new()),
        }
    }

    fn push_bytes(
        &mut self,
        bytes: &[u8],
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, StreamError> {
        match self {
            Decoder::Anthropic(decoder) => decoder.push_bytes(bytes, blocks),
            Decoder::Completions(decoder) => decoder.push_bytes(bytes, blocks),
            Decoder::Responses(decoder) => decoder.push_bytes(bytes, blocks),
        }
    }

    fn finish(&mut self, blocks: &mut Vec<DraftBlock>) -> Result<Vec<StreamEvent>, StreamError> {
        match self {
            Decoder::Anthropic(decoder) => decoder.finish(blocks),
            Decoder::Completions(decoder) => decoder.finish(blocks),
            Decoder::Responses(decoder) => decoder.finish(blocks),
        }
    }
}

thread_local! {
    /// Lines emitted by `print`/`console.log` during the current JS evaluation.
    static JS_OUTPUT: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn js_print(_this: &JsValue, args: &[JsValue], context: &mut Context) -> JsResult<JsValue> {
    let line = args
        .iter()
        .map(|value| {
            value
                .to_string(context)
                .map(|text| text.to_std_string_escaped())
                .unwrap_or_else(|_| value.display().to_string())
        })
        .collect::<Vec<_>>()
        .join(" ");
    JS_OUTPUT.with(|output| output.borrow_mut().push(line));
    Ok(JsValue::undefined())
}

/// Runs `code` in an embedded Boa JavaScript engine and returns everything the
/// snippet printed plus the value of its final expression.
fn eval_javascript(code: &str) -> Result<String, String> {
    JS_OUTPUT.with(|output| output.borrow_mut().clear());

    let mut context = Context::default();
    // Protect the host from runaway loops in model-generated code.
    context.runtime_limits_mut().set_loop_iteration_limit(10_000_000);
    context
        .register_global_builtin_callable(
            "print".into(),
            1,
            NativeFunction::from_fn_ptr(js_print),
        )
        .map_err(|err| format!("failed to initialize JavaScript runtime: {err}"))?;

    // Boa has no console object; expose the common logging methods so snippets
    // written for a browser or Node still run.
    let source = format!(
        "const console = {{ log: print, info: print, warn: print, error: print }};\n{code}"
    );
    let value = context
        .eval(Source::from_bytes(&source))
        .map_err(|err| format!("JavaScript error: {err}"))?;
    let rendered = value
        .to_string(&mut context)
        .map(|text| text.to_std_string_escaped())
        .unwrap_or_else(|_| value.display().to_string());
    let printed = JS_OUTPUT.with(|output| output.borrow().join("\n"));

    Ok(match (printed.is_empty(), rendered == "undefined") {
        (true, _) => rendered,
        (false, true) => printed,
        (false, false) => format!("{printed}\n{rendered}"),
    })
}

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "get_weather".into(),
            description: None,
            parameters: {
                let mut hashmap = HashMap::new();
                hashmap.insert(
                    "city".to_string(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "City".into(),
                    },
                );
                hashmap
            },
        },
        Tool {
            name: "read_file".into(),
            description: None,
            parameters: {
                let mut hashmap = HashMap::new();
                hashmap.insert(
                    "file_path".to_string(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Path to file on user's PC".into(),
                    },
                );
                hashmap
            },
        },
        Tool {
            name: "start_subagent".into(),
            description: Some(
                "Starts fresh-new agent that executes specific task, returns its last message."
                    .into(),
            ),
            parameters: {
                let mut hashmap = HashMap::new();
                hashmap.insert(
                    "prompt".to_string(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Blank-state prompt for agent".into(),
                    },
                );
                hashmap
            },
        },
        Tool {
            name: "eval_javascript".into(),
            description: Some(
                "Execute JavaScript code in an embedded Boa interpreter. Returns the printed \
                 output (print/console.log) followed by the value of the last expression."
                    .into(),
            ),
            parameters: {
                let mut hashmap = HashMap::new();
                hashmap.insert(
                    "code".to_string(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "JavaScript source code to execute".into(),
                    },
                );
                hashmap
            },
        },
    ]
}

fn request_body(
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

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let provider = Provider::from_env();
    println!("Provider: {provider:?}");

    let model = ModelConfig {
        kind: provider.kind("qwen3.6-35b-a3b"),
        tools: tools(),
        max_tokens: Some(50000),
    };
    let model = GlobalConfig {
        model_config: model,
        token: String::new(),
        endpoint: provider.endpoint().into(),
    };

    let user_prompt = std::env::var("PROMPT").unwrap_or_else(|_| {
        "Hi! List all available tools and try guess what they do.".to_string()
    });
    let mut history = vec![
        Entry::SystemPrompt {
            content: "You are helpful assistant. Avoid using markdown and LaTex notation.".into(),
        },
        Entry::UserMessage {
            content: MessageContent {
                text: Some(user_prompt),
                image: None,
            },
        },
    ];
    for i in &history {
        println!("{:?}", i)
    }

    loop {
        let body = request_body(provider, &history, &model.model_config)?;
        let url = model.endpoint.clone().parse::<hyper::Uri>()?;
        let host = url.host().expect("uri has no host");
        let port = url.port_u16().unwrap_or(80);
        let address = format!("{}:{}", host, port);

        // Open a TCP connection to the remote host.
        let stream = TcpStream::connect(address).await?;
        let io = TokioIo::new(stream);

        // Drive the HTTP state.
        let (mut sender, conn) = http1::handshake(io).await?;
        tokio::task::spawn(async move {
            if let Err(err) = conn.await {
                println!("Connection failed: {:?}", err);
            }
        });

        let authority = url.authority().unwrap().clone();
        let req = Request::builder()
            .uri(url.path())
            .header(hyper::header::HOST, authority.as_str())
            .header(hyper::header::CONTENT_TYPE, "application/json")
            .method("POST")
            .body(body)?;

        let mut res = sender.send_request(req).await?;
        println!("Response status: {}", res.status());
        if !res.status().is_success() {
            let mut body = Vec::new();
            while let Some(next) = res.frame().await {
                let frame = next?;
                if let Some(chunk) = frame.data_ref() {
                    body.extend_from_slice(chunk);
                }
            }
            anyhow::bail!("HTTP {}: {}", res.status(), String::from_utf8_lossy(&body));
        }

        // Live assistant turn. It stays in the history (as the last entry) while
        // the stream is running and is replaced by finalized entries afterwards.
        history.push(Entry::draft());
        let mut decoder = Decoder::new(provider);
        while let Some(next) = res.frame().await {
            let frame = next?;
            if let Some(chunk) = frame.data_ref() {
                let blocks = history
                    .last_mut()
                    .and_then(|entry| entry.as_draft_mut())
                    .expect("draft entry is last");
                for event in decoder.push_bytes(chunk, blocks)? {
                    match event {
                        StreamEvent::ThinkingDelta(text) | StreamEvent::TextDelta(text) => {
                            print!("{text}");
                            io::stdout().flush()?;
                        }
                        other => println!("\n{other:?}"),
                    }
                }
            }
        }

        {
            let blocks = history
                .last_mut()
                .and_then(|entry| entry.as_draft_mut())
                .expect("draft entry is last");
            for event in decoder.finish(blocks)? {
                println!("\n{event:?}");
            }
        }

        let draft = history.pop().expect("draft entry is last");
        let mut entries = finish_draft_for(draft, &model.model_config.kind)?;
        println!();
        for i in &entries {
            println!("{:?}", i)
        }
        if entries.is_empty() {
            println!("Stream produced no entries");
            return Ok(());
        }
        history.append(&mut entries);

        if let Entry::ResponseToolCall {
            thinking: _,
            content: _,
            tool_call,
        } = history.last().unwrap()
        {
            let mut tool_response = vec![];
            for i in tool_call {
                if i.tool_name == "get_weather" {
                    tool_response.push(ToolCallResult {
                        result: "30 degrees Celsius".into(),
                        is_error: false,
                        tool_use_id: i.tool_use_id.clone(),
                    });
                }
                if i.tool_name == "read_file" {
                    let v: Value = serde_json::from_str(&i.toolcall_params)?;
                    let path = v.get("file_path").unwrap().as_str().unwrap();
                    match fs::read_to_string(path) {
                        Ok(s) => tool_response.push(ToolCallResult {
                            result: s,
                            is_error: false,
                            tool_use_id: i.tool_use_id.clone(),
                        }),
                        Err(e) => tool_response.push(ToolCallResult {
                            result: e.to_string(),
                            is_error: true,
                            tool_use_id: i.tool_use_id.clone(),
                        }),
                    }
                }
                if i.tool_name == "start_subagent" {
                    let v: Value = serde_json::from_str(&i.toolcall_params)?;
                    let prompt = v.get("prompt").unwrap().as_str().unwrap();
                    tool_response.push(ToolCallResult {
                        result: format!("Not implemented yet. Incoming prompt was {}", prompt),
                        is_error: true,
                        tool_use_id: i.tool_use_id.clone(),
                    });
                }
                if i.tool_name == "eval_javascript" {
                    let result = match serde_json::from_str::<Value>(&i.toolcall_params) {
                        Ok(v) => match v.get("code").and_then(Value::as_str) {
                            Some(code) => eval_javascript(code),
                            None => Err("missing required string parameter `code`".into()),
                        },
                        Err(err) => Err(format!("invalid tool arguments: {err}")),
                    };
                    let (result, is_error) = match result {
                        Ok(output) => (output, false),
                        Err(err) => (err, true),
                    };
                    tool_response.push(ToolCallResult {
                        result,
                        is_error,
                        tool_use_id: i.tool_use_id.clone(),
                    });
                }
                if !matches!(
                    i.tool_name.as_str(),
                    "get_weather" | "read_file" | "start_subagent" | "eval_javascript"
                ) {
                    tool_response.push(ToolCallResult {
                        result: format!("Unknown tool: {}", i.tool_name),
                        is_error: true,
                        tool_use_id: i.tool_use_id.clone(),
                    });
                }
            }
            let entry = Entry::ToolCallResult {
                tool_call: tool_response,
            };
            println!("{:?}", entry);
            history.push(entry);
        } else {
            let mut buffer = String::new();
            if io::stdin().read_line(&mut buffer)? == 0 || buffer.trim().is_empty() {
                return Ok(());
            }
            history.push(Entry::UserMessage {
                content: MessageContent {
                    text: Some(buffer),
                    image: None,
                },
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::eval_javascript;

    #[test]
    fn returns_last_expression_value() {
        assert_eq!(eval_javascript("1 + 2 * 3").unwrap(), "7");
    }

    #[test]
    fn includes_printed_output_and_value() {
        assert_eq!(
            eval_javascript("console.log('hello'); 6 * 7").unwrap(),
            "hello\n42"
        );
    }

    #[test]
    fn reports_javascript_errors() {
        let error = eval_javascript("let = ;").unwrap_err();
        assert!(error.contains("JavaScript error"), "{error}");
    }
}
