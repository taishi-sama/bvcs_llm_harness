use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
};

use http_body_util::BodyExt as _;
use hyper::{Request, client::conn::http1};
use hyper_util::rt::TokioIo;
use llm_api::{
    generic_tool::{ParameterDescription, ParameterType, Tool},
    model_config::{GlobalConfig, ModelConfig, ModelKind},
    openai_completions,
};
use llm_history_model::history::{Entry, MessageContent, ToolCallResult};
use serde_json::Value;
use tokio::net::TcpStream;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let _client = reqwest::Client::new();
    let model = ModelConfig {
        kind: ModelKind::OpenAICompatible {
            prompt_cache_key: "llm_connector_example".into(),
            model_name: "qwen3.6-35b-a3b".into(),
            streaming: false,
        },
        tools: vec![
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
        ],
    };
    let model = GlobalConfig {
        model_config: model,
        token: "".into(),
        endpoint: "http://127.0.0.1:8080/v1/chat/completions".into(),
    };
    let mut history = vec![
        Entry::SystemPrompt {
            content: "You are automatic diagnostics system running under generic Linux system platform.".into(),
        },
        Entry::UserMessage {
            content: MessageContent {
                text: Some("Determine in minimal amount of steps what are characteristics of current machine using only available tools.".into()),
                image: None,
            },
        },
    ];
    for i in &history {
        println!("{:?}", i)
    }
    loop {
        let validation = openai_completions::to_completions_input(&history, &model.model_config);
        if let Err(errors) = validation {
            for e in errors {
                println!("{:?}", e)
            }
            return Ok(());
        }

        let response = validation.unwrap();
        let url = model.endpoint.clone().parse::<hyper::Uri>()?;
        // let res = client
        //     .post(model.endpoint.clone())
        //     .json(&response)
        //     .send()
        //     .await?;
        let host = url.host().expect("uri has no host");
        let port = url.port_u16().unwrap_or(80);

        let address = format!("{}:{}", host, port);

        // Open a TCP connection to the remote host
        let stream = TcpStream::connect(address).await?;

        // Use an adapter to access something implementing `tokio::io` traits as if they implement
        // `hyper::rt` IO traits.
        let io = TokioIo::new(stream);

        // Create the Hyper client
        let (mut sender, conn) = http1::handshake(io).await?;
        // Spawn a task to poll the connection, driving the HTTP state
        tokio::task::spawn(async move {
            if let Err(err) = conn.await {
                println!("Connection failed: {:?}", err);
            }
        });
        // // println!("{}", String::from_utf8_lossy(&res.bytes().await?.to_vec()));
        // let res: openai::response::Response = res.json().await?;
        // The authority of our URL will be the hostname of the httpbin remote
        let authority = url.authority().unwrap().clone();

        // Create an HTTP request with an empty body and a HOST header
        let req = Request::builder()
            .uri(url.path())
            .header(hyper::header::HOST, authority.as_str())
            .method("POST")
            .body(serde_json::to_string(&response).unwrap())?;

        // Await the response...
        let mut res = sender.send_request(req).await?;
        let mut buffer: Vec<u8> = vec![];
        println!("Response status: {}", res.status());
        while let Some(next) = res.frame().await {
            let frame = next?;
            if let Some(chunk) = frame.data_ref() {
                buffer.write_all(chunk).unwrap();
            }
        }
        let response_parsed: openai_completions::response::CompletionResponse =
            serde_json::from_slice(&buffer).unwrap();
        let mut entries = response_parsed.to_entries();
        for i in &entries {
            println!("{:?}", i)
        }
        history.append(&mut entries);
        if let Entry::ResponseToolCall {
            thinking: _,
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
            }
            let entry = Entry::ToolCallResult {
                tool_call: tool_response,
            };
            println!("{:?}", entry);
            history.push(entry);
        } else {
            let mut buffer = String::new();
            io::stdin().read_line(&mut buffer)?;
            history.push(Entry::UserMessage {
                content: MessageContent {
                    text: Some(buffer),
                    image: None,
                },
            });
        }
    }
}
