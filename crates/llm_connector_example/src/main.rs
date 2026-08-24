use std::io;

use futures_util::StreamExt;
use llm_api::{
    model_config::{GlobalConfig, ModelConfig, ModelKind},
    openai,
};
use llm_history_model::history::{Entry, MessageContent};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = reqwest::Client::new();
    let model = ModelConfig {
        kind: ModelKind::OpenAICompatible {
            model_name: "qwen3.6-35b-a3b".into(),
            streaming: false,
        },
        tools: Default::default(),
    };
    let model = GlobalConfig {
        model_config: model,
        token: "".into(),
        endpoint: "http://127.0.0.1:8080/v1/responses".into(),
    };
    let mut history = vec![
        Entry::SystemPrompt {
            content: "You are useful assistant".into(),
        },
        Entry::UserMessage {
            content: MessageContent {
                text: Some("Hello!".into()),
                image: None,
            },
        },
    ];
    for i in &history {
        println!("{:?}", i)
    }
    loop {
        let validation = openai::to_responses_input(&history, &model.model_config);
        if let Err(errors) = validation {
            for e in errors {
                println!("{:?}", e)
            }
            return Ok(());
        }

        let response = validation.unwrap();
        let res = client
            .post(model.endpoint.clone())
            .json(&response)
            .send()
            .await?;
        // println!("{}", String::from_utf8_lossy(&res.bytes().await?.to_vec()));
        let res: openai::response::Response = res.json().await?;
        let mut entries = res.to_entries();
        for i in &entries {
            println!("{:?}", i)
        }
        history.append(&mut entries);
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
