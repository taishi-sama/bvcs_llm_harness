use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct History {
    pub entries: Vec<Entry>,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ToolCall {
    pub tool_name: String,
    pub toolcall_params: String,
    pub tool_use_id: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCallResult {
    pub result: String,
    pub is_error: bool,
    pub tool_use_id: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub enum Thinking {
    Normal { thinking: String },
    Secret { thinking: String, signature: String },
}
#[derive(Serialize, Deserialize, Debug)]
pub enum ImageMIME {
    PNG,
    JPEG,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Image {
    pub mime: ImageMIME,
    pub base64: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct MessageContent {
    pub text: Option<String>,
    pub image: Option<Image>, //Support passing as Base64-encoded image for now
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Entry {
    /// Should be only one in beginning of history
    SystemPrompt {
        content: String,
    },
    UserMessage {
        content: MessageContent,
    },
    ResponseNormal {
        thinking: Option<Thinking>,
        content: String,
    },
    ResponseToolCall {
        thinking: Option<Thinking>,
        tool_call: Vec<ToolCall>,
    },
    ToolCallResult {
        tool_call: Vec<ToolCallResult>,
    },
}
