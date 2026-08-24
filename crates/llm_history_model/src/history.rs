pub struct History {
    pub entries: Vec<Entry>
}

pub enum Entry {
    UserMessage{content: String},
    ResponseNormal{thinking: Option<String>, content: String},
    ResponseToolCall{thinking: Option<String>, tool_call: String, toolcall_params: String},
    ToolCallResult{result: String}
}