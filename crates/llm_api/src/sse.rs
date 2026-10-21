//! Incremental Server-Sent Events (SSE) framing.
//!
//! The decoder is transport-agnostic: feed it raw HTTP body chunks and it yields
//! complete SSE events. It handles the framing details providers rely on:
//!
//! - `event:` and `data:` fields (multi-line `data` is joined with `\n`)
//! - CRLF and LF line endings
//! - comment lines (`:` heartbeats) and unknown fields (`id`, `retry`)
//! - events split across arbitrary chunk boundaries, including mid-UTF-8
//!
//! Provider-specific payload parsing lives in each provider's `stream` module.

/// One dispatched SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Value of the `event:` field, if the provider sent one.
    pub event: Option<String>,
    /// Joined `data:` payload. Never empty for a dispatched event.
    pub data: String,
}

/// Incremental SSE frame decoder.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
    event: Option<String>,
    data: String,
    has_data: bool,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends raw bytes and returns every event completed by them.
    ///
    /// Incomplete lines stay buffered. UTF-8 is decoded only once a full line is
    /// available, so multi-byte characters may be split arbitrarily across
    /// chunks.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(bytes);

        let mut out = Vec::new();
        let mut consumed = 0usize;
        while let Some(pos) = self.buf[consumed..].iter().position(|b| *b == b'\n') {
            let end = consumed + pos;
            let line = self.buf[consumed..end].to_vec();
            consumed = end + 1;
            let line = line.strip_suffix(b"\r").unwrap_or(&line);
            self.feed_line(line, &mut out);
        }
        if consumed > 0 {
            self.buf.drain(..consumed);
        }
        out
    }

    /// Flushes any trailing line and pending event at end of stream.
    ///
    /// Tolerates bodies that end without a final blank line.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let line = std::mem::take(&mut self.buf);
            let line = line.strip_suffix(b"\r").unwrap_or(&line).to_vec();
            self.feed_line(&line, &mut out);
        }
        self.dispatch(&mut out);
        out
    }

    fn feed_line(&mut self, line: &[u8], out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        // Comment / keep-alive.
        if line[0] == b':' {
            return;
        }

        let text = String::from_utf8_lossy(line);
        let (field, value) = match text.find(':') {
            Some(i) => {
                let value = &text[i + 1..];
                (&text[..i], value.strip_prefix(' ').unwrap_or(value))
            }
            None => (text.as_ref(), ""),
        };

        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            // `id`, `retry` and unknown fields are irrelevant here.
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        if self.has_data || self.event.is_some() {
            out.push(SseEvent {
                event: self.event.take(),
                data: std::mem::take(&mut self.data),
            });
            self.has_data = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(chunks: &[&str]) -> Vec<SseEvent> {
        let mut decoder = SseDecoder::new();
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(decoder.push(chunk.as_bytes()));
        }
        out.extend(decoder.finish());
        out
    }

    #[test]
    fn parses_event_and_data() {
        let events = collect(&["event: message_start\ndata: {\"type\":\"message_start\"}\n\n"]);
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("message_start".into()),
                data: "{\"type\":\"message_start\"}".into(),
            }]
        );
    }

    #[test]
    fn joins_multi_line_data_and_ignores_comments() {
        let events = collect(&[": ping\n\ndata: first\ndata: second\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "first\nsecond");
        assert_eq!(events[0].event, None);
    }

    #[test]
    fn handles_crlf_and_trailing_event_without_blank_line() {
        let events = collect(&["event: x\r\ndata: y\r\n\r\ndata: tail"]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "y");
        assert_eq!(events[1].data, "tail");
    }

    #[test]
    fn split_across_every_byte() {
        let raw = "event: delta\ndata: {\"text\":\"héllo\"}\n\n";
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        for b in raw.as_bytes() {
            events.extend(decoder.push(std::slice::from_ref(b)));
        }
        events.extend(decoder.finish());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"text\":\"héllo\"}");
    }

    #[test]
    fn ignores_id_and_retry() {
        let events = collect(&["id: 7\nretry: 100\ndata: payload\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "payload");
    }
}
