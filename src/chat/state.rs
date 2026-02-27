/// Chat overlay state: message list and visibility.

use bevy::prelude::*;

pub const MASCOT_WIDTH: u32 = 400;
pub const CHAT_PANEL_WIDTH: u32 = 350;
pub const WINDOW_WIDTH: u32 = MASCOT_WIDTH + CHAT_PANEL_WIDTH; // 750
pub const WINDOW_HEIGHT: u32 = 600;
// Raise chat panel from window bottom so its lower edge lines up with mascot feet.
pub const CHAT_PANEL_BOTTOM_INSET: f32 = 80.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub text: String,
    /// Whether this message is still being streamed.
    pub is_streaming: bool,
}

impl ChatMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self { role: MessageRole::User, text: text.into(), is_streaming: false }
    }

    pub fn assistant_streaming() -> Self {
        Self { role: MessageRole::Assistant, text: String::new(), is_streaming: true }
    }

    pub fn append(&mut self, token: &str) {
        self.text.push_str(token);
    }

    pub fn finish(&mut self) {
        self.is_streaming = false;
    }
}

#[derive(Resource)]
pub struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub visible: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self { messages: Vec::new(), visible: false }
    }
}

impl ChatState {
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    pub fn push_user(&mut self, text: impl Into<String>) {
        self.messages.push(ChatMessage::user(text));
    }

    /// Start a new streaming assistant message. Returns its index.
    pub fn start_assistant_message(&mut self) -> usize {
        self.messages.push(ChatMessage::assistant_streaming());
        self.messages.len() - 1
    }

    /// Append a token to the last assistant message.
    pub fn append_token(&mut self, token: &str) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == MessageRole::Assistant && msg.is_streaming {
                msg.append(token);
            }
        }
    }

    /// Finish the last streaming assistant message.
    pub fn finish_assistant_message(&mut self) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == MessageRole::Assistant {
                msg.finish();
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_hidden_and_empty() {
        let state = ChatState::default();
        assert!(!state.visible);
        assert!(state.messages.is_empty());
    }

    #[test]
    fn toggle_flips_visibility() {
        let mut state = ChatState::default();
        state.toggle();
        assert!(state.visible);
        state.toggle();
        assert!(!state.visible);
    }

    #[test]
    fn push_user_message() {
        let mut state = ChatState::default();
        state.push_user("hello");
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].role, MessageRole::User);
        assert_eq!(state.messages[0].text, "hello");
        assert!(!state.messages[0].is_streaming);
    }

    #[test]
    fn streaming_assistant_message_lifecycle() {
        let mut state = ChatState::default();
        let _idx = state.start_assistant_message();
        assert!(state.messages.last().unwrap().is_streaming);

        state.append_token("Hello");
        state.append_token(" world");
        assert_eq!(state.messages.last().unwrap().text, "Hello world");
        assert!(state.messages.last().unwrap().is_streaming);

        state.finish_assistant_message();
        assert!(!state.messages.last().unwrap().is_streaming);
    }

    #[test]
    fn append_token_only_affects_last_streaming_assistant() {
        let mut state = ChatState::default();
        state.push_user("hi");
        state.start_assistant_message();
        state.append_token("ok");
        // user message should be untouched
        assert_eq!(state.messages[0].text, "hi");
        assert_eq!(state.messages[1].text, "ok");
    }

    #[test]
    fn multiple_conversation_turns() {
        let mut state = ChatState::default();
        state.push_user("turn 1");
        state.start_assistant_message();
        state.append_token("response 1");
        state.finish_assistant_message();

        state.push_user("turn 2");
        state.start_assistant_message();
        state.append_token("response 2");
        state.finish_assistant_message();

        assert_eq!(state.messages.len(), 4);
    }

    #[test]
    fn chat_message_user_constructor() {
        let msg = ChatMessage::user("test");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.text, "test");
        assert!(!msg.is_streaming);
    }
}
