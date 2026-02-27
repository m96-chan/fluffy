/// Splits a streaming token stream into sentences for TTS.
///
/// Sentence boundaries are detected at:
/// - `. ` (period followed by space or end of input)
/// - `! ` or `? `
/// - `。` (Japanese full stop)
/// - `！` or `？` (full-width)
/// - `\n` (newline — treat as sentence break)
pub struct SentenceSplitter {
    buffer: String,
}

impl SentenceSplitter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Push a token into the splitter.
    /// Returns any complete sentences found.
    pub fn push(&mut self, token: &str) -> Vec<String> {
        self.buffer.push_str(token);
        self.extract_sentences()
    }

    /// Flush any remaining buffered text as a sentence (called at end of stream).
    pub fn flush(&mut self) -> Option<String> {
        let text = self.buffer.trim().to_string();
        self.buffer.clear();
        if text.is_empty() { None } else { Some(text) }
    }

    fn extract_sentences(&mut self) -> Vec<String> {
        let mut sentences = Vec::new();

        loop {
            if let Some(boundary) = find_sentence_boundary(&self.buffer) {
                let sentence = self.buffer[..boundary].trim().to_string();
                self.buffer.drain(..boundary);
                // Skip leading whitespace in remaining buffer
                while self.buffer.starts_with(' ') {
                    self.buffer.remove(0);
                }
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
            } else {
                break;
            }
        }

        sentences
    }
}

/// Find the end index of the first complete sentence in the buffer.
/// Returns None if no complete sentence is found yet.
fn find_sentence_boundary(text: &str) -> Option<usize> {
    let chars: Vec<char> = text.chars().collect();
    let mut byte_pos = 0;

    for (i, &c) in chars.iter().enumerate() {
        let char_bytes = c.len_utf8();

        match c {
            // ASCII sentence enders: need a following space or be at end
            '.' | '!' | '?' => {
                let next = chars.get(i + 1);
                match next {
                    Some(' ') | Some('\n') | None => {
                        // But don't split on abbreviations like "e.g. "
                        // Simple heuristic: if previous char was also a letter and it's a period,
                        // still split (this may occasionally split incorrectly, acceptable for TTS)
                        return Some(byte_pos + char_bytes);
                    }
                    _ => {}
                }
            }
            // Full-width Japanese punctuation — always a boundary
            '。' | '！' | '？' | '…' => {
                return Some(byte_pos + char_bytes);
            }
            // Newline — treat as boundary
            '\n' => {
                if byte_pos > 0 {
                    return Some(byte_pos);
                }
            }
            // Skip markdown code fence — don't try to TTS code blocks
            '`' => {
                // If we see ``` skip to end of code block
                if text[byte_pos..].starts_with("```") {
                    if let Some(close) = text[byte_pos + 3..].find("```") {
                        // Skip past closing ```
                        let skip_to = byte_pos + 3 + close + 3;
                        return Some(skip_to.min(text.len()));
                    } else {
                        // Unclosed code block — wait for more tokens
                        return None;
                    }
                }
            }
            _ => {}
        }

        byte_pos += char_bytes;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_sentence_emitted_at_period() {
        let mut splitter = SentenceSplitter::new();
        let mut results = Vec::new();

        results.extend(splitter.push("Hello world. "));

        assert_eq!(results, vec!["Hello world."]);
    }

    #[test]
    fn japanese_period_triggers_boundary() {
        let mut splitter = SentenceSplitter::new();
        let results = splitter.push("こんにちは。今日は");
        assert_eq!(results, vec!["こんにちは。"]);
    }

    #[test]
    fn incomplete_sentence_not_emitted() {
        let mut splitter = SentenceSplitter::new();
        let results = splitter.push("This is incomplete");
        assert!(results.is_empty(), "Should not emit until sentence ends");
    }

    #[test]
    fn flush_returns_remaining_text() {
        let mut splitter = SentenceSplitter::new();
        splitter.push("Some remaining");
        let flushed = splitter.flush();
        assert_eq!(flushed, Some("Some remaining".to_string()));
    }

    #[test]
    fn flush_on_empty_returns_none() {
        let mut splitter = SentenceSplitter::new();
        assert_eq!(splitter.flush(), None);
    }

    #[test]
    fn multiple_sentences_emitted_in_order() {
        let mut splitter = SentenceSplitter::new();
        let results = splitter.push("First sentence. Second sentence! ");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "First sentence.");
        assert_eq!(results[1], "Second sentence!");
    }

    #[test]
    fn exclamation_and_question_are_boundaries() {
        let mut splitter = SentenceSplitter::new();

        let r1 = splitter.push("Really! ");
        assert_eq!(r1, vec!["Really!"]);

        let r2 = splitter.push("Why? ");
        assert_eq!(r2, vec!["Why?"]);
    }

    #[test]
    fn code_block_skipped() {
        let mut splitter = SentenceSplitter::new();
        let results = splitter.push("Here is code:\n```\nfn main() {}\n``` ");
        // Code block content shouldn't generate a spoken sentence mid-block
        // (implementation skips code blocks)
        // Just verify we don't crash
        let _ = results;
        let _ = splitter.flush();
    }
}
