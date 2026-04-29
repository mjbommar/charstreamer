use std::collections::HashSet;

use charstreamer_core::{
    CandidateSlice, FeatureAppender, FeatureBlock, FeatureError, FeatureMatrixViewMut,
    FeatureScratch, TextBytes,
};

#[derive(Clone, Debug, Default)]
pub struct LegacyFeatureTables {
    pub abbreviations: HashSet<String>,
    pub list_markers: Vec<String>,
    pub list_conjunctions: Vec<String>,
    pub list_intros: Vec<String>,
    pub terminal_sentence_chars: HashSet<char>,
    pub terminal_paragraph_chars: HashSet<char>,
    pub primary_terminators: HashSet<char>,
    pub secondary_terminators: HashSet<char>,
    pub opening_quotes: HashSet<char>,
    pub closing_quotes: HashSet<char>,
    pub punctuation_chars: HashSet<char>,
    pub whitespace_chars: HashSet<char>,
}

#[derive(Clone, Debug)]
struct LegacyCharText<'a> {
    text: &'a str,
    chars: Vec<char>,
    char_starts: Vec<usize>,
}

impl<'a> LegacyCharText<'a> {
    fn new(text: &'a str) -> Self {
        let mut chars = Vec::with_capacity(text.chars().count());
        let mut char_starts = Vec::with_capacity(text.chars().count() + 1);
        for (byte_index, ch) in text.char_indices() {
            char_starts.push(byte_index);
            chars.push(ch);
        }
        char_starts.push(text.len());
        Self {
            text,
            chars,
            char_starts,
        }
    }

    fn len(&self) -> usize {
        self.chars.len()
    }

    fn char_at(&self, index: usize) -> Option<char> {
        self.chars.get(index).copied()
    }

    fn slice(&self, start: usize, end: usize) -> &str {
        &self.text[self.char_starts[start]..self.char_starts[end]]
    }

    fn position_of_byte_offset(&self, byte_offset: usize) -> Option<usize> {
        self.char_starts[..self.char_starts.len().saturating_sub(1)]
            .binary_search(&byte_offset)
            .ok()
    }
}

/// Character-aware parity appender matching the original `charboundary` feature family:
/// encoded window plus 8 heuristic channels.
#[derive(Clone, Debug)]
pub struct CharBoundaryLegacyAppender {
    left_window: usize,
    right_window: usize,
    tables: LegacyFeatureTables,
}

impl CharBoundaryLegacyAppender {
    #[must_use]
    pub fn new(left_window: usize, right_window: usize, tables: LegacyFeatureTables) -> Self {
        Self {
            left_window,
            right_window,
            tables,
        }
    }

    fn feature_width(&self) -> usize {
        self.left_window + self.right_window + 1 + 8
    }

    fn encode_char(&self, ch: Option<char>) -> f32 {
        let Some(ch) = ch else {
            return -3.0;
        };
        if ch.is_alphabetic() {
            let lowered = ch.to_lowercase().next().unwrap_or(ch);
            return ((lowered as i64) - ('a' as i64) + 1) as f32;
        }
        if ch.is_ascii_digit() {
            return 0.0;
        }
        if self.tables.terminal_sentence_chars.contains(&ch) {
            return -1.0;
        }
        if self.tables.terminal_paragraph_chars.contains(&ch) {
            return -2.0;
        }
        if self.tables.whitespace_chars.contains(&ch) || ch.is_whitespace() {
            return -3.0;
        }
        if self.tables.punctuation_chars.contains(&ch) {
            return -4.0;
        }
        -5.0
    }

    fn is_in_abbreviation(&self, text: &LegacyCharText<'_>, position: usize) -> bool {
        if text.char_at(position) != Some('.') {
            return false;
        }

        let mut word_start = position;
        while word_start > 0 {
            let prev = text
                .char_at(word_start - 1)
                .expect("previous char must exist");
            if prev.is_alphanumeric() || prev == '.' {
                word_start -= 1;
            } else {
                break;
            }
        }

        self.tables
            .abbreviations
            .contains(text.slice(word_start, position + 1))
    }

    fn is_quote_balanced(&self, text: &LegacyCharText<'_>, position: usize) -> bool {
        let Some(ch) = text.char_at(position) else {
            return true;
        };
        if !self.tables.opening_quotes.contains(&ch) && !self.tables.closing_quotes.contains(&ch) {
            return true;
        }

        let prefix = text.slice(0, position + 1);
        let straight_double = prefix.matches('"').count();
        let curly_double_open = prefix.matches('\u{201c}').count();
        let curly_double_close = prefix.matches('\u{201d}').count();
        let straight_single = prefix.matches('\'').count();
        let curly_single_open = prefix.matches('\u{2018}').count();
        let curly_single_close = prefix.matches('\u{2019}').count();

        straight_double.is_multiple_of(2)
            && curly_double_open == curly_double_close
            && straight_single.is_multiple_of(2)
            && curly_single_open == curly_single_close
    }

    fn is_word_likely_complete(&self, text: &LegacyCharText<'_>, position: usize) -> bool {
        let Some(ch) = text.char_at(position) else {
            return true;
        };
        if !self.tables.closing_quotes.contains(&ch) {
            return true;
        }
        if position + 1 >= text.len() {
            return true;
        }

        let next = text.char_at(position + 1).expect("next char must exist");
        if next.is_whitespace()
            || self.tables.terminal_sentence_chars.contains(&next)
            || self.tables.punctuation_chars.contains(&next)
        {
            return true;
        }
        if next.is_lowercase() {
            return false;
        }
        true
    }

    fn is_in_list_item(
        &self,
        text: &LegacyCharText<'_>,
        position: usize,
        window_size: usize,
    ) -> bool {
        let Some(ch) = text.char_at(position) else {
            return false;
        };

        if !"()0123456789abcdefghijklmnopqrstuvwxyz.•·○●■□▪▫".contains(ch) {
            if position > 0 && position + 1 < text.len() {
                let prev = text
                    .char_at(position - 1)
                    .expect("previous char must exist");
                if ".,;:".contains(ch) || ".,;:".contains(prev) {
                    // fall through to broader context checks
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }

        for marker in &self.tables.list_markers {
            let marker_len = marker.chars().count();
            if position >= marker_len && text.slice(position - marker_len, position) == marker {
                return true;
            }
            if position + marker_len <= text.len()
                && text.slice(position, position + marker_len) == marker
            {
                return true;
            }
        }

        let prev = position
            .checked_sub(1)
            .and_then(|index| text.char_at(index))
            .unwrap_or('\0');
        let next = text.char_at(position + 1).unwrap_or('\0');
        if ".,;:()[]".contains(ch) || ".,;:()".contains(prev) || ".,;:()[]".contains(next) {
            let start = position.saturating_sub(window_size);
            let context_before = text.slice(start, position);
            let has_colon_before = {
                let min_intro_len = 5;
                if context_before.chars().count() >= min_intro_len {
                    let tail_start = context_before
                        .char_indices()
                        .nth(context_before.chars().count() - min_intro_len)
                        .map(|(offset, _)| offset)
                        .unwrap_or(0);
                    context_before[tail_start..].contains(':')
                } else {
                    false
                }
            };

            if has_colon_before
                && self
                    .tables
                    .list_intros
                    .iter()
                    .any(|intro| context_before.contains(intro))
            {
                return true;
            }

            if ch == ';' || ch == ',' || prev == ';' || prev == ',' {
                let end = (position + window_size).min(text.len());
                let context_after = text.slice(position, end);
                for conj in &self.tables.list_conjunctions {
                    if let Some(conj_pos) = context_after.find(conj)
                        && conj_pos > 0
                        && conj_pos < 5
                    {
                        let before = context_after[..conj_pos].chars().last().unwrap_or('\0');
                        if before == ';' || before == ',' {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    fn is_semicolon_in_list(
        &self,
        text: &LegacyCharText<'_>,
        position: usize,
        window_size: usize,
    ) -> bool {
        if text.char_at(position) != Some(';') {
            return false;
        }

        let start = position.saturating_sub(window_size);
        let end = (position + window_size).min(text.len());
        let context = text.slice(start, end);
        if context.matches(';').count() >= 2 {
            return true;
        }
        if self
            .tables
            .list_markers
            .iter()
            .any(|marker| context.contains(marker))
        {
            return true;
        }
        if self
            .tables
            .list_intros
            .iter()
            .any(|intro| text.slice(start, position).contains(intro))
        {
            return true;
        }
        self.tables
            .list_conjunctions
            .iter()
            .any(|conj| text.slice(position, end).contains(conj))
    }

    fn is_near_colon(
        &self,
        text: &LegacyCharText<'_>,
        position: usize,
        window_size: usize,
    ) -> bool {
        let start = position.saturating_sub(window_size);
        text.slice(start, position).contains(':')
    }
}

impl FeatureAppender<f32> for CharBoundaryLegacyAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("charboundary_legacy", self.feature_width())
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.feature_width() {
            return Err(FeatureError::new(
                "charboundary legacy appender got a mismatched destination view",
            ));
        }
        let Some(text) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "charboundary legacy appender requires valid UTF-8 text",
            ));
        };

        let view = LegacyCharText::new(text);
        let window_width = self.left_window + self.right_window + 1;

        for (row_index, position) in positions.data.iter().enumerate() {
            let char_index = view
                .position_of_byte_offset(position.as_usize())
                .ok_or_else(|| {
                    FeatureError::new("candidate position is not a UTF-8 scalar start")
                })?;
            let row = out.row_mut(row_index);

            for (column, relative) in
                (-(self.left_window as isize)..=(self.right_window as isize)).enumerate()
            {
                let target = char_index as isize + relative;
                row[column] = if target < 0 {
                    -3.0
                } else {
                    self.encode_char(view.char_at(target as usize))
                };
            }

            row[window_width] = if self.is_in_abbreviation(&view, char_index) {
                1.0
            } else {
                0.0
            };

            let current = view.char_at(char_index).unwrap_or('\0');
            row[window_width + 1] = if self.tables.primary_terminators.contains(&current) {
                1.0
            } else if self.tables.secondary_terminators.contains(&current) {
                -1.0
            } else {
                0.0
            };
            row[window_width + 2] = if self.is_quote_balanced(&view, char_index) {
                1.0
            } else {
                0.0
            };
            row[window_width + 3] = if self.is_word_likely_complete(&view, char_index) {
                1.0
            } else {
                0.0
            };
            row[window_width + 4] = if view.char_at(char_index + 1).is_some_and(char::is_lowercase)
            {
                1.0
            } else {
                0.0
            };
            row[window_width + 5] = if self.is_in_list_item(&view, char_index, 20) {
                1.0
            } else {
                0.0
            };
            row[window_width + 6] = if self.is_semicolon_in_list(&view, char_index, 50) {
                1.0
            } else {
                0.0
            };
            row[window_width + 7] = if self.is_near_colon(&view, char_index, 10) {
                1.0
            } else {
                0.0
            };
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use charstreamer_core::{
        CandidateBuffer, CandidateScanner, FeatureAppender, ScanRange, TextBytes,
    };

    use crate::{ByteSet256, ByteSetScanner, CharBoundaryLegacyAppender, LegacyFeatureTables};

    #[test]
    fn legacy_appender_uses_expected_width() {
        let tables = LegacyFeatureTables {
            abbreviations: HashSet::from(["Mr.".to_string()]),
            list_markers: vec!["(1)".to_string()],
            list_conjunctions: vec![" and ".to_string()],
            list_intros: vec!["following:".to_string()],
            terminal_sentence_chars: HashSet::from(['.', '!', '?', ';', '"', '\'', ':']),
            terminal_paragraph_chars: HashSet::from(['\n', '\r']),
            primary_terminators: HashSet::from(['.', '!', '?']),
            secondary_terminators: HashSet::from(['"', '\'', ';', ':']),
            opening_quotes: HashSet::from(['"', '\'']),
            closing_quotes: HashSet::from(['"', '\'']),
            punctuation_chars: HashSet::from(['.', ',', ';', ':', '"', '\'']),
            whitespace_chars: HashSet::from([' ', '\n', '\r', '\t']),
        };
        let appender = CharBoundaryLegacyAppender::new(5, 3, tables);
        let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b"."));
        let text = TextBytes::from_utf8("Mr. Smith.");
        let mut candidates = CandidateBuffer::new();
        scanner.scan_into(text, ScanRange::full(text), &mut candidates);
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(candidates.len(), appender.block().width);
        appender
            .append_into(
                text,
                candidates.as_slice(),
                matrix.as_view_mut(),
                &mut charstreamer_core::FeatureScratch::default(),
            )
            .expect("legacy appender should succeed");

        assert_eq!(appender.block().width, 17);
        assert_eq!(matrix.cols, 17);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.data[9], 1.0);
    }
}
