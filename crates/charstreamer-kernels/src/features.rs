use charstreamer_core::{
    ByteWindowSpec, CandidateSlice, FeatureAppender, FeatureBlock, FeatureError, FeatureKernel,
    FeatureMatrixViewMut, FeatureSchema, FeatureScratch, TextBytes,
};
use icu::properties::{GeneralCategory, maps};
use serde::{Deserialize, Serialize};

use crate::AsciiClassTable;

fn bool_to_f32(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn is_sentence_terminal_char(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '…')
}

fn is_closing_quote_or_bracket_char(ch: char) -> bool {
    matches!(ch, '"' | '\'' | ')' | ']' | '}' | '>' | '”' | '’' | '»')
}

fn floor_char_boundary(text: &str, position: usize) -> usize {
    let mut position = position.min(text.len());
    while position > 0 && !text.is_char_boundary(position) {
        position -= 1;
    }
    position
}

fn ceil_char_boundary(text: &str, position: usize) -> usize {
    let mut position = position.min(text.len());
    while position < text.len() && !text.is_char_boundary(position) {
        position += 1;
    }
    position
}

fn line_start_before_or_at(text: &str, position: usize) -> usize {
    let position = floor_char_boundary(text, position);
    text[..position]
        .rfind(['\n', '\r'])
        .map_or(0, |index| index + 1)
}

fn line_end_after_or_at(text: &str, position: usize) -> usize {
    let position = floor_char_boundary(text, position);
    text[position..]
        .find(['\n', '\r'])
        .map_or(text.len(), |relative| position + relative)
}

fn current_line(text: &str, position: usize) -> &str {
    let start = line_start_before_or_at(text, position);
    let end = line_end_after_or_at(text, position);
    &text[start..end]
}

fn next_nonspace_char(text: &str, offset: usize) -> Option<(usize, char)> {
    let offset = ceil_char_boundary(text, offset);
    text.get(offset..)?
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(relative, ch)| (offset + relative, ch))
}

fn previous_nonspace_char(text: &str, offset: usize) -> Option<(usize, char)> {
    let offset = floor_char_boundary(text, offset);
    text.get(..offset)?
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_whitespace())
}

fn char_at_or_after(text: &str, offset: usize) -> Option<(usize, char)> {
    let offset = ceil_char_boundary(text, offset);
    text.get(offset..)?
        .char_indices()
        .next()
        .map(|(relative, ch)| (offset + relative, ch))
}

fn after_candidate_token(text: &str, position: usize) -> usize {
    let Some((_, center)) = char_at_or_after(text, position) else {
        return position.min(text.len());
    };
    let mut cursor = position + center.len_utf8();
    while let Some((_, ch)) = char_at_or_after(text, cursor) {
        if !is_closing_quote_or_bracket_char(ch) {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn whitespace_run_newline_count(text: &str, offset: usize) -> usize {
    let offset = ceil_char_boundary(text, offset);
    text.get(offset..)
        .unwrap_or_default()
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .filter(|&ch| matches!(ch, '\n' | '\r'))
        .count()
}

fn preceding_whitespace_run_newline_count(text: &str, offset: usize) -> usize {
    let offset = floor_char_boundary(text, offset);
    text.get(..offset)
        .unwrap_or_default()
        .chars()
        .rev()
        .take_while(|ch| ch.is_whitespace())
        .filter(|&ch| matches!(ch, '\n' | '\r'))
        .count()
}

fn starts_heading_like(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with('#')
        || trimmed
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
            && trimmed.chars().take(48).any(|ch| ch == ':')
}

fn starts_list_like(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with(['-', '*', '•']) {
        return true;
    }

    let mut chars = trimmed.chars().peekable();
    let mut digits = 0_usize;
    while chars.peek().is_some_and(|ch| ch.is_ascii_digit()) {
        digits += 1;
        chars.next();
    }
    digits > 0 && matches!(chars.next(), Some('.' | ')'))
}

fn has_metadata_colon(line: &str) -> bool {
    let trimmed = line.trim_start();
    let Some(colon) = trimmed.find(':') else {
        return false;
    };
    colon <= 40
        && trimmed[..colon].chars().any(|ch| ch.is_ascii_alphabetic())
        && !trimmed[..colon].contains('.')
}

fn next_line_after(text: &str, offset: usize) -> Option<&str> {
    let offset = floor_char_boundary(text, offset);
    let search = text.get(offset..)?;
    let newline = search.find(['\n', '\r'])?;
    let line_start = offset + newline + 1;
    Some(current_line(text, line_start))
}

fn normalized_byte(text: TextBytes<'_>, index: isize) -> f32 {
    f32::from(text.padded_byte(index)) / 255.0
}

fn encoded_boundary_byte(byte: u8) -> f32 {
    match byte {
        0 => -3.0,
        b if b.is_ascii_alphabetic() => f32::from(b.to_ascii_lowercase() - b'a' + 1),
        b if b.is_ascii_digit() => 0.0,
        b'.' | b'!' | b'?' | b';' | b':' | b'"' | b'\'' => -1.0,
        b'\n' | b'\r' => -2.0,
        b if b.is_ascii_whitespace() => -3.0,
        b if b.is_ascii_punctuation() => -4.0,
        _ => -5.0,
    }
}

/// Reusable byte classes for configuration-driven count features.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ByteClass {
    AsciiUpper,
    AsciiLower,
    AsciiAlpha,
    AsciiDigit,
    AsciiAlnum,
    AsciiWhitespace,
    AsciiPunctuation,
    LineBreak,
    OpenBracket,
    CloseBracket,
}

impl ByteClass {
    fn matches(self, byte: u8, classes: &AsciiClassTable) -> bool {
        match self {
            Self::AsciiUpper => classes.is_upper(byte),
            Self::AsciiLower => classes.is_lower(byte),
            Self::AsciiAlpha => classes.is_alpha(byte),
            Self::AsciiDigit => classes.is_digit(byte),
            Self::AsciiAlnum => classes.is_alnum(byte),
            Self::AsciiWhitespace => classes.is_space(byte),
            Self::AsciiPunctuation => classes.is_punct(byte),
            Self::LineBreak => matches!(byte, b'\n' | b'\r'),
            Self::OpenBracket => matches!(byte, b'(' | b'[' | b'{'),
            Self::CloseBracket => matches!(byte, b')' | b']' | b'}'),
        }
    }
}

/// Full Unicode General Category values, aligned with `alea-preprocess`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum UnicodeCategory {
    Ll,
    Lu,
    Lt,
    Lm,
    Lo,
    Mn,
    Mc,
    Me,
    Nd,
    Nl,
    No,
    Pc,
    Pd,
    Ps,
    Pe,
    Pi,
    Pf,
    Po,
    Sm,
    Sc,
    Sk,
    So,
    Zs,
    Zl,
    Zp,
    Cc,
    Cf,
    Cs,
    Co,
    Cn,
}

impl UnicodeCategory {
    fn of(ch: char) -> Self {
        match maps::general_category().get(ch) {
            GeneralCategory::LowercaseLetter => Self::Ll,
            GeneralCategory::UppercaseLetter => Self::Lu,
            GeneralCategory::TitlecaseLetter => Self::Lt,
            GeneralCategory::ModifierLetter => Self::Lm,
            GeneralCategory::OtherLetter => Self::Lo,
            GeneralCategory::NonspacingMark => Self::Mn,
            GeneralCategory::SpacingMark => Self::Mc,
            GeneralCategory::EnclosingMark => Self::Me,
            GeneralCategory::DecimalNumber => Self::Nd,
            GeneralCategory::LetterNumber => Self::Nl,
            GeneralCategory::OtherNumber => Self::No,
            GeneralCategory::ConnectorPunctuation => Self::Pc,
            GeneralCategory::DashPunctuation => Self::Pd,
            GeneralCategory::OpenPunctuation => Self::Ps,
            GeneralCategory::ClosePunctuation => Self::Pe,
            GeneralCategory::InitialPunctuation => Self::Pi,
            GeneralCategory::FinalPunctuation => Self::Pf,
            GeneralCategory::OtherPunctuation => Self::Po,
            GeneralCategory::MathSymbol => Self::Sm,
            GeneralCategory::CurrencySymbol => Self::Sc,
            GeneralCategory::ModifierSymbol => Self::Sk,
            GeneralCategory::OtherSymbol => Self::So,
            GeneralCategory::SpaceSeparator => Self::Zs,
            GeneralCategory::LineSeparator => Self::Zl,
            GeneralCategory::ParagraphSeparator => Self::Zp,
            GeneralCategory::Control => Self::Cc,
            GeneralCategory::Format => Self::Cf,
            GeneralCategory::Surrogate => Self::Cs,
            GeneralCategory::PrivateUse => Self::Co,
            GeneralCategory::Unassigned => Self::Cn,
        }
    }

    fn group(self) -> UnicodeCategoryGroup {
        match self {
            Self::Ll | Self::Lu | Self::Lt | Self::Lm | Self::Lo => UnicodeCategoryGroup::L,
            Self::Mn | Self::Mc | Self::Me => UnicodeCategoryGroup::M,
            Self::Nd | Self::Nl | Self::No => UnicodeCategoryGroup::N,
            Self::Pc | Self::Pd | Self::Ps | Self::Pe | Self::Pi | Self::Pf | Self::Po => {
                UnicodeCategoryGroup::P
            }
            Self::Sm | Self::Sc | Self::Sk | Self::So => UnicodeCategoryGroup::S,
            Self::Zs | Self::Zl | Self::Zp => UnicodeCategoryGroup::Z,
            Self::Cc | Self::Cf | Self::Cs | Self::Co | Self::Cn => UnicodeCategoryGroup::C,
        }
    }
}

/// Coarse Unicode category groups, aligned with `alea-preprocess`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum UnicodeCategoryGroup {
    L,
    M,
    N,
    P,
    S,
    Z,
    C,
}

fn decode_unicode_categories(
    text: &str,
) -> (
    Vec<charstreamer_core::BytePos>,
    Vec<UnicodeCategory>,
    Vec<UnicodeCategoryGroup>,
) {
    let mut starts = Vec::new();
    let mut categories = Vec::new();
    let mut groups = Vec::new();
    for (offset, ch) in text.char_indices() {
        let category = UnicodeCategory::of(ch);
        starts.push(charstreamer_core::BytePos::from_usize(offset));
        categories.push(category);
        groups.push(category.group());
    }
    (starts, categories, groups)
}

fn scalar_index_at_or_before(
    scalar_starts: &[charstreamer_core::BytePos],
    position: usize,
) -> Option<usize> {
    if scalar_starts.is_empty() {
        return None;
    }

    match scalar_starts.binary_search(&charstreamer_core::BytePos::from_usize(position)) {
        Ok(index) => Some(index),
        Err(0) => Some(0),
        Err(index) => Some(index - 1),
    }
}

/// Rolling byte window appender.
#[derive(Clone, Copy, Debug)]
pub struct ByteWindowAppender {
    spec: ByteWindowSpec,
}

impl ByteWindowAppender {
    #[must_use]
    pub fn new(spec: ByteWindowSpec) -> Self {
        Self { spec }
    }
}

impl FeatureAppender<f32> for ByteWindowAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("byte_window", self.spec.width())
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.spec.width() {
            return Err(FeatureError::new(
                "byte-window appender got a mismatched destination view",
            ));
        }

        let left = self.spec.left as isize;
        let right = self.spec.right as isize;
        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize() as isize;
            let row = out.row_mut(row_index);
            for (column, rel) in (-left..=right).enumerate() {
                row[column] = normalized_byte(text, center + rel);
            }
        }
        Ok(())
    }
}

/// Character-encoder-like byte window that more closely matches the original Python feature family.
#[derive(Clone, Copy, Debug)]
pub struct EncodedByteWindowAppender {
    spec: ByteWindowSpec,
}

impl EncodedByteWindowAppender {
    #[must_use]
    pub fn new(spec: ByteWindowSpec) -> Self {
        Self { spec }
    }
}

impl FeatureAppender<f32> for EncodedByteWindowAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("encoded_byte_window", self.spec.width())
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.spec.width() {
            return Err(FeatureError::new(
                "encoded-byte-window appender got a mismatched destination view",
            ));
        }

        let left = self.spec.left as isize;
        let right = self.spec.right as isize;
        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize() as isize;
            let row = out.row_mut(row_index);
            for (column, rel) in (-left..=right).enumerate() {
                row[column] = encoded_boundary_byte(text.padded_byte(center + rel));
            }
        }
        Ok(())
    }
}

/// ASCII neighbor classification appender.
#[derive(Clone, Debug, Default)]
pub struct AsciiClassAppender {
    classes: AsciiClassTable,
}

impl AsciiClassAppender {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl FeatureAppender<f32> for AsciiClassAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("ascii_classes", 6)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != 6 {
            return Err(FeatureError::new(
                "ascii-class appender got a mismatched destination view",
            ));
        }

        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize() as isize;
            let prev = text.padded_byte(center - 1);
            let next = text.padded_byte(center + 1);
            let row = out.row_mut(row_index);
            row[0] = bool_to_f32(self.classes.is_space(prev));
            row[1] = bool_to_f32(self.classes.is_upper(prev));
            row[2] = bool_to_f32(self.classes.is_lower(prev));
            row[3] = bool_to_f32(self.classes.is_space(next));
            row[4] = bool_to_f32(self.classes.is_upper(next));
            row[5] = bool_to_f32(self.classes.is_lower(next));
        }
        Ok(())
    }
}

/// UTF-8 structural features for candidate boundary positions.
///
/// This block is intentionally model-agnostic: it does not know the task label
/// or candidate source, only the text shape around the supplied byte offset.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryShapeAppender;

impl BoundaryShapeAppender {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl FeatureAppender<f32> for BoundaryShapeAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("boundary_shape", 22)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != 22 {
            return Err(FeatureError::new(
                "boundary-shape appender got a mismatched destination view",
            ));
        }
        let Some(text) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "boundary-shape features require valid UTF-8 text",
            ));
        };

        for (row_index, position) in positions.data.iter().enumerate() {
            let offset = position.as_usize().min(text.len());
            let center = char_at_or_after(text, offset).map(|(_, ch)| ch);
            let after_token = after_candidate_token(text, offset);
            let next = next_nonspace_char(text, after_token);
            let previous = previous_nonspace_char(text, offset);
            let line = current_line(text, offset);
            let next_line = next_line_after(text, after_token).unwrap_or_default();
            let immediate_next =
                char_at_or_after(text, offset + center.map_or(0, char::len_utf8)).map(|(_, ch)| ch);
            let newline_count_after = whitespace_run_newline_count(text, after_token);
            let newline_count_before =
                preceding_whitespace_run_newline_count(text, line_start_before_or_at(text, offset));
            let row = out.row_mut(row_index);
            row[0] = bool_to_f32(center.is_some_and(is_sentence_terminal_char));
            row[1] = bool_to_f32(center == Some('.'));
            row[2] = bool_to_f32(center == Some('!'));
            row[3] = bool_to_f32(center == Some('?'));
            row[4] = bool_to_f32(matches!(center, Some(':' | ';')));
            row[5] = bool_to_f32(center.is_some_and(is_closing_quote_or_bracket_char));
            row[6] = bool_to_f32(previous.is_some_and(|(_, ch)| is_sentence_terminal_char(ch)));
            row[7] = bool_to_f32(next.is_some_and(|(_, ch)| ch.is_uppercase()));
            row[8] = bool_to_f32(next.is_some_and(|(_, ch)| ch.is_lowercase()));
            row[9] = bool_to_f32(next.is_some_and(|(_, ch)| ch.is_ascii_digit()));
            row[10] = bool_to_f32(next.is_some_and(|(_, ch)| matches!(ch, '"' | '\'' | '“' | '‘')));
            row[11] = bool_to_f32(next.is_some_and(|(_, ch)| ch == '#'));
            row[12] = bool_to_f32(next.is_some_and(|(_, ch)| matches!(ch, '-' | '*' | '•')));
            row[13] = bool_to_f32(next.is_none());
            row[14] = bool_to_f32(immediate_next.is_some_and(is_closing_quote_or_bracket_char));
            row[15] = bool_to_f32(newline_count_after >= 2);
            row[16] = bool_to_f32(newline_count_before >= 2);
            row[17] = bool_to_f32(starts_heading_like(line));
            row[18] = bool_to_f32(starts_list_like(line));
            row[19] = bool_to_f32(has_metadata_colon(line));
            row[20] = bool_to_f32(starts_heading_like(next_line));
            row[21] = bool_to_f32(starts_list_like(next_line));
        }
        Ok(())
    }
}

/// Counts selected bytes over a centered byte window.
#[derive(Clone, Debug)]
pub struct SelectedByteCountAppender {
    name: &'static str,
    spec: ByteWindowSpec,
    bytes: Vec<u8>,
}

impl SelectedByteCountAppender {
    #[must_use]
    pub fn new(name: &'static str, spec: ByteWindowSpec, bytes: Vec<u8>) -> Self {
        Self { name, spec, bytes }
    }
}

impl FeatureAppender<f32> for SelectedByteCountAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new(self.name, self.bytes.len())
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.bytes.len() {
            return Err(FeatureError::new(
                "selected-byte-count appender got a mismatched destination view",
            ));
        }

        let left = self.spec.left as isize;
        let right = self.spec.right as isize;
        let bytes = text.bytes();
        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize() as isize;
            let window_start = (center - left).max(0) as usize;
            let window_end = (center + right + 1).max(0) as usize;
            let clamped_end = window_end.min(bytes.len());
            let window = &bytes[window_start..clamped_end];
            let denom = window.len().max(1) as f32;
            let row = out.row_mut(row_index);
            for (column, needle) in self.bytes.iter().copied().enumerate() {
                let count = window.iter().filter(|&&byte| byte == needle).count() as f32;
                row[column] = count / denom;
            }
        }
        Ok(())
    }
}

/// Counts selected bytes from a position until the next newline.
#[derive(Clone, Debug)]
pub struct LineByteCountAppender {
    name: &'static str,
    bytes: Vec<u8>,
}

impl LineByteCountAppender {
    #[must_use]
    pub fn new(name: &'static str, bytes: Vec<u8>) -> Self {
        Self { name, bytes }
    }
}

/// Counts reusable byte classes over a centered byte window.
#[derive(Clone, Debug)]
pub struct ByteClassCountAppender {
    name: &'static str,
    spec: ByteWindowSpec,
    classes: Vec<ByteClass>,
    ascii: AsciiClassTable,
}

impl ByteClassCountAppender {
    #[must_use]
    pub fn new(name: &'static str, spec: ByteWindowSpec, classes: Vec<ByteClass>) -> Self {
        Self {
            name,
            spec,
            classes,
            ascii: AsciiClassTable::default(),
        }
    }
}

impl FeatureAppender<f32> for ByteClassCountAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new(self.name, self.classes.len())
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.classes.len() {
            return Err(FeatureError::new(
                "byte-class-count appender got a mismatched destination view",
            ));
        }

        let left = self.spec.left as isize;
        let right = self.spec.right as isize;
        let bytes = text.bytes();
        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize() as isize;
            let window_start = (center - left).max(0) as usize;
            let window_end = (center + right + 1).max(0) as usize;
            let clamped_end = window_end.min(bytes.len());
            let window = &bytes[window_start..clamped_end];
            let denom = window.len().max(1) as f32;
            let row = out.row_mut(row_index);
            row.fill(0.0);
            for &byte in window {
                for (column, class) in self.classes.iter().copied().enumerate() {
                    if class.matches(byte, &self.ascii) {
                        row[column] += 1.0;
                    }
                }
            }
            for value in row.iter_mut() {
                *value /= denom;
            }
        }
        Ok(())
    }
}

/// Counts reusable byte classes separately on the left and right sides of a position.
#[derive(Clone, Debug)]
pub struct DirectionalByteClassCountAppender {
    name: &'static str,
    spec: ByteWindowSpec,
    classes: Vec<ByteClass>,
    ascii: AsciiClassTable,
}

/// Counts full Unicode General Categories separately on the left and right sides of a position.
#[derive(Clone, Debug)]
pub struct DirectionalUnicodeCategoryCountAppender {
    name: &'static str,
    spec: ByteWindowSpec,
    categories: Vec<UnicodeCategory>,
}

impl DirectionalUnicodeCategoryCountAppender {
    #[must_use]
    pub fn new(name: &'static str, spec: ByteWindowSpec, categories: Vec<UnicodeCategory>) -> Self {
        Self {
            name,
            spec,
            categories,
        }
    }
}

/// Counts coarse Unicode Category Groups separately on the left and right sides of a position.
#[derive(Clone, Debug)]
pub struct DirectionalUnicodeCategoryGroupCountAppender {
    name: &'static str,
    spec: ByteWindowSpec,
    groups: Vec<UnicodeCategoryGroup>,
}

impl DirectionalUnicodeCategoryGroupCountAppender {
    #[must_use]
    pub fn new(
        name: &'static str,
        spec: ByteWindowSpec,
        groups: Vec<UnicodeCategoryGroup>,
    ) -> Self {
        Self { name, spec, groups }
    }
}

impl DirectionalByteClassCountAppender {
    #[must_use]
    pub fn new(name: &'static str, spec: ByteWindowSpec, classes: Vec<ByteClass>) -> Self {
        Self {
            name,
            spec,
            classes,
            ascii: AsciiClassTable::default(),
        }
    }
}

impl FeatureAppender<f32> for DirectionalByteClassCountAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new(self.name, self.classes.len() * 2)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.classes.len() * 2 {
            return Err(FeatureError::new(
                "directional-byte-class-count appender got a mismatched destination view",
            ));
        }

        let left = self.spec.left;
        let right = self.spec.right;
        let bytes = text.bytes();
        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize();
            let left_start = center.saturating_sub(left);
            let left_window = &bytes[left_start..center];
            let right_start = center.saturating_add(1).min(bytes.len());
            let right_end = right_start.saturating_add(right).min(bytes.len());
            let right_window = &bytes[right_start..right_end];
            let left_denom = left_window.len().max(1) as f32;
            let right_denom = right_window.len().max(1) as f32;
            let row = out.row_mut(row_index);
            row.fill(0.0);

            for &byte in left_window {
                for (column, class) in self.classes.iter().copied().enumerate() {
                    if class.matches(byte, &self.ascii) {
                        row[column] += 1.0;
                    }
                }
            }
            for &byte in right_window {
                for (index, class) in self.classes.iter().copied().enumerate() {
                    let column = self.classes.len() + index;
                    if class.matches(byte, &self.ascii) {
                        row[column] += 1.0;
                    }
                }
            }
            let split = self.classes.len();
            for value in &mut row[..split] {
                *value /= left_denom;
            }
            for value in &mut row[split..] {
                *value /= right_denom;
            }
        }
        Ok(())
    }
}

impl FeatureAppender<f32> for DirectionalUnicodeCategoryCountAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new(self.name, self.categories.len() * 2)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.categories.len() * 2 {
            return Err(FeatureError::new(
                "directional-unicode-category-count appender got a mismatched destination view",
            ));
        }

        let Some(text_str) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "directional unicode category features require valid UTF-8 text",
            ));
        };
        let (scalar_starts, categories, _) = decode_unicode_categories(text_str);
        let left = self.spec.left;
        let right = self.spec.right;
        let split = self.categories.len();

        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize();
            let Some(center_scalar) = scalar_index_at_or_before(&scalar_starts, center) else {
                continue;
            };
            let left_start = center_scalar.saturating_sub(left);
            let left_window = &categories[left_start..center_scalar];
            let right_start = (center_scalar + 1).min(categories.len());
            let right_end = right_start.saturating_add(right).min(categories.len());
            let right_window = &categories[right_start..right_end];
            let left_denom = left_window.len().max(1) as f32;
            let right_denom = right_window.len().max(1) as f32;
            let row = out.row_mut(row_index);
            row.fill(0.0);

            for &category in left_window {
                for (column, expected) in self.categories.iter().copied().enumerate() {
                    if category == expected {
                        row[column] += 1.0;
                    }
                }
            }
            for &category in right_window {
                for (index, expected) in self.categories.iter().copied().enumerate() {
                    let column = split + index;
                    if category == expected {
                        row[column] += 1.0;
                    }
                }
            }

            for value in &mut row[..split] {
                *value /= left_denom;
            }
            for value in &mut row[split..] {
                *value /= right_denom;
            }
        }

        Ok(())
    }
}

impl FeatureAppender<f32> for DirectionalUnicodeCategoryGroupCountAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new(self.name, self.groups.len() * 2)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.groups.len() * 2 {
            return Err(FeatureError::new(
                "directional-unicode-category-group-count appender got a mismatched destination view",
            ));
        }

        let Some(text_str) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "directional unicode category-group features require valid UTF-8 text",
            ));
        };
        let (scalar_starts, _, groups) = decode_unicode_categories(text_str);
        let left = self.spec.left;
        let right = self.spec.right;
        let split = self.groups.len();

        for (row_index, position) in positions.data.iter().enumerate() {
            let center = position.as_usize();
            let Some(center_scalar) = scalar_index_at_or_before(&scalar_starts, center) else {
                continue;
            };
            let left_start = center_scalar.saturating_sub(left);
            let left_window = &groups[left_start..center_scalar];
            let right_start = (center_scalar + 1).min(groups.len());
            let right_end = right_start.saturating_add(right).min(groups.len());
            let right_window = &groups[right_start..right_end];
            let left_denom = left_window.len().max(1) as f32;
            let right_denom = right_window.len().max(1) as f32;
            let row = out.row_mut(row_index);
            row.fill(0.0);

            for &group in left_window {
                for (column, expected) in self.groups.iter().copied().enumerate() {
                    if group == expected {
                        row[column] += 1.0;
                    }
                }
            }
            for &group in right_window {
                for (index, expected) in self.groups.iter().copied().enumerate() {
                    let column = split + index;
                    if group == expected {
                        row[column] += 1.0;
                    }
                }
            }

            for value in &mut row[..split] {
                *value /= left_denom;
            }
            for value in &mut row[split..] {
                *value /= right_denom;
            }
        }

        Ok(())
    }
}

impl FeatureAppender<f32> for LineByteCountAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new(self.name, self.bytes.len())
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.bytes.len() {
            return Err(FeatureError::new(
                "line-byte-count appender got a mismatched destination view",
            ));
        }

        let bytes = text.bytes();
        for (row_index, position) in positions.data.iter().enumerate() {
            let start = position.as_usize();
            let rest = &bytes[start..];
            let line_end = rest
                .iter()
                .position(|&byte| byte == b'\n')
                .unwrap_or(rest.len());
            let line = &rest[..line_end];
            let denom = line.len().max(1) as f32;
            let row = out.row_mut(row_index);
            for (column, needle) in self.bytes.iter().copied().enumerate() {
                let count = line.iter().filter(|&&byte| byte == needle).count() as f32;
                row[column] = count / denom;
            }
        }
        Ok(())
    }
}

/// Continuous shape metrics for the current line around each candidate.
#[derive(Clone, Copy, Debug, Default)]
pub struct LineShapeMetricsAppender;

impl LineShapeMetricsAppender {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl FeatureAppender<f32> for LineShapeMetricsAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("line_shape_metrics", 14)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != 14 {
            return Err(FeatureError::new(
                "line-shape-metrics appender got a mismatched destination view",
            ));
        }
        let Some(text) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "line-shape metrics require valid UTF-8 text",
            ));
        };

        for (row_index, position) in positions.data.iter().enumerate() {
            let line = current_line(text, position.as_usize());
            let trimmed = line.trim();
            let leading_ws_bytes = line.len().saturating_sub(line.trim_start().len());
            let mut chars = 0_usize;
            let mut alphabetic = 0_usize;
            let mut uppercase = 0_usize;
            let mut lowercase = 0_usize;
            let mut digits = 0_usize;
            let mut whitespace = 0_usize;
            let mut punctuation = 0_usize;
            let mut symbols = 0_usize;
            let mut quote_chars = 0_usize;
            for ch in line.chars() {
                chars += 1;
                alphabetic += usize::from(ch.is_alphabetic());
                uppercase += usize::from(ch.is_uppercase());
                lowercase += usize::from(ch.is_lowercase());
                digits += usize::from(ch.is_ascii_digit());
                whitespace += usize::from(ch.is_whitespace());
                punctuation += usize::from(ch.is_ascii_punctuation());
                symbols += usize::from(ch.is_ascii_graphic() && !ch.is_ascii_alphanumeric());
                quote_chars += usize::from(matches!(ch, '"' | '\'' | '“' | '”' | '‘' | '’'));
            }
            let denom = chars.max(1) as f32;
            let alpha_denom = alphabetic.max(1) as f32;
            let row = out.row_mut(row_index);
            row[0] = capped_len_feature(line.len(), 1024.0);
            row[1] = capped_len_feature(trimmed.len(), 1024.0);
            row[2] = (leading_ws_bytes as f32 / 64.0).min(1.0);
            row[3] = alphabetic as f32 / denom;
            row[4] = digits as f32 / denom;
            row[5] = whitespace as f32 / denom;
            row[6] = punctuation as f32 / denom;
            row[7] = symbols as f32 / denom;
            row[8] = uppercase as f32 / alpha_denom;
            row[9] = lowercase as f32 / alpha_denom;
            row[10] = quote_chars as f32 / denom;
            row[11] = bool_to_f32(trimmed.ends_with(':'));
            row[12] = bool_to_f32(
                trimmed
                    .chars()
                    .next_back()
                    .is_some_and(|ch| matches!(ch, '.' | ';' | ',')),
            );
            row[13] = bool_to_f32(
                trimmed
                    .chars()
                    .all(|ch| !ch.is_alphabetic() || ch.is_uppercase() || !ch.is_lowercase()),
            );
        }
        Ok(())
    }
}

/// Neighbor-line and coarse document-position metrics for line-level candidates.
#[derive(Clone, Copy, Debug, Default)]
pub struct LineContextMetricsAppender;

impl LineContextMetricsAppender {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl FeatureAppender<f32> for LineContextMetricsAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("line_context_metrics", 22)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != 22 {
            return Err(FeatureError::new(
                "line-context-metrics appender got a mismatched destination view",
            ));
        }
        let Some(text) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "line-context metrics require valid UTF-8 text",
            ));
        };

        for (row_index, position) in positions.data.iter().enumerate() {
            let offset = position.as_usize().min(text.len());
            let line_start = line_start_before_or_at(text, offset);
            let line_end = line_end_after_or_at(text, offset);
            let previous = previous_line_before(text, line_start).unwrap_or_default();
            let next = next_line_after(text, line_end).unwrap_or_default();
            let previous_shape = compact_line_shape(previous);
            let next_shape = compact_line_shape(next);
            let row = out.row_mut(row_index);
            row[0] = bool_to_f32(line_start == 0);
            row[1] = bool_to_f32(line_end >= text.len());
            row[2] = offset as f32 / text.len().max(1) as f32;
            row[3] = bool_to_f32(preceding_whitespace_run_newline_count(text, line_start) >= 2);
            row[4] = bool_to_f32(whitespace_run_newline_count(text, line_end) >= 2);
            row[5] = previous_shape.len;
            row[6] = next_shape.len;
            row[7] = previous_shape.alpha_ratio;
            row[8] = next_shape.alpha_ratio;
            row[9] = previous_shape.digit_ratio;
            row[10] = next_shape.digit_ratio;
            row[11] = previous_shape.upper_ratio;
            row[12] = next_shape.upper_ratio;
            row[13] = previous_shape.punct_ratio;
            row[14] = next_shape.punct_ratio;
            row[15] = bool_to_f32(starts_heading_like(previous));
            row[16] = bool_to_f32(starts_heading_like(next));
            row[17] = bool_to_f32(starts_list_like(previous));
            row[18] = bool_to_f32(starts_list_like(next));
            row[19] = bool_to_f32(has_metadata_colon(previous));
            row[20] = bool_to_f32(has_metadata_colon(next));
            row[21] = bool_to_f32(previous.trim().is_empty() || next.trim().is_empty());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CompactLineShape {
    len: f32,
    alpha_ratio: f32,
    digit_ratio: f32,
    upper_ratio: f32,
    punct_ratio: f32,
}

fn previous_line_before(text: &str, line_start: usize) -> Option<&str> {
    if line_start == 0 {
        return None;
    }
    let mut previous_end = line_start;
    while previous_end > 0 && matches!(text.as_bytes()[previous_end - 1], b'\n' | b'\r') {
        previous_end -= 1;
    }
    let previous_start = line_start_before_or_at(text, previous_end);
    text.get(previous_start..previous_end)
}

fn compact_line_shape(line: &str) -> CompactLineShape {
    let trimmed = line.trim();
    let mut chars = 0_usize;
    let mut alphabetic = 0_usize;
    let mut uppercase = 0_usize;
    let mut digits = 0_usize;
    let mut punctuation = 0_usize;
    for ch in trimmed.chars() {
        chars += 1;
        alphabetic += usize::from(ch.is_alphabetic());
        uppercase += usize::from(ch.is_uppercase());
        digits += usize::from(ch.is_ascii_digit());
        punctuation += usize::from(ch.is_ascii_punctuation());
    }
    let denom = chars.max(1) as f32;
    let alpha_denom = alphabetic.max(1) as f32;
    CompactLineShape {
        len: capped_len_feature(trimmed.len(), 1024.0),
        alpha_ratio: alphabetic as f32 / denom,
        digit_ratio: digits as f32 / denom,
        upper_ratio: uppercase as f32 / alpha_denom,
        punct_ratio: punctuation as f32 / denom,
    }
}

fn capped_len_feature(len: usize, cap: f32) -> f32 {
    ((len as f32 + 1.0).ln() / cap.ln()).min(1.0)
}

/// Encoded byte prefix and suffix of the current trimmed line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LineEdgeByteWindowAppender {
    prefix: usize,
    suffix: usize,
}

impl LineEdgeByteWindowAppender {
    #[must_use]
    pub fn new(prefix: usize, suffix: usize) -> Self {
        Self { prefix, suffix }
    }
}

impl FeatureAppender<f32> for LineEdgeByteWindowAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("line_edge_byte_window", self.prefix + self.suffix)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        let width = self.prefix + self.suffix;
        if out.rows != positions.len() || out.cols != width {
            return Err(FeatureError::new(
                "line-edge-byte-window appender got a mismatched destination view",
            ));
        }
        let Some(text) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "line-edge byte window requires valid UTF-8 text",
            ));
        };

        for (row_index, position) in positions.data.iter().enumerate() {
            let line = current_line(text, position.as_usize()).trim();
            let bytes = line.as_bytes();
            let row = out.row_mut(row_index);
            row.fill(0.0);
            for index in 0..self.prefix.min(bytes.len()) {
                row[index] = normalized_byte_value(bytes[index]);
            }
            for index in 0..self.suffix.min(bytes.len()) {
                row[self.prefix + index] = normalized_byte_value(bytes[bytes.len() - 1 - index]);
            }
        }
        Ok(())
    }
}

fn normalized_byte_value(byte: u8) -> f32 {
    (byte as f32 + 1.0) / 256.0
}

/// Signed feature-hashed byte n-grams from the current trimmed line.
///
/// This block is task-agnostic lexical signal: it does not know any label
/// vocabulary, and it keeps feature width fixed regardless of corpus size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LineByteNgramHashAppender {
    buckets: usize,
    min_n: usize,
    max_n: usize,
}

impl LineByteNgramHashAppender {
    #[must_use]
    pub fn new(buckets: usize, min_n: usize, max_n: usize) -> Self {
        Self {
            buckets,
            min_n,
            max_n,
        }
    }
}

impl FeatureAppender<f32> for LineByteNgramHashAppender {
    fn block(&self) -> FeatureBlock {
        FeatureBlock::new("line_byte_ngram_hash", self.buckets)
    }

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        _scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if self.buckets == 0 || self.min_n == 0 || self.min_n > self.max_n {
            return Err(FeatureError::new(
                "line-byte-ngram-hash appender got an invalid configuration",
            ));
        }
        if out.rows != positions.len() || out.cols != self.buckets {
            return Err(FeatureError::new(
                "line-byte-ngram-hash appender got a mismatched destination view",
            ));
        }
        let Some(text) = text.as_utf8_str() else {
            return Err(FeatureError::new(
                "line-byte-ngram-hash features require valid UTF-8 text",
            ));
        };

        for (row_index, position) in positions.data.iter().enumerate() {
            let bytes = current_line(text, position.as_usize()).trim().as_bytes();
            let row = out.row_mut(row_index);
            row.fill(0.0);
            let mut ngrams = 0_usize;
            for n in self.min_n..=self.max_n.min(bytes.len()) {
                for start in 0..=bytes.len() - n {
                    let hash = hash_lower_ascii_ngram(&bytes[start..start + n], n as u8);
                    let bucket = (hash as usize) % self.buckets;
                    let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
                    row[bucket] += sign;
                    ngrams += 1;
                }
            }
            if ngrams > 0 {
                let scale = (ngrams as f32).sqrt().recip();
                for value in row.iter_mut() {
                    *value *= scale;
                }
            }
        }
        Ok(())
    }
}

fn hash_lower_ascii_ngram(bytes: &[u8], n: u8) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ u64::from(n);
    for &byte in bytes {
        let normalized = if byte.is_ascii_uppercase() {
            byte.to_ascii_lowercase()
        } else {
            byte
        };
        hash ^= u64::from(normalized);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Zero-extra-allocation composite kernel over stable column blocks.
pub struct CompositeFeatureKernel {
    appenders: Vec<Box<dyn FeatureAppender<f32> + Send + Sync>>,
    schema: FeatureSchema,
}

impl CompositeFeatureKernel {
    #[must_use]
    pub fn new(appenders: Vec<Box<dyn FeatureAppender<f32> + Send + Sync>>) -> Self {
        let mut offset = 0;
        let mut blocks = Vec::with_capacity(appenders.len());
        for appender in &appenders {
            let block = appender.block().with_offset(offset);
            offset += block.width;
            blocks.push(block);
        }
        Self {
            appenders,
            schema: FeatureSchema::new(blocks),
        }
    }

    #[must_use]
    pub fn boundary_demo() -> Self {
        Self::new(vec![
            Box::new(ByteWindowAppender::new(ByteWindowSpec::new(1, 1))),
            Box::new(AsciiClassAppender::new()),
            Box::new(BoundaryShapeAppender::new()),
        ])
    }

    #[must_use]
    pub fn format_demo() -> Self {
        Self::new(vec![Box::new(LineByteCountAppender::new(
            "format_counts",
            vec![b'<', b'>', b'/', b'=', b',', b'"'],
        ))])
    }

    #[must_use]
    pub fn schema(&self) -> &FeatureSchema {
        &self.schema
    }
}

impl FeatureKernel<f32> for CompositeFeatureKernel {
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn extract_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        mut out: FeatureMatrixViewMut<'_, f32>,
        scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError> {
        if out.rows != positions.len() || out.cols != self.schema.total_dim() {
            return Err(FeatureError::new(
                "composite kernel got a mismatched destination view",
            ));
        }

        out.fill(0.0);
        for (block, appender) in self.schema.blocks().iter().zip(&self.appenders) {
            let block_view = out.reborrow().subview(block.offset, block.width);
            appender.append_into(text, positions, block_view, scratch)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use charstreamer_core::{
        ByteWindowSpec, CandidateBuffer, CandidateScanner, FeatureAppender, FeatureKernel,
        ScanRange, TextBytes,
    };

    use crate::{
        BoundaryShapeAppender, ByteClass, ByteClassCountAppender, ByteSet256, ByteSetScanner,
        CompositeFeatureKernel, DirectionalByteClassCountAppender,
        DirectionalUnicodeCategoryCountAppender, DirectionalUnicodeCategoryGroupCountAppender,
        EncodedByteWindowAppender, LineByteNgramHashAppender, LineContextMetricsAppender,
        SelectedByteCountAppender, UnicodeCategory, UnicodeCategoryGroup,
    };

    #[test]
    fn composite_kernel_uses_expected_width() {
        let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!\n"));
        let kernel = CompositeFeatureKernel::boundary_demo();
        let text = TextBytes::from_utf8("Alpha. Beta?");
        let mut candidates = CandidateBuffer::new();
        scanner.scan_into(text, ScanRange::full(text), &mut candidates);

        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(candidates.len(), kernel.schema().total_dim());
        let mut scratch = charstreamer_core::FeatureScratch::default();
        kernel
            .extract_into(
                text,
                candidates.as_slice(),
                matrix.as_view_mut(),
                &mut scratch,
            )
            .expect("feature extraction should succeed");

        assert_eq!(kernel.schema().total_dim(), 31);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 31);
    }

    #[test]
    fn selected_byte_count_appender_counts_expected_symbols() {
        let appender =
            SelectedByteCountAppender::new("symbols", ByteWindowSpec::new(0, 7), vec![b'<', b',']);
        let text = TextBytes::from_utf8("<a>,b,c");
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize(0)],
        };
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(1, 2);
        let mut scratch = charstreamer_core::FeatureScratch::default();
        appender
            .append_into(text, positions, matrix.as_view_mut(), &mut scratch)
            .expect("feature extraction should succeed");
        assert!(matrix.data[0] > 0.0);
        assert!(matrix.data[1] > 0.0);
    }

    #[test]
    fn byte_class_count_appender_counts_expected_classes() {
        let appender = ByteClassCountAppender::new(
            "classes",
            ByteWindowSpec::new(0, 5),
            vec![
                ByteClass::AsciiUpper,
                ByteClass::AsciiDigit,
                ByteClass::AsciiWhitespace,
                ByteClass::OpenBracket,
                ByteClass::LineBreak,
            ],
        );
        let text = TextBytes::from_utf8("A9 (\n)");
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize(0)],
        };
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(1, 5);
        let mut scratch = charstreamer_core::FeatureScratch::default();
        appender
            .append_into(text, positions, matrix.as_view_mut(), &mut scratch)
            .expect("feature extraction should succeed");
        assert!(matrix.data.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn directional_byte_class_count_appender_splits_left_and_right_context() {
        let appender = DirectionalByteClassCountAppender::new(
            "directional_classes",
            ByteWindowSpec::new(3, 3),
            vec![ByteClass::AsciiUpper, ByteClass::AsciiLower],
        );
        let text = TextBytes::from_utf8("AA. bb");
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize(2)],
        };
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(1, 4);
        let mut scratch = charstreamer_core::FeatureScratch::default();
        appender
            .append_into(text, positions, matrix.as_view_mut(), &mut scratch)
            .expect("feature extraction should succeed");
        assert!(matrix.data[0] > 0.0);
        assert_eq!(matrix.data[1], 0.0);
        assert_eq!(matrix.data[2], 0.0);
        assert!(matrix.data[3] > 0.0);
    }

    #[test]
    fn encoded_window_appender_uses_expected_width() {
        let appender = EncodedByteWindowAppender::new(ByteWindowSpec::new(5, 3));
        assert_eq!(appender.block().width, 9);
    }

    #[test]
    fn line_byte_ngram_hash_appender_is_case_stable_and_normalized() {
        let appender = LineByteNgramHashAppender::new(64, 3, 4);
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize(0)],
        };
        let mut upper = charstreamer_core::FeatureMatrix::<f32>::default();
        let mut lower = charstreamer_core::FeatureMatrix::<f32>::default();
        upper.resize_zeroed(1, appender.block().width);
        lower.resize_zeroed(1, appender.block().width);
        let mut scratch = charstreamer_core::FeatureScratch::default();

        appender
            .append_into(
                TextBytes::from_utf8("Case Number\nBody"),
                positions,
                upper.as_view_mut(),
                &mut scratch,
            )
            .expect("line n-gram feature extraction should succeed");
        appender
            .append_into(
                TextBytes::from_utf8("case number\nBody"),
                positions,
                lower.as_view_mut(),
                &mut scratch,
            )
            .expect("line n-gram feature extraction should succeed");

        assert_eq!(upper.data, lower.data);
        assert!(upper.data.iter().any(|value| *value != 0.0));
        assert!(upper.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn line_context_metrics_appender_uses_neighbor_lines_and_position() {
        let appender = LineContextMetricsAppender::new();
        let text = TextBytes::from_utf8("TITLE\n\n1. Item\nBody line.\n");
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize("TITLE\n\n".len())],
        };
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(1, appender.block().width);
        let mut scratch = charstreamer_core::FeatureScratch::default();

        appender
            .append_into(text, positions, matrix.as_view_mut(), &mut scratch)
            .expect("line-context feature extraction should succeed");

        assert_eq!(matrix.data[3], 1.0);
        assert_eq!(matrix.data[18], 0.0);
        assert!(matrix.data[2] > 0.0);
        assert!(matrix.data[6] > 0.0);
    }

    #[test]
    fn boundary_shape_appender_detects_quote_absorbing_boundary_shape() {
        let appender = BoundaryShapeAppender::new();
        let text = TextBytes::from_utf8("\"Done.\"\n\nNext.");
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize(5)],
        };
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(1, appender.block().width);
        let mut scratch = charstreamer_core::FeatureScratch::default();
        appender
            .append_into(text, positions, matrix.as_view_mut(), &mut scratch)
            .expect("boundary-shape feature extraction should succeed");
        assert_eq!(matrix.data[0], 1.0);
        assert_eq!(matrix.data[1], 1.0);
        assert_eq!(matrix.data[7], 1.0);
        assert_eq!(matrix.data[14], 1.0);
        assert_eq!(matrix.data[15], 1.0);
    }

    #[test]
    fn directional_unicode_category_count_appender_splits_left_and_right_context() {
        let appender = DirectionalUnicodeCategoryCountAppender::new(
            "unicode_directional_categories",
            ByteWindowSpec::new(3, 3),
            vec![
                UnicodeCategory::Lu,
                UnicodeCategory::Po,
                UnicodeCategory::Ll,
            ],
        );
        let text = TextBytes::from_utf8("Ä.” β");
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize("Ä".len())],
        };
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(1, 6);
        let mut scratch = charstreamer_core::FeatureScratch::default();
        appender
            .append_into(text, positions, matrix.as_view_mut(), &mut scratch)
            .expect("unicode feature extraction should succeed");
        assert!(matrix.data[0] > 0.0);
        assert!(matrix.data[5] > 0.0);
    }

    #[test]
    fn directional_unicode_category_group_count_appender_splits_left_and_right_context() {
        let appender = DirectionalUnicodeCategoryGroupCountAppender::new(
            "unicode_directional_groups",
            ByteWindowSpec::new(3, 3),
            vec![UnicodeCategoryGroup::L, UnicodeCategoryGroup::P],
        );
        let text = TextBytes::from_utf8("Ä.” β");
        let positions = charstreamer_core::CandidateSlice {
            data: &[charstreamer_core::BytePos::from_usize("Ä".len())],
        };
        let mut matrix = charstreamer_core::FeatureMatrix::<f32>::default();
        matrix.resize_zeroed(1, 4);
        let mut scratch = charstreamer_core::FeatureScratch::default();
        appender
            .append_into(text, positions, matrix.as_view_mut(), &mut scratch)
            .expect("unicode group feature extraction should succeed");
        assert!(matrix.data[0] > 0.0);
        assert!(matrix.data[3] > 0.0);
    }
}
