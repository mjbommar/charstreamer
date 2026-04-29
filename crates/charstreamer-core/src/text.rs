use crate::data::{BytePos, ScalarPos};

/// Canonical byte-oriented text view.
#[derive(Clone, Copy, Debug)]
pub struct TextBytes<'a> {
    bytes: &'a [u8],
    is_ascii: bool,
    is_utf8: bool,
}

impl<'a> TextBytes<'a> {
    #[must_use]
    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            is_ascii: bytes.is_ascii(),
            is_utf8: std::str::from_utf8(bytes).is_ok(),
        }
    }

    #[must_use]
    pub fn from_utf8(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            is_ascii: text.is_ascii(),
            is_utf8: true,
        }
    }

    #[must_use]
    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    #[must_use]
    pub fn as_utf8_str(self) -> Option<&'a str> {
        std::str::from_utf8(self.bytes).ok()
    }

    #[must_use]
    pub fn len(self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }

    #[must_use]
    pub fn is_ascii(self) -> bool {
        self.is_ascii
    }

    #[must_use]
    pub fn is_utf8(self) -> bool {
        self.is_utf8
    }

    #[must_use]
    pub fn byte(self, index: usize) -> Option<u8> {
        self.bytes.get(index).copied()
    }

    #[must_use]
    pub fn padded_byte(self, index: isize) -> u8 {
        if index < 0 {
            return 0;
        }
        self.byte(index as usize).unwrap_or(0)
    }
}

/// ASCII-only execution view.
#[derive(Clone, Copy, Debug)]
pub struct AsciiByteView<'a> {
    bytes: &'a [u8],
}

impl<'a> AsciiByteView<'a> {
    #[must_use]
    pub fn try_new(text: TextBytes<'a>) -> Option<Self> {
        text.is_ascii().then_some(Self {
            bytes: text.bytes(),
        })
    }

    #[must_use]
    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// Decoded UTF-8 scalar side table.
#[derive(Clone, Debug, Default)]
pub struct Utf8ScalarView {
    pub scalar_starts: Vec<BytePos>,
    pub scalar_values: Vec<u32>,
}

impl Utf8ScalarView {
    #[must_use]
    pub fn decode(text: &str) -> Self {
        let mut scalar_starts = Vec::with_capacity(text.chars().count());
        let mut scalar_values = Vec::with_capacity(text.chars().count());
        for (offset, ch) in text.char_indices() {
            scalar_starts.push(BytePos::from_usize(offset));
            scalar_values.push(ch as u32);
        }
        Self {
            scalar_starts,
            scalar_values,
        }
    }
}

/// Byte-to-scalar index translation for Python-facing edge APIs.
#[derive(Clone, Debug, Default)]
pub struct ByteToCharMap {
    byte_to_scalar: Vec<u32>,
}

impl ByteToCharMap {
    #[must_use]
    pub fn from_utf8(text: &str) -> Self {
        let mut byte_to_scalar = vec![0_u32; text.len() + 1];
        for (scalar_index, (byte_index, ch)) in text.char_indices().enumerate() {
            let start = byte_index;
            let end = byte_index + ch.len_utf8();
            for slot in &mut byte_to_scalar[start..end] {
                *slot = scalar_index as u32;
            }
            byte_to_scalar[end] = (scalar_index + 1) as u32;
        }
        Self { byte_to_scalar }
    }

    #[must_use]
    pub fn scalar_index_of(&self, byte_offset: usize) -> Option<ScalarPos> {
        self.byte_to_scalar.get(byte_offset).copied().map(ScalarPos)
    }
}
