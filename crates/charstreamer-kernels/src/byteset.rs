/// Reusable 256-byte membership table.
#[derive(Clone, Debug)]
pub struct ByteSet256 {
    table: [u8; 256],
    members: Vec<u8>,
}

impl ByteSet256 {
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut table = [0_u8; 256];
        let mut members = Vec::with_capacity(bytes.len());
        for &byte in bytes {
            if table[byte as usize] == 0 {
                table[byte as usize] = 1;
                members.push(byte);
            }
        }
        Self { table, members }
    }

    #[must_use]
    pub fn contains(&self, byte: u8) -> bool {
        self.table[byte as usize] != 0
    }

    #[must_use]
    pub fn members(&self) -> &[u8] {
        &self.members
    }
}

/// Cheap ASCII byte-class table used by feature appenders.
#[derive(Clone, Debug)]
pub struct AsciiClassTable {
    table: [u8; 256],
}

impl Default for AsciiClassTable {
    fn default() -> Self {
        let mut table = [0_u8; 256];
        for byte in u8::MIN..=u8::MAX {
            let mut flags = 0_u8;
            if byte.is_ascii_whitespace() {
                flags |= Self::SPACE;
            }
            if byte.is_ascii_uppercase() {
                flags |= Self::UPPER;
            }
            if byte.is_ascii_lowercase() {
                flags |= Self::LOWER;
            }
            if byte.is_ascii_digit() {
                flags |= Self::DIGIT;
            }
            if byte.is_ascii_punctuation() {
                flags |= Self::PUNCT;
            }
            table[byte as usize] = flags;
        }
        Self { table }
    }
}

impl AsciiClassTable {
    const SPACE: u8 = 1 << 0;
    const UPPER: u8 = 1 << 1;
    const LOWER: u8 = 1 << 2;
    const DIGIT: u8 = 1 << 3;
    const PUNCT: u8 = 1 << 4;

    #[must_use]
    pub fn is_space(&self, byte: u8) -> bool {
        self.table[byte as usize] & Self::SPACE != 0
    }

    #[must_use]
    pub fn is_upper(&self, byte: u8) -> bool {
        self.table[byte as usize] & Self::UPPER != 0
    }

    #[must_use]
    pub fn is_lower(&self, byte: u8) -> bool {
        self.table[byte as usize] & Self::LOWER != 0
    }

    #[must_use]
    pub fn is_digit(&self, byte: u8) -> bool {
        self.table[byte as usize] & Self::DIGIT != 0
    }

    #[must_use]
    pub fn is_alpha(&self, byte: u8) -> bool {
        self.table[byte as usize] & (Self::UPPER | Self::LOWER) != 0
    }

    #[must_use]
    pub fn is_alnum(&self, byte: u8) -> bool {
        self.table[byte as usize] & (Self::UPPER | Self::LOWER | Self::DIGIT) != 0
    }

    #[must_use]
    pub fn is_punct(&self, byte: u8) -> bool {
        self.table[byte as usize] & Self::PUNCT != 0
    }
}
