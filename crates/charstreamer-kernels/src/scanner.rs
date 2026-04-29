use charstreamer_core::{BytePos, CandidateBuffer, CandidateScanner, ScanRange, TextBytes};
use memchr::{memchr, memchr2, memchr3};

use crate::ByteSet256;

/// Portable scanner with cheap `memchr` specialization for small byte sets.
#[derive(Clone, Debug)]
pub struct ByteSetScanner {
    bytes: ByteSet256,
}

impl ByteSetScanner {
    #[must_use]
    pub fn new(bytes: ByteSet256) -> Self {
        Self { bytes }
    }
}

impl CandidateScanner for ByteSetScanner {
    fn scan_into(&self, text: TextBytes<'_>, range: ScanRange, out: &mut CandidateBuffer) {
        out.clear();

        let start = range.start.as_usize();
        let end = range.end.as_usize();
        let haystack = &text.bytes()[start..end];
        match self.bytes.members() {
            [] => {}
            [a] => {
                let mut offset = 0;
                while let Some(found) = memchr(*a, &haystack[offset..]) {
                    let index = start + offset + found;
                    out.push(BytePos::from_usize(index));
                    offset += found + 1;
                }
            }
            [a, b] => {
                let mut offset = 0;
                while let Some(found) = memchr2(*a, *b, &haystack[offset..]) {
                    let index = start + offset + found;
                    out.push(BytePos::from_usize(index));
                    offset += found + 1;
                }
            }
            [a, b, c] => {
                let mut offset = 0;
                while let Some(found) = memchr3(*a, *b, *c, &haystack[offset..]) {
                    let index = start + offset + found;
                    out.push(BytePos::from_usize(index));
                    offset += found + 1;
                }
            }
            _ => {
                for (offset, &byte) in haystack.iter().enumerate() {
                    if self.bytes.contains(byte) {
                        out.push(BytePos::from_usize(start + offset));
                    }
                }
            }
        }
    }
}

/// UTF-8 character scanner that emits byte offsets at matching scalar starts.
#[derive(Clone, Debug)]
pub struct Utf8CharSetScanner {
    chars: Vec<char>,
}

impl Utf8CharSetScanner {
    #[must_use]
    pub fn new(mut chars: Vec<char>) -> Self {
        chars.sort_unstable();
        chars.dedup();
        Self { chars }
    }

    fn contains(&self, ch: char) -> bool {
        self.chars.binary_search(&ch).is_ok()
    }
}

impl CandidateScanner for Utf8CharSetScanner {
    fn scan_into(&self, text: TextBytes<'_>, range: ScanRange, out: &mut CandidateBuffer) {
        out.clear();
        let Some(text) = text.as_utf8_str() else {
            return;
        };

        let start = range.start.as_usize();
        let end = range.end.as_usize().min(text.len());
        for (byte_index, ch) in text.char_indices() {
            if byte_index < start {
                continue;
            }
            if byte_index >= end {
                break;
            }
            if self.contains(ch) {
                out.push(BytePos::from_usize(byte_index));
            }
        }
    }
}

/// Emits every `stride` bytes starting at the beginning of the range.
#[derive(Clone, Copy, Debug)]
pub struct StrideScanner {
    stride: usize,
}

impl StrideScanner {
    #[must_use]
    pub fn new(stride: usize) -> Self {
        assert!(stride > 0, "stride must be greater than zero");
        Self { stride }
    }
}

impl CandidateScanner for StrideScanner {
    fn scan_into(&self, text: TextBytes<'_>, range: ScanRange, out: &mut CandidateBuffer) {
        out.clear();
        let end = range.end.as_usize().min(text.len());
        let mut offset = range.start.as_usize();
        while offset < end {
            out.push(BytePos::from_usize(offset));
            offset += self.stride;
        }
    }
}

/// Emits byte positions at the start of the buffer and after each newline.
#[derive(Clone, Copy, Debug, Default)]
pub struct LineStartScanner;

impl LineStartScanner {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl CandidateScanner for LineStartScanner {
    fn scan_into(&self, text: TextBytes<'_>, range: ScanRange, out: &mut CandidateBuffer) {
        out.clear();
        let start = range.start.as_usize();
        let end = range.end.as_usize().min(text.len());
        if start >= end {
            return;
        }

        out.push(BytePos::from_usize(start));
        let haystack = &text.bytes()[start..end];
        let mut offset = 0;
        while let Some(found) = memchr(b'\n', &haystack[offset..]) {
            let next = start + offset + found + 1;
            if next < end {
                out.push(BytePos::from_usize(next));
            }
            offset += found + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use charstreamer_core::{CandidateScanner, ScanRange, TextBytes};

    use crate::{ByteSet256, ByteSetScanner, LineStartScanner, StrideScanner, Utf8CharSetScanner};

    #[test]
    fn scanner_finds_expected_positions() {
        let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".!?"));
        let text = TextBytes::from_utf8("A. B! C?");
        let mut out = charstreamer_core::CandidateBuffer::new();
        scanner.scan_into(text, ScanRange::full(text), &mut out);
        let actual: Vec<usize> = out
            .positions()
            .iter()
            .map(|position| position.as_usize())
            .collect();
        assert_eq!(actual, vec![1, 4, 7]);
    }

    #[test]
    fn stride_scanner_emits_regular_offsets() {
        let scanner = StrideScanner::new(3);
        let text = TextBytes::from_utf8("abcdefghij");
        let mut out = charstreamer_core::CandidateBuffer::new();
        scanner.scan_into(text, ScanRange::full(text), &mut out);
        let actual: Vec<usize> = out
            .positions()
            .iter()
            .map(|position| position.as_usize())
            .collect();
        assert_eq!(actual, vec![0, 3, 6, 9]);
    }

    #[test]
    fn line_start_scanner_emits_line_boundaries() {
        let scanner = LineStartScanner::new();
        let text = TextBytes::from_utf8("one\ntwo\nthree");
        let mut out = charstreamer_core::CandidateBuffer::new();
        scanner.scan_into(text, ScanRange::full(text), &mut out);
        let actual: Vec<usize> = out
            .positions()
            .iter()
            .map(|position| position.as_usize())
            .collect();
        assert_eq!(actual, vec![0, 4, 8]);
    }

    #[test]
    fn utf8_char_scanner_emits_matching_scalar_starts() {
        let scanner = Utf8CharSetScanner::new(vec!['.', '”', '?']);
        let text = TextBytes::from_utf8("A.” B?");
        let mut out = charstreamer_core::CandidateBuffer::new();
        scanner.scan_into(text, ScanRange::full(text), &mut out);
        let actual: Vec<usize> = out
            .positions()
            .iter()
            .map(|position| position.as_usize())
            .collect();
        assert_eq!(actual, vec![1, 2, 7]);
    }
}
