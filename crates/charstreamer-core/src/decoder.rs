use crate::data::{BytePos, ByteSpan, CandidateSlice, LabelAtPos, LabeledSpan, ScoreMatrixView};
use crate::error::DecodeError;
use crate::text::TextBytes;
use crate::traits::{Decoder, ScoreDecoder};

/// Threshold decoder that turns positive candidate scores into byte spans.
#[derive(Clone, Copy, Debug)]
pub struct ThresholdSpanDecoder {
    threshold: f32,
}

impl ThresholdSpanDecoder {
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Decoder<f32, ByteSpan> for ThresholdSpanDecoder {
    fn decode_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        scores: &[f32],
        out: &mut Vec<ByteSpan>,
    ) -> Result<(), DecodeError> {
        if positions.len() != scores.len() {
            return Err(DecodeError::new(
                "positions and scores must have matching lengths",
            ));
        }

        out.clear();
        let mut start = BytePos(0);
        for (position, score) in positions.data.iter().zip(scores.iter().copied()) {
            if score < self.threshold {
                continue;
            }
            let end = BytePos(position.0.saturating_add(1));
            if end > start {
                out.push(ByteSpan { start, end });
                start = end;
            }
        }

        let text_end = BytePos::from_usize(text.len());
        if start < text_end {
            out.push(ByteSpan {
                start,
                end: text_end,
            });
        }

        Ok(())
    }
}

fn argmax_row(row: &[f32]) -> Result<(usize, f32), DecodeError> {
    let mut best_index = None;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in row.iter().copied().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = Some(index);
        }
    }

    best_index
        .map(|index| (index, best_value))
        .ok_or_else(|| DecodeError::new("score rows must be non-empty"))
}

/// Argmax decoder for per-position labels.
#[derive(Clone, Debug)]
pub struct ArgmaxLabelDecoder<L> {
    labels: Vec<L>,
}

impl<L> ArgmaxLabelDecoder<L> {
    #[must_use]
    pub fn new(labels: Vec<L>) -> Self {
        Self { labels }
    }
}

impl<L> ScoreDecoder<f32, LabelAtPos<L>> for ArgmaxLabelDecoder<L>
where
    L: Clone,
{
    fn decode_scores_into(
        &self,
        _text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        scores: ScoreMatrixView<'_, f32>,
        out: &mut Vec<LabelAtPos<L>>,
    ) -> Result<(), DecodeError> {
        if positions.len() != scores.rows {
            return Err(DecodeError::new(
                "positions and score rows must have matching lengths",
            ));
        }
        if scores.cols != self.labels.len() {
            return Err(DecodeError::new(
                "score columns and decoder labels must match",
            ));
        }

        out.clear();
        for (row_index, position) in positions.data.iter().enumerate() {
            let (label_index, score) = argmax_row(scores.row(row_index))?;
            out.push(LabelAtPos {
                position: *position,
                label: self.labels[label_index].clone(),
                score,
            });
        }
        Ok(())
    }
}

/// Groups argmax labels into contiguous byte spans between adjacent positions.
#[derive(Clone, Debug)]
pub struct ContiguousSpanDecoder<L> {
    labels: Vec<L>,
}

impl<L> ContiguousSpanDecoder<L> {
    #[must_use]
    pub fn new(labels: Vec<L>) -> Self {
        Self { labels }
    }
}

impl<L> ScoreDecoder<f32, LabeledSpan<L>> for ContiguousSpanDecoder<L>
where
    L: Clone,
{
    fn decode_scores_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        scores: ScoreMatrixView<'_, f32>,
        out: &mut Vec<LabeledSpan<L>>,
    ) -> Result<(), DecodeError> {
        if positions.len() != scores.rows {
            return Err(DecodeError::new(
                "positions and score rows must have matching lengths",
            ));
        }
        if scores.cols != self.labels.len() {
            return Err(DecodeError::new(
                "score columns and decoder labels must match",
            ));
        }

        out.clear();
        if positions.is_empty() {
            return Ok(());
        }

        let (mut current_index, mut current_score) = argmax_row(scores.row(0))?;
        let mut current_start = positions.data[0];

        for row_index in 1..positions.len() {
            let (label_index, score) = argmax_row(scores.row(row_index))?;
            let boundary = positions.data[row_index];
            if label_index != current_index {
                out.push(LabeledSpan {
                    span: ByteSpan {
                        start: current_start,
                        end: boundary,
                    },
                    label: self.labels[current_index].clone(),
                    score: current_score,
                });
                current_start = boundary;
                current_index = label_index;
                current_score = score;
            } else {
                current_score = current_score.max(score);
            }
        }

        out.push(LabeledSpan {
            span: ByteSpan {
                start: current_start,
                end: BytePos::from_usize(text.len()),
            },
            label: self.labels[current_index].clone(),
            score: current_score,
        });

        Ok(())
    }
}
