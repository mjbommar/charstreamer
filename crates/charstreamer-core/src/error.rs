use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureError {
    message: String,
}

impl FeatureError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for FeatureError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for FeatureError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PredictError {
    message: String,
}

impl PredictError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for PredictError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for PredictError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FitError {
    message: String,
}

impl FitError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for FitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for FitError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeError {
    message: String,
}

impl DecodeError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for DecodeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for DecodeError {}

#[derive(Debug)]
pub enum PipelineError {
    Feature(FeatureError),
    Predict(PredictError),
    Decode(DecodeError),
}

impl Display for PipelineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Feature(error) => write!(f, "feature extraction failed: {error}"),
            Self::Predict(error) => write!(f, "prediction failed: {error}"),
            Self::Decode(error) => write!(f, "decode failed: {error}"),
        }
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Feature(error) => Some(error),
            Self::Predict(error) => Some(error),
            Self::Decode(error) => Some(error),
        }
    }
}

impl From<FeatureError> for PipelineError {
    fn from(error: FeatureError) -> Self {
        Self::Feature(error)
    }
}

impl From<PredictError> for PipelineError {
    fn from(error: PredictError) -> Self {
        Self::Predict(error)
    }
}

impl From<DecodeError> for PipelineError {
    fn from(error: DecodeError) -> Self {
        Self::Decode(error)
    }
}
