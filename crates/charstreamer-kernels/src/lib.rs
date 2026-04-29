//! Reusable scanning and feature kernels for `charstreamer`.

mod byteset;
mod features;
mod legacy;
mod scanner;

pub use byteset::{AsciiClassTable, ByteSet256};
pub use features::{
    AsciiClassAppender, BoundaryShapeAppender, ByteClass, ByteClassCountAppender,
    ByteWindowAppender, CompositeFeatureKernel, DirectionalByteClassCountAppender,
    DirectionalUnicodeCategoryCountAppender, DirectionalUnicodeCategoryGroupCountAppender,
    EncodedByteWindowAppender, LineByteCountAppender, LineByteNgramHashAppender,
    LineContextMetricsAppender, LineEdgeByteWindowAppender, LineShapeMetricsAppender,
    SelectedByteCountAppender, UnicodeCategory, UnicodeCategoryGroup,
};
pub use legacy::{CharBoundaryLegacyAppender, LegacyFeatureTables};
pub use scanner::{ByteSetScanner, LineStartScanner, StrideScanner, Utf8CharSetScanner};
