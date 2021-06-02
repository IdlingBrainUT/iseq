//! Define Errors

use std::error;
use std::fmt;

/// Errors of iSeq calculation.
#[derive(Debug)]
pub enum ISeqError {
    /// The value was assigned to outside of the matrix.
    AssignOutOfRange { shape: (usize, usize), index: (usize, usize) },
    /// Elements were shifted to outside of the padded matrix.
    ShiftOutOfRange { pad_size: usize, shift: usize },
    /// A ratio that is too large or too small was specified.
    RateOutOfRange { valid: (f64, f64), val: f64 },
    /// An unusual range was defined.
    InvalidRange { range: (f64, f64) },
    /// An array of inappropriate lengths was used for calculation.
    InvalidLength { length: usize },
    /// Axis that is too large was specified.
    AxisOutOfRange { valid: usize, val: usize },
    /// The number of process is too small.
    InsufficientProcs { procs: usize, needed: usize },
}

impl fmt::Display for ISeqError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ISeqError::AssignOutOfRange { shape, index } => {
                write!(f, "Assign out of range!: shape={:?}, assigned to {:?}",
                shape, index)
            }
            ISeqError::ShiftOutOfRange { pad_size, shift } => {
                write!(f, "Shift out of range!: padding_size={}, shift_size={}",
                pad_size, shift)
            }
            ISeqError::RateOutOfRange { valid, val } => {
                write!(f, "Rate out of range!: valid_range={:?}, specified_value={}",
                valid, val)
            }
            ISeqError::InvalidRange { range } => {
                write!(f, "Invalid range!: start={:3}, end={:3}",
                range.0, range.1)
            }
            ISeqError::InvalidLength { length } => {
                write!(f, "Invalid length!: length={}", length)
            }
            ISeqError::AxisOutOfRange { valid, val } => {
                write!(f, "Axis out of range!: Axis<={}, given {}", valid, val)
            }
            ISeqError::InsufficientProcs { procs, needed } => {
                write!(f, "The number of process is too small!: procs={}, needed={}", procs, needed)
            }
        }
    }
}

impl error::Error for ISeqError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}