/// Trait of float value for iSeq.

use std::fmt::{Debug, Display, LowerExp};
use std::iter::Sum;
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};
use std::str::FromStr;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use rand::distributions::uniform::SampleUniform;

pub trait FloatISeq:
    Float + FromPrimitive 
    + FromStr + ToString 
    + Sum 
    + SampleUniform
    + ScalarOperand
    + Debug + Display + LowerExp
    + AddAssign + SubAssign + MulAssign + DivAssign
    + 'static {}

impl FloatISeq for f64 {}
impl FloatISeq for f32 {}