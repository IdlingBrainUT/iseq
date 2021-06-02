//! Trait of numerics for MPI.

use mpi::datatype::Equivalence;
use iseq_linalg::traits::FloatISeq;

/// Trait of numerics for MPI.
pub trait FloatISeqMPI: FloatISeq + Equivalence {}

impl FloatISeqMPI for f64 {}
impl FloatISeqMPI for f32 {}