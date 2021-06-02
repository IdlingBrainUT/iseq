//! Assertion macro.

/// Assert when the 2 values are not close.
#[macro_export]
macro_rules! assert_close_l2 {
    ($a:expr, $b:expr, $tol:expr, $t:ty) => {
        let n2: $t = $a.iter().zip($b.iter()).map(|(&ai, &bi)| {
            (ai - bi) * (ai - bi)
        }).sum();
        assert!(n2.sqrt() < $tol);
    };
}