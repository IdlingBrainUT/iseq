/// Utility functions.

use num_traits::Float;

use crate::traits::FloatISeq;

/// Add 1 to a usize value.
pub fn add1(a: usize, lim: usize) -> usize {
    if a + 1 >= lim {
        a
    } else {
        a + 1
    }
}

/// Subtract 1 from a usize value.
pub fn sub1(a: usize) -> usize {
    if a == 0 {
        a
    } else {
        a - 1
    }
}

/// Return the smaller float value.
pub fn minf<T: FloatISeq>(a: T, b: T) -> T {
    if a < b { a }
    else { b } 
}

/// Return the larger float value.
pub fn maxf<T: FloatISeq>(a: T, b: T) -> T {
    if a > b { a }
    else { b } 
}

/// Return indexes in order of smallest element first.
pub fn argsort<T: PartialOrd + Copy>(v: &Vec<T>) -> Vec<usize> {
    let mut vi = v.iter().enumerate().map(|(i, &e)| (e, i)).collect::<Vec<(T, usize)>>();
    vi.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    vi.iter().map(|(_, i)| *i).collect::<Vec<usize>>()
}

/// Rearrange `v` according to the index order shown in `index`. 
pub fn rearrange<T: Copy>(v: &mut Vec<T>, index: &Vec<usize>) {
    let vc = v.clone();
    for (j, &i) in index.iter().enumerate() {
        v[j] = vc[i];
    }
}

/// Enumerate the numbers in the range in 10-fold increments.
pub fn axis_log10<T: Float>(range: (T, T)) -> Vec<T> {
    let (mut tmp, end) = range;
    let ten = T::from(10).unwrap();
    let mut v = vec![tmp];
    loop {
        tmp = tmp * ten;
        if end < tmp {
            if tmp - end < end / ten {
                v.push(tmp);
            }
            break;
        }
        v.push(tmp);
    }

    v
}