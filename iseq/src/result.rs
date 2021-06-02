// Results which will be returned by iSeq functions.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ResultISeq {
    Success,
    UpperEqualsLower,
}

pub fn result_match(r: ResultISeq) {
    match r {
        ResultISeq::Success => {
            println!("-> Success");
        }
        ResultISeq::UpperEqualsLower => {
            println!("-> The upper and lower lambda have same value.");
        }
    };
}