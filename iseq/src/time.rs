//! Getting the time.

use chrono::offset::Local;

/// Return the current time as String.
pub fn now_string() -> String {
    let time = Local::now();
    let date = time.date().to_string();
    let date = date.split("+").collect::<Vec<&str>>();
    let time = time.time().to_string();
    let time = time.split(".").collect::<Vec<&str>>()[0];
    let time = time.split(":").collect::<Vec<&str>>();
    format!("{}T{}-{}-{}", date[0], time[0], time[1], time[2])
}

/// Return the current time as Tuple.
pub fn now_tuple() -> (usize, usize, usize, usize, usize, usize) {
    let time = Local::now();
    let date = time.date().to_string();
    let date = date.split("+").collect::<Vec<&str>>()[0].to_string();
    let date = date.split("-").map(|e| e.parse().ok().unwrap()).collect::<Vec<usize>>();
    let time = time.time().to_string();
    let time = time.split(".").collect::<Vec<&str>>()[0];
    let time = time.split(":").map(|e| e.parse().ok().unwrap()).collect::<Vec<usize>>();
    (date[0], date[1], date[2], time[0], time[1], time[2])
}