//! I/O functions.

use ndarray::*;
use std::fs;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::str::FromStr;
use num_traits::Float;

/// Read a csv file of X to Array2.
/// 
/// [NOTE] the shape of csv should be (T, N), then return the array of shape (N, T).
pub fn read_csv_to_arr<T>(filepath: &str, header: bool) -> Result<Array2<T>, ShapeError>
where
    T: FromStr + Clone,
{
    let file = File::open(filepath).expect(&format!("File not found!: {}", filepath));
    let mut reader = csv::ReaderBuilder::new().has_headers(header).from_reader(file);
    let mut x: Vec<T> = Vec::new();
    
    let mut rows = 0;
    for result in reader.records() {
        let record = result.expect(&format!("File cannot open!: {}", filepath));
        for r in record.iter() {
            x.push(r.parse().ok().unwrap());
        }
        rows += 1;
    }
    let cols = x.len() / rows;
    let a = Array1::from(x).into_shape((rows, cols))?;
    Ok(a.t().to_owned())
}

/// Trait to overwrite the exisiting array with the csv contents.
pub trait CsvArray {
    /// Overwrite with the csv contents.
    fn read_csv_inplace(&mut self, filepath: &str, header: bool);
}

impl<A: FromStr + Clone, S: DataMut<Elem = A>> CsvArray for ArrayBase<S, Ix2> {
    fn read_csv_inplace(&mut self, filepath: &str, header: bool) {
        let file = File::open(filepath).expect(&format!("File not found!: {}", filepath));
        let mut reader = csv::ReaderBuilder::new().has_headers(header).from_reader(file);
    
        for (result, mut self_sub) in reader.records()
                                .zip(self.axis_iter_mut(Axis(0)))
        {
            let record = result.expect(&format!("File cannot open!: {}", filepath));
            for (ri, si) in record.iter().zip(self_sub.iter_mut()) {
                *si = ri.parse().ok().unwrap();
            }
        }
    }
}

/// The Array can be saved to a csv file.
#[allow(bare_trait_objects)]
pub trait SaveToCSV {
    /// Core part of save function.
    fn save_to_csv_core(&self, output: &mut BufWriter<File>) -> Result<(), Box<std::error::Error>>;
    /// Save the array to csv without header.
    fn save_to_csv(&self, filename: &str) -> Result<(), Box<std::error::Error>>;
    /// Save the array to csv with header.
    fn save_to_csv_with_header(&self, filename: &str, header: &str) -> Result<(), Box<std::error::Error>>;
}

#[allow(bare_trait_objects)]
impl<A, S> SaveToCSV for ArrayBase<S, Ix2>
where
    A: Float + ToString,
    S: Data<Elem = A>,
{
    fn save_to_csv_core(&self, output: &mut BufWriter<File>) -> Result<(), Box<std::error::Error>> {
        let shape = self.shape();
        let y_size = shape[0];
        let x_size = shape[1];
        for y in 0..y_size {
            let a1 = self.slice(s![y, ..]);
            let mut s = String::new();
            for (i, &e) in a1.iter().enumerate() {
                s = s + &(e.to_string());
                if i == x_size - 1 { s = s + "\n"; }
                else { s = s + ","; }
            }
            output.write_all(s.as_bytes())?;
        }
        Ok(())
    }

    fn save_to_csv(&self, filename: &str) -> Result<(), Box<std::error::Error>> {
        let mut output = BufWriter::new(File::create(filename)?);
        self.save_to_csv_core(&mut output)?;
        Ok(())
    }

    fn save_to_csv_with_header(&self, filename: &str, header: &str) -> Result<(), Box<std::error::Error>> {
        let mut output = BufWriter::new(File::create(filename)?);
        output.write_all(format!("{}\n", header).as_bytes())?;
        self.save_to_csv_core(&mut output)?;
        Ok(())
    }
}

/// Make directory.
pub fn mkdir(dirname: &str) {
    let _ = fs::create_dir(dirname);
}