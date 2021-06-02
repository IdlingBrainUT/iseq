//! Calculation environment.

use ndarray::*;
use std::cmp::{max, min};
use iseq_linalg::lambda::*;
use iseq_linalg::random::*;
use iseq_linalg::stats::*;
use iseq_linalg::traits::*;

use crate::args::*;
use crate::io::*;
use crate::result::*;
use crate::iseq::*;
use crate::time::*;

#[macro_export]
macro_rules! def_env {
    ($(#[$attr:meta])* struct $name:ident, $tr:ident) => {
        pub struct $name<T: $tr> {
            /// iSeq
            pub iseq: ISeq<T>,

            /// Path of csv file of X.
            pub filepath: String,

            /// The number of patterns (K).
            pub k: usize,
            /// The length of time window of patterns (L).
            pub l: usize,

            /// Zero threshold.
            /// 
            /// Values less than z_th are considered as zero.
            pub z_th: T,

            /// The maximum number of times the update calculation can be done.
            pub max_iter: usize,
            /// How many calculations are performed for the same value of lambda.
            pub iter_per_lambda: usize,
            /// To what extent do you ignore the difference between the numbers below.
            pub tolerance: T,
            /// Range of lambda to be calculated.
            pub lambda_range: (T, T),
            /// Seed values for the random number generator.
            pub random_seed: u32,
            /// Save reconstruction error matrix & x_hat as csv file.
            pub error_save: bool,

            /// Path of the csv file of M.
            pub maskpath: Option<String>,

            /// The name of directory which files are saved.
            pub dirname: String,
        }
        
        impl<T: $tr> $name<T> {
            /// Create an Env struct.
            pub fn new(
                filepath: &str, header: bool, k: usize, l: usize, z_th: T,
                max_iter: usize, iter_per_lambda: usize, tolerance: T,
                lambda_start: T, lambda_end: T, lambda_once: T,
                random_seed: u32, verbose: bool, error_save: bool, maskpath_ref: Option<&String>, n_null: usize,
                sd_significant: T, n_significant: usize, p_integrate: T, 
                time: &str,
            ) -> Self {
                let iseq = ISeq::new(
                    filepath, header, k, l, z_th,  
                    max_iter, tolerance, n_null, 
                    sd_significant, n_significant, p_integrate, 
                    verbose
                );
                let filepath = filepath.to_string();
                let lambda_range = if lambda_once.is_infinite() {
                    (lambda_start, lambda_end)
                } else {
                    (lambda_once, lambda_once)
                };
                let dirname = filepath.split(".").collect::<Vec<&str>>()[0].to_owned() + "_" + time;
                let maskpath = match maskpath_ref {
                    Some(path) => Some(path.to_owned()),
                    None => None,
                };
                Self {
                    iseq,
                    filepath, k, l, z_th, max_iter, iter_per_lambda,
                    tolerance, lambda_range, random_seed, error_save, maskpath, dirname,
                }
            }
  
            /// Create an Env struct from args.
            pub fn from_args(a: &Args<T>, time: &str) -> Self {
                let filepath = a.filepath.to_owned();
                let header = a.header;
                let k = a.k;
                let l = a.l;
                let z_th = a.z_th;
                let max_iter = a.max_iter;
                let iter_per_lambda = a.iter_per_lambda;
                let tolerance = a.tolerance;
                let (lambda_start, lambda_end) = (a.lambda_start, a.lambda_end);
                let lambda_once = a.lambda_once;
                let random_seed = a.random_seed;
                let verbose = a.verbose;
                let error_save = a.error_save;
                let maskpath = a.maskpath.as_ref();
                let n_null = a.n_null;
                let sd_significant = a.sd_significant;
                let n_significant = a.n_significant;
                let p_integrate = a.p_integrate; 
                Self::new(
                    &filepath, header, k, l, z_th,
                    max_iter, iter_per_lambda, tolerance,
                    lambda_start, lambda_end, lambda_once,
                    random_seed, verbose, error_save, maskpath, n_null,
                    sd_significant, n_significant, p_integrate, 
                    time
                )
            }

            /// Initialize iSeq & run iSeq up to max_iter times.
            pub fn run_iseq(&mut self, lambda: T) {
                self.iseq.init(self.k, lambda, self.random_seed);
                self.iseq.solve();
            }

            /// Solve iSeq for all the given lambda and return the results
            pub fn calc_costs(&mut self, lambda_vec_run: &Vec<T>) -> (Vec<T>, Vec<T>, Vec<u32>, Vec<usize>, Vec<usize>) {
                let n = lambda_vec_run.len();
                let mut recon_vec = Vec::with_capacity(n);
                let mut xorth_vec = Vec::with_capacity(n);
                let mut seed_vec = Vec::with_capacity(n);
                let mut k_vec = Vec::with_capacity(n);
                let mut valid_vec = Vec::with_capacity(n);
                for &lambda in lambda_vec_run.iter() {      
                    self.iseq.reset_mask(self.maskpath.as_ref());              
                    self.run_iseq(lambda);
                    recon_vec.push(self.iseq.recon_cost_mask(false));
                    xorth_vec.push(self.iseq.xortho_cost());
                    seed_vec.push(self.iseq.random_seed);
                    k_vec.push(self.iseq.k);
                    let sig = self.iseq.significant();
                    valid_vec.push(sig.len());

                    self.random_seed = incr_seed(self.random_seed);
                }
                (recon_vec, xorth_vec, seed_vec, k_vec, valid_vec)
            }

            /// Returns "dirname/filename" for save functions.
            pub fn filename(&self) -> String {
                let f_core0: Vec<&str> = self.filepath.split(".").collect();
                let f_core1: Vec<&str> = f_core0[0].split("/").collect();
                let f_core1_len = f_core1.len();
                self.dirname.to_owned() + "/" + f_core1[f_core1_len - 1]
            }

            /// Save the array of results.
            #[allow(bare_trait_objects)]
            pub fn save_costs<S: Data<Elem = T>>(
                &self, save_arr: &ArrayBase<S, Ix2>
            ) -> Result<(), Box<std::error::Error>> {
                let now = &now_string();
                println!("costs are saved at -> {}", now);
                let f = self.filename();
                let filename = f.to_owned() + "_cost_" + now + ".csv";
                let header = "Lambda,LambdaCalc,RankID,Recon,Xortho,Seed,K,significantK";
                save_arr.save_to_csv_with_header(&filename, header)?;
                Ok(())
            }

            /// Search appropriate range of lambda based on the array of results.
            pub fn search_lambda_range<S: Data<Elem = T>>(save_arr: &ArrayBase<S, Ix2>) -> (usize, (T, T)) {
                let k_new = save_arr.slice(s![.., 7]).max_axis(Axis(0)).into_scalar().to_usize().unwrap();
                let n = save_arr.shape()[0];
                if n <= 1 {
                    panic!()
                }

                let mut index_range = (0, n-1);
                for i in 1..n {
                    let ki = save_arr[[i, 6]].to_usize().unwrap();
                    if ki <= k_new {
                        index_range = if ki == k_new {
                            (max(i-1, 0), min(i+1, n-1))
                        } else {
                            (max(i-1, 0), i)
                        };
                        break;
                    }
                }

                if index_range.0 == index_range.1 {
                    if index_range.0 == 0 {
                        index_range.1 += 1;
                    } else if index_range.0 == n - 1 {
                        index_range.0 -= 1;
                    } else {
                        index_range.0 -= 1;
                        index_range.1 += 1
                    }
                }

                if index_range.1 - index_range.0 == 1 {
                    if save_arr[[index_range.0, 0]] > save_arr[[index_range.0, 1]] {
                        index_range.0 -= 1;
                    } else if save_arr[[index_range.1, 0]] < save_arr[[index_range.1, 1]] {
                        index_range.1 += 1;
                    } else if index_range.1 < n - 1 {
                        if save_arr[[index_range.1, 6]] == save_arr[[index_range.1+1, 6]] {
                            index_range.1 += 1;
                        }
                    }
                }

                (k_new, (save_arr[[index_range.0, 0]], save_arr[[index_range.1, 0]]))
            }

            /// Output most appropriate result and returns the task ID which made it.
            pub fn find_final_result<S: Data<Elem = T>>(&self, save_arr: &ArrayBase<S, Ix2>) -> usize {
                let n = save_arr.shape()[0];
                let mut v = vec![0usize; n];
                let mut sig_rate = Array1::zeros(n);
                Zip::from(&mut sig_rate)
                    .and(save_arr.slice(s![.., 6]))
                    .and(save_arr.slice(s![.., 7]))
                    .apply(|a, &b, &c| {
                        *a = c / b;
                    });
                let rate_max = sig_rate.max_axis(Axis(0)).into_scalar();
                for i in 0..n {
                    if sig_rate[i] ==  rate_max {
                        v[i] = 1;
                    }
                }

                let mut m = T::infinity();
                let mut i_final = 0;
                for i in 0..n {
                    let rc = save_arr[[i, 3]];
                    if v[i] == 1 && m > rc {
                        m = rc;
                        i_final = i;
                    }
                }

                println!("# final result");
                println!("lambda = {}", save_arr[[i_final, 0]]);
                println!("reconstruction error = {}", save_arr[[i_final, 3]]);
                println!("xortho error = {}", save_arr[[i_final, 4]]);
                println!("{} patterns are detected!", save_arr[[i_final, 6]].to_usize().unwrap());
                
                save_arr[[i_final, 2]].to_usize().unwrap()
            }
        }
    };
} // def_env!

def_env!(
    /// Environment for series calculation.
    struct Env,
    FloatISeq
);

impl<T: FloatISeq> Env<T> {
    /// Find appropriate range of lambda based on the number of significant sequences.
    #[allow(bare_trait_objects)]
    pub fn narrow_lambda(&mut self) -> Result<ResultISeq, Box<std::error::Error>> {
        self.narrow_core(NarrowMode::Narrow)
    }

    /// Find appropriate value of lambda and output its result based on reconstruction accuracy.
    #[allow(bare_trait_objects)]
    pub fn final_result(&mut self) -> Result<ResultISeq, Box<std::error::Error>> {
        self.narrow_core(NarrowMode::Final(1))
    }

    /// Run iSeq only 1 lambda value (iter_per_lambda parallel)
    #[allow(bare_trait_objects)]
    pub fn run_once(&mut self, lambda_once: T) -> Result<ResultISeq, Box<std::error::Error>> {
        self.narrow_core(NarrowMode::Once(lambda_once))
    }

    #[allow(bare_trait_objects)]
    pub fn narrow_core(&mut self, mode: NarrowMode<T>) -> Result<ResultISeq, Box<std::error::Error>> {
        let lambda_vec = match lambda_vec(self.lambda_range, mode) {
            Ok(v) => v,
            Err(e) => panic!(e),
        };
        
        let n_lambda = lambda_vec.len();
        let lambda_vec_run = match mode {
            NarrowMode::Final(_) => lambda_vec.clone(),
            _ => lambda_vec_all(&lambda_vec, self.iter_per_lambda),
        };
        let (recon_vec, xortho_vec, seed_vec, k_vec, valid_vec) = self.calc_costs(&lambda_vec_run);
        let mut save_arr = Array2::zeros((n_lambda, 8));
        
        rearrange_costs(&mut save_arr, &lambda_vec, &lambda_vec_run, &recon_vec, &xortho_vec, &seed_vec, &k_vec, &valid_vec);
        let _ = match mode {
            NarrowMode::Once(_) => Ok(()),
            _ => self.save_costs(&save_arr),
        }?;

        match mode {
            NarrowMode::Narrow => {
                let (k, r) = Self::search_lambda_range(&save_arr);
                self.lambda_range = r;
                self.k = k;
            }
            NarrowMode::Final(_) => {
                let i = self.find_final_result(&save_arr);

                println!("Re-calculate final result...");
                self.random_seed = seed_vec[i];
                self.run_iseq(lambda_vec_run[i]);
                let _ = self.iseq.save_wh(&self.filename())?;

                println!("Matrices W and H are saved.");

                if self.error_save {
                    self.iseq.save_error(&self.filename())?;
                }   
            }
            NarrowMode::Once(_) => {
                println!("# final result");
                println!("lambda = {}", save_arr[[0, 0]]);
                println!("reconstruction error = {}", save_arr[[0, 3]]);
                println!("xortho error = {}", save_arr[[0, 4]]);
                println!("{} patterns are detected!", save_arr[[0, 6]].to_usize().unwrap());

                println!("Re-calculate final result...");
                self.random_seed = save_arr[[0, 5]].to_u32().unwrap();
                self.run_iseq(self.lambda_range.0);
                let _ = self.iseq.save_wh(&self.filename())?;

                println!("Matrices W and H are saved.");
            }
        }

        Ok(ResultISeq::Success)
    }
}