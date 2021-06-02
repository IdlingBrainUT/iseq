//! Get the parameter settings from args.

use std::env;
use iseq_linalg::traits::FloatISeq;

/// Preserving parameter settings.
pub struct Args<T: FloatISeq> {
    /// -f: Path of the csv file of X.
    /// 
    /// [NOTE] The shape of csv should be (T, N), because usually T >> N.
    /// 
    /// (default) "readtest.csv"
    pub filepath: String,

    /// -h: Whether the csv file has a header data or not.
    /// 
    /// [NOTE] Even if it does, it can have up to one line.
    /// 
    /// (default) false
    pub header: bool,

    /// -k: Maximum number of patterns (K).
    /// 
    /// (default) 20
    pub k: usize,

    /// -l: The length of time window of patterns (L).
    /// 
    /// (default) 50
    pub l: usize,

    /// -zth: threshold of non-zero values.
    /// 
    /// (default) 0.001
    pub z_th: T,

    /// -mi: The maximum number of times the update calculation can be done.
    /// 
    /// (default) 30
    pub max_iter: usize,

    /// -ipl: How many calculations are performed for the same value of lambda.
    /// 
    /// (defalut) 5
    pub iter_per_lambda: usize,

    /// -tl: To what extent do you ignore the difference between the numbers below.
    /// 
    /// (default) negative infinity
    pub tolerance: T,

    /// -ls: The minimum value for narrowing lambda.
    /// 
    /// [NOTE] This value must be set as a logarithm with base 10.
    /// 
    /// [EXAMPLE] If you want to set 0.01 -> -ls -2
    /// 
    /// (default) -6
    pub lambda_start: T,

    /// -le: The maximum value for narrowing lambda.
    /// 
    /// [NOTE] This value must be set as a logarithm with base 10.
    /// 
    /// [EXAMPLE] If you want to set 1.0 -> -le 0
    /// 
    /// (default) 0
    pub lambda_end: T,

    /// -rs: Seed values for the random number generator.
    /// 
    /// [NOTE] One of the 32 bit number (0-4294967295) must be set.
    /// 
    /// If not, this program will use default seed 0, and will output the same result for the same input every time.
    /// 
    /// (default) 0
    pub random_seed: u32,

    /// -lo: If this parameter is not infinity, do not search for lambda, and only run one seqNMF with this lambda value.
    /// 
    /// [NOTE] If this parameter is not infinity, the value of lambda_start and lambda_end will be ignored.
    /// 
    /// [NOTE] Please specify as a real number.
    /// 
    /// (default) infinity
    pub lambda_once: T,

    /// -nu: The number of null-neurons.
    /// 
    /// (default) 100
    pub n_null: usize,

    /// -mp: Path of the csv file of M.
    /// 
    /// [NOTE] The shape of csv should be (T, N) -> Same size as X.
    /// 
    /// (defalut) None
    pub maskpath: Option<String>,

    /// -ssi: Multiplier for standard deviation of null-neurons' activity 
    /// to set significance level.
    /// 
    /// (default) 5.0
    pub sd_significant: T,

    /// -nsi: The minimum number of significant cells contained in significant sequences.
    /// 
    /// (defalut) 20
    pub n_significant: usize,

    /// -pin: Significance level for integrating similar sequences.
    /// 
    /// (defalut) 0.05
    pub p_integrate: T,

    /// -v: Output the calculation process.
    /// 
    /// (default) false
    pub verbose: bool,

    /// -es: Save reconstruction error matrix & x_hat as csv file.
    /// 
    /// (default) false
    pub error_save: bool,
}

impl<T: FloatISeq> Args<T> {
    /// Get the parameter settings from args.
    pub fn new() -> Self {
        let one = T::one();
        let two = one + one;
        let ten = two * two * two + two;
        let mut filepath = "readtest.csv".to_string();
        let mut header = false;
        let mut k = 20;
        let mut l = 50;
        let mut z_th = T::from(0.001).unwrap();
        let mut max_iter = 30;
        let mut iter_per_lambda = 5;
        let mut tolerance = T::neg_infinity();
        let mut ls = -6;
        let mut le = 0;
        let mut random_seed = 0;
        let mut lambda_once = T::infinity();
        let mut n_null = 100;
        let mut maskpath = None;
        let mut sd_significant = T::from_usize(5).unwrap();
        let mut n_significant = 20;
        let mut p_integrate = T::from_f32(0.05).unwrap();
        let mut verbose = false;
        let mut error_save = false;

        let args: Vec<String> = env::args().collect();
        let mut i = 1;
        let size = args.len();
        while i < size {
            let op = &args[i] as &str;
            if i + 1 == size { panic!("Invalid number of args!"); }
            let item = (&args[i+1]).to_owned();
            match op {
                "-f" => { filepath = item.to_string(); }
                "-h" => { header = match &item as &str {
                            "true" => true,
                            "false" => false,
                            _ => panic!("Invalid item of -h!"),
                        }
                    }
                "-k" => { k = item.parse().ok().unwrap(); }
                "-l" => { l = item.parse().ok().unwrap(); }
                "-zth" => { z_th = item.parse().ok().unwrap(); }
                "-mi" => { max_iter = item.parse().ok().unwrap(); }
                "-ipl" => { iter_per_lambda = item.parse().ok().unwrap(); }
                "-tl" => { tolerance = item.parse().ok().unwrap(); }
                "-ls" => { ls = item.parse().ok().unwrap(); }
                "-le" => { le = item.parse().ok().unwrap(); }
                "-rs" => { random_seed = item.parse().ok().unwrap(); }
                "-lo" => { lambda_once = item.parse().ok().unwrap(); }
                "-nu" => { n_null = item.parse().ok().unwrap(); }
                "-mp" => { maskpath = Some(item.to_owned()); }
                "-ssi" => { sd_significant = item.parse().ok().unwrap(); }
                "-nsi" => { n_significant = item.parse().ok().unwrap(); }
                "-pin" => { p_integrate = item.parse().ok().unwrap(); }
                "-v" => { 
                    verbose = match &item as &str {
                        "true" => true,
                        "false" => false,
                        _ => panic!("Invalid item of -v!"),
                    }
                }
                "-es" => { 
                    error_save = match &item as &str {
                        "true" => true,
                        "false" => false,
                        _ => panic!("Invalid item of -v!"),
                    }
                }
                a => { panic!("Unknown args {}!", a) }
            };
            i += 2;
        }

        let lambda_start = ten.powi(ls);
        let lambda_end = ten.powi(le);

        Self {
            filepath, header, k, l, z_th, max_iter, iter_per_lambda,
            tolerance, lambda_start, lambda_end, random_seed,
            lambda_once, n_null, maskpath,
            sd_significant, n_significant, p_integrate,
            verbose, error_save
        }
    }

    /// View parameter settings.
    pub fn view(&self) {
        println!("=== params start ===");
        println!("filepath: {}", self.filepath);
        println!("header: {}", self.header);
        println!("k: {}", self.k);
        println!("l: {}", self.l);
        println!("non-zero value threshold: {}", self.z_th);
        println!("max_iter: {}", self.max_iter);
        println!("iter_per_lambda: {}", self.iter_per_lambda);
        println!("tolerance: {}", self.tolerance);
        println!("lambda_start: {}", self.lambda_start);
        println!("lambda_end: {}", self.lambda_end);
        println!("random_seed: {}", self.random_seed);
        println!("lambda_once: {}", self.lambda_once);
        println!("number of null neurons: {}", self.n_null);
        match self.maskpath.as_ref() {
            Some(p) => println!("maskpath: {}", p),
            None => println!("maskpath: None"),
        }
        println!("significance level: {}SD", self.sd_significant);
        println!("minimum number of sig-cell: {}", self.n_significant);
        println!("integration level: {}", self.p_integrate);
        println!("verbose: {}", self.verbose);
        println!("error save: {}", self.error_save);
        println!("=== params  end  ===");
    }
}