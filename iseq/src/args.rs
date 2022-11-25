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

    /// -mi: The maximum number of times the update calculation can be done.
    /// 
    /// (default) 100
    pub n_iter: [usize; 4],

    /// -zth: threshold of non-zero values.
    /// 
    /// (default) 0.001
    pub z_th: T,

    /// -tl: To what extent do you ignore the difference between the numbers below.
    /// 
    /// (default) 1e-7
    pub tolerance: T,

    /// -rs: Seed values for the random number generator.
    /// 
    /// [NOTE] One of the 32 bit number (0-4294967295) must be set.
    /// 
    /// If not, this program will use default seed 0, and will output the same result for the same input every time.
    /// 
    /// (default) None
    pub random_seed: Option<u32>,

    /// -cr: Significance level for integrating similar sequences.
    /// 
    /// (defalut) 0.3
    pub comp_rate: T,

    /// -cm
    /// 
    /// (default) 0.3
    pub corr_max: T,

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
        let mut n_iter = [30, 30, 10, 30];
        let mut tolerance = T::from(1e-7).unwrap();
        let mut random_seed = 0;
        let mut comp_rate = 0.3;
        let mut corr_max = 0.3;
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
                "-ni" => {
                    let n0: usize = item.parse().ok().unwrap();
                    let n1: usize = (&args[i+2]).to_owned().parse().ok().unwrap();
                    let n2: usize = (&args[i+3]).to_owned().parse().ok().unwrap();
                    let n3: usize = (&args[i+4]).to_owned().parse().ok().unwrap();
                    i += 3;
                }
                "-tl" => { tolerance = item.parse().ok().unwrap(); }
                "-rs" => {
                    let seed: u32 = item.parse().ok().unwrap();
                    random_seed = Some(seed);
                }
                "-cr" => { comp_rate = item.parse().ok().unwrap(); }
                "-cm" => { corr_max = item.parse().ok().unwrap(); }
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

        Self {
            filepath, header, k, l, z_th, n_iter, 
            tolerance, random_seed,
            comp_rate, corr_max,
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
        println!("n_iter: {:?}", self.n_iter);
        println!("tolerance: {}", self.tolerance);
        println!("random seed: {:?}", self.random_seed);
        println!("compression rate: {}", self.comp_rate);
        println!("maximum correlation: {}", self.corr_max);
        println!("verbose: {}", self.verbose);
        println!("error save: {}", self.error_save);
        println!("=== params  end  ===");
    }
}