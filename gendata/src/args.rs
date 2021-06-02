//! Get the parameter settings from args.

use std::env;

/// Preserving parameter settings.
pub struct Args{
    /// -f: The name of the csv file of generative data (shape=(n, t))
    /// 
    /// (default) "gendata.csv"
    pub filename: String,
    /// -k: The number of patterns (K).
    /// 
    /// (default) 3
    pub k: usize,
    /// -l: The length of the time window of patterns (L).
    /// 
    /// (default) 50
    pub l: usize,

    pub t: usize,

    pub nk: usize,
    pub n: usize,

    pub c1: f32,
    pub c2: f32,
    pub g: usize,
    pub fr: f32,

    pub lambda: f32,
    pub sigma: f32,

    pub random_seed: u32,
    
}

impl Args {
    /// Get the parameter settings from args.
    pub fn new() -> Self {
        let mut filename = "gendata.csv".to_string();
        let mut k = 3;
        let mut l = 50;
        let mut t = 3000;
        let mut nk = 100;
        let mut c1 = 0.5;
        let mut c2 = 2.0;
        let mut g = 10;
        let mut fr = 0.1;

        let mut lambda = 1.0;
        let mut sigma = 0.05;

        let mut random_seed = 0;

        let args: Vec<String> = env::args().collect();
        let mut i = 1;
        let size = args.len();
        while i < size {
            let op = &args[i] as &str;
            if i + 1 == size { panic!("Invalid number of args!"); }
            let item = (&args[i+1]).to_owned();
            match op {
                "-f" => { filename = item.to_string(); }
                "-k" => { k = item.parse().ok().unwrap(); }
                "-l" => { l = item.parse().ok().unwrap(); }
                "-t" => { t = item.parse().ok().unwrap(); }
                "-nk" => { nk = item.parse().ok().unwrap(); }
                "-c1" => { c1 = item.parse().ok().unwrap(); }
                "-c2" => { c2 = item.parse().ok().unwrap(); }
                "-g" => { g = item.parse().ok().unwrap(); }
                "-fr" => { fr = item.parse().ok().unwrap(); }
                "-la" => { lambda = item.parse().ok().unwrap(); }
                "-si" => { sigma = item.parse().ok().unwrap(); }
                "-rs" => { random_seed = item.parse().ok().unwrap(); }
                a => { panic!("Unknown args {}!", a) }
            };
            i += 2;
        }

        let n = nk * k;

        Self {
            filename, k, l, t, nk, n, c1, c2, g, fr,
            lambda, sigma, random_seed,
        }
    }

    /// View parameter settings.
    pub fn view(&self) {
        println!("filename: {}", self.filename);
        println!("=== Data size ===");
        println!("Length of Data: {}", self.t);
        println!("Number of Neurons: {}", self.n);
        println!("Number of Patterns: {}", self.k);
        println!("Number of Neurons in a Pattern: {}", self.nk);
        println!("Length of Patterns: {}", self.l);
        println!("=== Parameters ===");
        println!("C1: {}", self.c1);
        println!("C2: {}", self.c2);
        println!("G: {}", self.g);
        println!("Firing Rate: {}", self.fr);
        println!("Lambda: {}", self.lambda);
        println!("Sigma: {}", self.sigma);
        println!("random_seed: {}", self.random_seed);
    }
}