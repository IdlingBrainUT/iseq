use ndarray::*;
use gendata::args::*;
use gendata::calcium::calcium;
use gendata::random::*;
use iseq::io::*;

fn main() {
    let args: Args = Args::new();
    args.view();

    let k = args.k;
    let l = args.l;
    let t = args.t;
    let nk = args.nk;
    let n = args.n;
    let c1 = args.c1;
    let c2 = args.c2;
    let g = args.g;
    let fr = args.fr;
    let lambda = args.lambda;
    let sigma = args.sigma;
    let random_seed = args.random_seed;

    let mut data: Array2<f32> = Array2::zeros((t, n));
    let mut dist = DistUtil::from_seed(random_seed);
    
    let t_long = t + l - 1;
    for ki in 0..k {
        let index_start = nk * ki;
        let mut y: Array1<f32> = Array1::zeros(t_long);
        for ti in 0..t_long {
            if dist.unin() < fr {
                let u = dist.expon(lambda);
                let mut ts = ti;
                while ts < t_long && ts - ti < g {
                    y[ts] += u * calcium(ts, ti, c1, c2);
                    ts += 1;
                }
            }
        }
        for nki in 0..nk {
            let shift = dist.unii((0, l));
            Zip::from(data.slice_mut(s![.., index_start+nki]))
                .and(&y.slice(s![shift..shift+t]))
                .apply(|a, &b| {
                    let tmp = b + dist.normal(0.0, sigma);
                    if tmp > 0.01 {
                        *a = tmp;
                    }
                });
        }
    }
    let _ = data.save_to_csv(&args.filename);
}
