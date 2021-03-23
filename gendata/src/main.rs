use ndarray::*;
use gendata::calcium::calcium;
use gendata::consts::*;
use gendata::random::*;
use iseqnmf::io::*;

fn main() {
    let mut data: Array2<f32> = Array2::zeros((T, N));
    let mut dist = DistUtil::from_seed(SEED);
    let t = T + L - 1;
    for ki in 0..K {
        let index_start = NK * ki;
        let mut y: Array1<f32> = Array1::zeros(t);
        for ti in 0..t {
            if dist.unin() < FR {
                let u = dist.expon(LAMBDA);
                let mut ts = ti;
                while ts < t && ts - ti < G {
                    y[ts] += u * calcium(ts, ti, C1, C2);
                    ts += 1;
                }
            }
        }
        for nki in 0..NK {
            let shift = dist.unii((0, L));
            Zip::from(data.slice_mut(s![.., index_start+nki]))
                .and(&y.slice(s![shift..shift+T]))
                .apply(|a, &b| {
                    let tmp = b + dist.normal(0.0, SIGMA);
                    if tmp > 0.01 {
                        *a = tmp;
                    }
                });
        }
    }
    let _ = data.save_to_csv("gendata.csv");
}
