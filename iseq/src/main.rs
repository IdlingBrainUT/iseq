use iseq::args::*;
use iseq::env::*;
use iseq::result::*;
use iseq::io::*;
use iseq::time::*;

fn main() {
    let args: Args<f32> = Args::new();
    let mut env = Env::from_args(&args, &now_string());

    println!("# args");
    args.view();
    mkdir(&env.dirname);

    if args.lambda_once.is_infinite() {
        let opt_r = env.narrow_lambda();
        match opt_r {
            Ok(r) => {
                println!("# narrow");
                result_match(r);
            }
            Err(_) => panic!("I/O error!"),
        };
        let _ = env.final_result();
    } else {
        let _ = env.run_once(args.lambda_once);
    }
}