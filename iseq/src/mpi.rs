use mpi::traits::*;

use iseq::args::*;
use iseq::env_mpi::*;
use iseq::result::*;
use iseq::io::*;
use iseq::time::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let root_rank = 0;

    let args: Args<f32> = Args::new();
    let mut env = EnvMPI::from_args(&args, &now_string());
    if rank == root_rank {
        println!("# args");
        args.view();
        mkdir(&env.dirname);
    }

    if args.lambda_once.is_infinite() {
        let opt_r = env.narrow_lambda(&world);
        if rank == root_rank {
            match opt_r {
                Ok(r) => {
                    println!("# narrow");
                    result_match(r);
                }
                Err(_) => panic!("I/O error!"),
            };
        }
        let _ = env.final_result(&world);
    } else {
        let _ = env.run_once(args.lambda_once, &world);
    }
}