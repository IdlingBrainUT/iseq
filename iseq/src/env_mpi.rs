//! Calculation environment for MPI.

use mpi::point_to_point as p2p;
use mpi::traits::*;
use mpi::topology::SystemCommunicator;
use ndarray::*;
use std::cmp::{max, min};
use iseq_linalg::lambda::*;
use iseq_linalg::random::*;
use iseq_linalg::stats::*;

use crate::args::*;
use crate::def_env;
use crate::io::*;
use crate::result::*;
use crate::iseq::*;
use crate::time::*;
use crate::traits::*;

def_env!(
    /// Test
    struct EnvMPI,
    FloatISeqMPI
);

impl<T: FloatISeqMPI> EnvMPI<T> {
    /// Add rank * padding to random_seed
    pub fn seed_shift(&mut self, rank: i32, padding: usize) {
        let rank = rank as u32;
        let padding = padding as u32;
        let add = rank * padding;
        self.random_seed = if u32::MAX - add < self.random_seed {
            add - (u32::MAX - self.random_seed)
        } else {
            self.random_seed + add
        };
    }

    /// Find appropriate range of lambda based on the number of significant sequences.
    #[allow(bare_trait_objects)]
    pub fn narrow_lambda(&mut self, world: &SystemCommunicator) -> Result<ResultISeq, Box<std::error::Error>> {
        self.narrow_core(NarrowMode::Narrow, world)
    }

    /// Find appropriate value of lambda and output its result based on reconstruction accuracy.
    #[allow(bare_trait_objects)]
    pub fn final_result(&mut self, world: &SystemCommunicator) -> Result<ResultISeq, Box<std::error::Error>> {
        self.narrow_core(NarrowMode::Final(world.size() as usize), world)
    }

    /// Run iSeq only 1 lambda value (iter_per_lambda parallel)
    #[allow(bare_trait_objects)]
    pub fn run_once(&mut self, lambda_once: T, world: &SystemCommunicator) -> Result<ResultISeq, Box<std::error::Error>> {
        self.narrow_core(NarrowMode::Once(lambda_once), world)
    }

    /// Core function of calculation methods.
    #[allow(bare_trait_objects)]
    pub fn narrow_core(&mut self, mode: NarrowMode<T>, world: &SystemCommunicator) -> Result<ResultISeq, Box<std::error::Error>> {
        let lambda_vec = match lambda_vec(self.lambda_range, mode) {
            Ok(v) => v,
            Err(e) => panic!(e),
        };

        let rank = world.rank();
        let procs = world.size();

        let n_lambda = lambda_vec.len();
        let lambda_vec_all = match mode {
            NarrowMode::Final(_) => lambda_vec.clone(),
            _ => lambda_vec_all(&lambda_vec, self.iter_per_lambda),
        };
        let task_size = lambda_vec_all.len();

        let my_size = task_size / procs as usize + if task_size as i32 % procs > rank { 1 } else { 0 };
        let local_size = (task_size + procs as usize - 1) / procs as usize * 5;
        let mut local = vec![T::zero(); local_size];
        let my_id = (0..my_size).map(|i| procs as usize * i + rank as usize).collect::<Vec<usize>>();
        let lambda_vec_run = lambda_vec_run(&lambda_vec_all, &my_id);
        
        self.seed_shift(rank, task_size);
        let (my_recon, my_xorth, my_seed, my_k, my_valid) = self.calc_costs(&lambda_vec_run);
        for i in 0..my_size {
            local[5 * i] = my_recon[i];
            local[5 * i + 1] = my_xorth[i];
            local[5 * i + 2] = T::from(my_seed[i]).unwrap();
            local[5 * i + 3] = T::from(my_k[i]).unwrap();
            local[5 * i + 4] = T::from(my_valid[i]).unwrap();
        }

        let root_rank = 0;
        let root_process = world.process_at_rank(root_rank);
        let mut recvbuf = vec![T::zero(); local_size];
        let mut res = 0usize;
        let mut res_k_i = 0usize;
        let mut res_lo = T::zero();
        let mut res_up = T::zero();
        if rank == root_rank {
            let mut recon_vec = vec![T::zero(); task_size];
            let mut xortho_vec = vec![T::zero(); task_size];
            let mut seed_vec = vec![0u32; task_size];
            let mut k_vec = vec![0usize; task_size];
            let mut valid_vec = vec![0usize; task_size];
            for (i, &e) in my_id.iter().enumerate() {
                recon_vec[e] = local[5 * i];
                xortho_vec[e] = local[5 * i + 1];
                seed_vec[e] = local[5 * i + 2].to_u32().unwrap();
                k_vec[e] = local[5 * i + 3].to_usize().unwrap();
                valid_vec[e] = local[5 * i + 4].to_usize().unwrap();
            }
            
            for i in 1..procs as usize {
                let send_process = world.process_at_rank(i as i32);
                p2p::send_receive_into(&local[..], &send_process, &mut recvbuf[..], &send_process);
                let mut id = i;
                let mut pos = 0;
                while id < task_size {
                    recon_vec[id] = recvbuf[5 * pos];
                    xortho_vec[id] = recvbuf[5 * pos + 1];
                    seed_vec[id] = recvbuf[5 * pos + 2].to_u32().unwrap();
                    k_vec[id] = recvbuf[5 * pos + 3].to_usize().unwrap();
                    valid_vec[id] = recvbuf[5 * pos + 4].to_usize().unwrap();
                    id += procs as usize;
                    pos += 1;
                }
            }

            let mut save_arr = Array2::zeros((n_lambda, 8));
            match mode {
                NarrowMode::Final(_) => {
                    
                    for i in 0..n_lambda {
                        save_arr[[i, 0]] = lambda_vec[i];
                        save_arr[[i, 1]] = lambda_vec[i];
                        save_arr[[i, 2]] = T::from(i).unwrap();
                        save_arr[[i, 3]] = recon_vec[i];
                        save_arr[[i, 4]] = xortho_vec[i];
                        save_arr[[i, 5]] = T::from(seed_vec[i]).unwrap();
                        save_arr[[i, 6]] = T::from(k_vec[i]).unwrap();
                        save_arr[[i, 7]] = T::from(valid_vec[i]).unwrap();
                    }
                    
                    // rearrange_costs(&mut save_arr, &lambda_vec, &lambda_vec_all, &recon_vec, &xortho_vec, &seed_vec, &k_vec, &valid_vec);
                }
                _ => {
                    rearrange_costs(&mut save_arr, &lambda_vec, &lambda_vec_all, &recon_vec, &xortho_vec, &seed_vec, &k_vec, &valid_vec);
                }
            }
            let r = match mode {
                NarrowMode::Once(_) => Ok(()),
                _ => self.save_costs(&save_arr),
            };

            let v = match r {
                Ok(()) => vec![0usize; procs as usize],
                Err(_) => vec![1usize; procs as usize],
            };
            root_process.scatter_into_root(&v[..], &mut res);

            match mode {
                NarrowMode::Narrow => {
                    let (k, (r0, r1)) = Self::search_lambda_range(&save_arr);
                    let k_scatter = vec![k; procs as usize];
                    let lo_scatter = vec![r0; procs as usize];
                    let up_scatter = vec![r1; procs as usize];
                    root_process.scatter_into_root(&k_scatter[..], &mut res_k_i);
                    root_process.scatter_into_root(&lo_scatter[..], &mut res_lo);
                    root_process.scatter_into_root(&up_scatter[..], &mut res_up);
                }
                NarrowMode::Final(_) => {
                    let i = self.find_final_result(&save_arr);
                    let i_scatter = vec![i; procs as usize];
                    root_process.scatter_into_root(&i_scatter[..], &mut res_k_i);
                }
                NarrowMode::Once(_) => {
                    let i = save_arr[[0, 2]].to_usize().unwrap();
                    let i_scatter = vec![i; procs as usize];
                    root_process.scatter_into_root(&i_scatter[..], &mut res_k_i);

                    println!("# final result");
                    println!("lambda = {}", save_arr[[0, 0]]);
                    println!("reconstruction error = {}", save_arr[[0, 3]]);
                    println!("xortho error = {}", save_arr[[0, 4]]);
                    println!("{} patterns are detected!", save_arr[[0, 6]].to_usize().unwrap());
                }
            }
        } else {
            p2p::send_receive_into(&local[..], &root_process, &mut recvbuf[..], &root_process);
            root_process.scatter_into(&mut res);
            match mode {
                NarrowMode::Narrow => {
                    root_process.scatter_into(&mut res_k_i);
                    root_process.scatter_into(&mut res_lo);
                    root_process.scatter_into(&mut res_up);
                }
                _ => {
                    root_process.scatter_into(&mut res_k_i);
                }
            }
        }
        
        if res > 0 {
            panic!("file save error!")
        }

        match mode {
            NarrowMode::Narrow => {
                self.lambda_range = (res_lo, res_up);
                self.k = res_k_i;

                Ok(ResultISeq::Success)
            }
            _ => {
                let opt_process = world.process_at_rank(res_k_i as i32);

                if rank as usize == res_k_i {
                    let r = self.iseq.save_wh(&self.filename());

                    let v = match r {
                        Ok(()) => vec![0usize; procs as usize],
                        Err(_) => vec![1usize; procs as usize],
                    };
                    opt_process.scatter_into_root(&v[..], &mut res);
                } else {
                    opt_process.scatter_into(&mut res);
                }

                if res > 0 {
                    panic!("file save error!");
                }

                Ok(ResultISeq::Success)
            }
        }
    }
}