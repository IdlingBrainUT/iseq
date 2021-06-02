# iSeq

The stable algorithm to detect comparable sequences.

## Binary File

|version|iSeq for MPI (computer cluster)|iSeq (general computer)|
|---|---|---|
|0.0.1|[iseq_mpi](https://mathcode.ushitora.net/iseq_mpi)|[iseq](https://mathcode.ushitora.net/iseq)|

## Parameters

- -f: Path of the csv file of X. (default) "readtest.csv"
- -h: Whether the csv file has a header data or not. (default) false
- -k: Maximum number of patterns (K). (default) 20
- -l: The length of time window of patterns (L). (defalut) 50
- -zth: threshold of non-zero values. (defalut) 0.001
- -mi: The maximum number of times the update calculation can be done. (default) 30
- -ipl: How many calculations are performed for the same value of lambda. (defalut) 5
- -tl: To what extent do you ignore the difference between the numbers below. (default) -inf
- -ls: The minimum value for narrowing lambda. (default) -6
- -le: The maximum value for narrowing lambda. (default) 0
- -rs: Seed values for the random number generator. (default) 0
- -lo: If this parameter is not infinity, do not search for lambda, and only run one seqNMF with this lambda value. (default) infinity
- -nu: The number of null-neurons. (default) 100
- -mp: Path of the csv file of M. (defalut) None
- -ssi: Multiplier for standard deviation of null-neurons' activity (default) 5.0
- -nsi: The minimum number of significant cells contained in significant sequences. (defalut) 20
- -pin: Significance level for integrating similar sequences. (defalut) 0.05
- -v: Output the calculation process. (default) false
- -es: Save reconstruction error matrix & x_hat as csv file. (default) false

## How to use

### iSeq for MPI (computer cluster)

```bash
mpirun -np 40 ./iseq_mpi -f filename.csv -ls -10 -k 20 -l 50 -mi 100 -rs 0 -zth 0.001 -es false -ipl 5
```

### iSeq (general computer)

```bash
./iseq -f filename.csv -ls -10 -k 20 -l 50 -mi 100 -rs 0 -zth 0.001 -es false -ipl 5
```