# iSeq

The stable algorithm to detect comparable sequences.

## Binary File

|version|iSeq for MPI (computer cluster)|iSeq (general computer)|
|:---:|:---:|:---:|
|0.0.1|[iseq_mpi](https://mathcode.ushitora.net/iseq_mpi)|[iseq](https://mathcode.ushitora.net/iseq)|

## Parameters

|Option|meaning|(default)|
|:---:|:---|:---:|
|-f|Path of the csv file of X.|"readtest.csv"|
|-h|Whether the csv file has a header data or not.|false|
|-k|Maximum number of patterns (K).|20|
|-l|The length of time window of patterns (L).|50|
|-zth|Threshold of non-zero values|0.001|
|-mi|The maximum number of times the update calculation can be done|30|
|-ipl|How many calculations are performed for the same value of lambda.|5|
|-tl|To what extent do you ignore the difference between the numbers below.|-inf|
|-ls|The minimum value for narrowing lambda.|-6|
|-le|The maximum value for narrowing lambda.|0|
|-rs|Seed values for the random number generator.|0|
|-lo|If this parameter is not infinity, do not search for lambda, and only run one seqNMF with this lambda value.|inf|
|-nu|The number of null-neurons.|100|
|-mp|Path of the csv file of M.|None|
|-ssi|Multiplier for standard deviation of null-neurons' activity.|5.0|
|-nsi|The minimum number of significant cells contained in significant sequences.|20|
|-pin|Significance level for integrating similar sequences.|0.05|
|-v|Output the calculation process.|false|
|-es|Save reconstruction error matrix & x_hat as csv file.|false|

## How to use

### iSeq for MPI (computer cluster)

```bash
mpirun -np 40 ./iseq_mpi -f filename.csv -ls -10 -k 20 -l 50 -mi 100 -rs 0 -zth 0.001 -es false -ipl 5
```

### iSeq (general computer)

```bash
./iseq -f filename.csv -ls -10 -k 20 -l 50 -mi 100 -rs 0 -zth 0.001 -es false -ipl 5
```