# EM algorithm
In `./em_algorithm`, run
```
python main.py x.csz z.csv params.dat --figure
```
`x.csv` is input data csv file, `z.csv` is output posterior probabilities, and `params.dat` is parameters of each cluster.
`--figure` is optional, we can check 3D graph of the data colored by each class.

# Gibbs sampling
In `./gibbs`, run
```
python main.py x.csz z.csv params.dat --figure
```
`x.csv` is input data csv file, `z.csv` is output posterior probabilities, and `params.dat` is parameters of each cluster.
`--figure` is optional, we can check 3D graph of the data colored by each class.

If you encountered with numpy.linalg error, this is caused by unsuitably sampled initial value.
Please try sometimes to avoid this error.

# appropriate K
I tried 1~10 clusters with EM algorithm, and check the likelihood and AIC.
Here is the result.
| clusters | likelihood | AIC |
| -------- | ---------- | --- |
| 1        | -68789.6033 | 137561.2065 |
| 2        | -62992.9799 | 125947.9582 |
| 3        | -59621.9665 | 119185.9330 |
| 4        | -56631.7318 | 113185.4635 |
| 

We can also check the fact by drawing data on a 3D graph
![em](./em_algorithm/yRryXbjSky.png)
