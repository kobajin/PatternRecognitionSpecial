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
```
// when encountered with unsuitable initial value, you can see the below sentence.
Encountered with unsuitable initial values. Trying next values.
````

# appropriate K
I tried 1~8 clusters with EM algorithm, and check the likelihood and AIC.
Here is the result.
| clusters | likelihood | AIC |
| -------- | ---------- | --- |
| 1        | -68789.6033 | 137597.2065 |
| 2        | -63032.5094 | 126103.0181 |
| 3        | -60010.1741 | 120078.2530 |
| 4        | -56631.7363 | 113341.4635 |
| 5        | -56625.4261 | 113348.6788 |
| 6        | -56620.5372 | 113358.8811 |
| 7        | -56614.7430 | 113367.3005 | 
| 8        | -56619.1064 | 113396.0193 |

The likelihood of model tends to become larger when the number of parameter is large.
Therefore, we cannot understand whether the model is good or not only by likelihood.
Since AIC considers the number of parameters in the model as a penalty, using AIC can avoid such problems.

We can see such tendencies in the above table, and by considering AIC, the optimal number of K is 4.
We can also check the fact by drawing data on a 3D graph
![em](./em_algorithm/yRryXbjSky.png)
