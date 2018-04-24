# StatClubProject

Some ideas for modelling and analysing strong lensed QSO in GAIA DR2 data set.

## Code
The folder lens contains python code for SIE and SIS lens modelling, plotting and basic inference. 
The folder gaiapix contains python code to make healpix map easier.

## Notebooks
The folder [notebooks](notebooks) contains a list of notebooks. 

### SIE lens model
The SIS and SIE series shows how to simulate Gaia like lensed QSO including proper motion based on SIE lens model and the inference ones explore some inference Bayesian models trying to quantify the significance of the relative lens-source proper motion for some simulated LQSO.

### Simulate LQSO Gaia query
The simulation series expose the strategy used to simulate data/simDataSet1.csv.gzip, a Gaia DR2 like query search around some known QSOs.
* [Simulate LQSO Gaia query Part 1](notebooks/Simulation%20of%20DR2%20query%20part%201.ipynb)
* [Simulate LQSO Gaia query Part 2](notebooks/Simulation%20of%20DR2%20query%20part%202.ipynb)
* [Simulate LQSO Gaia query Part 3](notebooks/Simulation%20of%20DR2%20query%20part%203.ipynb)

## Data
The folder data contains some data.
