# cs372-ProjectRhymer

This repository contains source codes used in ProjectRhymer, a CS372 team project.

## How to run
### Run REPL using `demo.py`
You can run our Read-Eval-Print Loop demo by running `python demo.py`.
For the configurable parameters, edit initial values of variables in the `demo.py`.

### Run benchmark using `run_benchmark.py`
You can test several combinations of parameters using the same inputs by running `python run_benchmark.py`.

## Code Structure
```
demo.py: Demo code
evaluation.py: Functions for calculating rhyme scores
misc.py: Utility functions
pipeline.py: Main pipeline function
pronunciation.py: Functions for the pronunciation step
restructurer.py: Functions for the restructure step
semantics.py: Functions for the semantics step
stats_result.py: Used to analyze benchmark results
```
