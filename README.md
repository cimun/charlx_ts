# charlx_ts

## Installation

To install all dependencies of CHARLX, create a new `conda` environment with `python` 3.12. Clone both the `foobench` and `CondEvo` repositories into the current folder and install them via `pip`. Additionally, install `ipykernel` for using the tutorial notebook, `seaborn` for plotting, `ase` for handling atomic structures, `mace-torch` for the evaluation with MACE foundation models, and `sella` for transition state optimization. The following lines of code execute all necessary commands:

```bash
conda create -n charlx python=3.12 ipython
conda activate charlx
git clone https://github.com/bhartl/foobench.git
pip install -e foobench/
git clone https://github.com/bhartl/CondEvo.git
pip install -e CondEvo/
pip install ipykernel seaborn ase mace-torch sella

# If using a CUDA-enabled GPU:
pip install --upgrade "jax[cuda12_local]==0.4.36" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
