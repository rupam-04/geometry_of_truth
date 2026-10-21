Selected experiments from Samuel Marks and Max Tegmark’s [The Geometry of Truth](https://arxiv.org/abs/2310.06824), implemented using `meta-llama/Llama-2-13b-hf`.

## Implemented experiments

[geometry_of_truth.ipynb](geometry_of_truth.ipynb) includes:

- Last-token activation extraction, PCA visualization, and a 40-layer probe-accuracy sweep.
- Difference-of-means (MM) and standardized logistic-regression (LR) probes.
- Cross-dataset generalization across `cities`, `sp_en_trans`, and `larger_than`.
- A Spanish/English few-shot truth-classification baseline.
- Activation interventions using truth directions, including MM/LR effect comparisons.

The `neg_cities` dataset supplements probe training for the intervention experiments. Activations are extracted at layer 14; interventions span layers 8–14.

## Setup

Use Python 3.11, gated Hugging Face access to Llama-2, and a GPU environment compatible with bitsandbytes 8-bit loading. Run from the project directory:

```bash
conda create -n geometry_of_truth python=3.11.15 -y
conda activate geometry_of_truth
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements.txt
python -m ipykernel install --user --name geometry-of-truth --display-name 'Geometry of Truth'
export HF_TOKEN='<your Hugging Face access token>'
```

Open the notebook, select the **Geometry of Truth** kernel, and run the cells in order. To execute it from the command line and save a separate output notebook:

```bash
mkdir -p results
python -m nbconvert --to notebook --execute geometry_of_truth.ipynb \
  --ExecutePreprocessor.kernel_name=geometry-of-truth \
  --ExecutePreprocessor.timeout=-1 \
  --output geometry_of_truth.executed.ipynb --output-dir results
```