# Quantum Anomaly Detection

A library exploring quantum computing approaches to anomaly detection, comparing them against classical methods across four data types: tabular, time series, image, and graph/network data.

All quantum circuits are built manually at gate level using [Qiskit](https://qiskit.org/), with full circuit visualization and educational explanations in Jupyter notebooks.

## Quantum Methods

### Quantum Kernel + One-Class SVM
Encodes classical data into quantum states via parameterized feature maps (ZZ, Pauli). The quantum kernel matrix is computed using statevector fidelity, then fed to a One-Class SVM for anomaly detection.

$$K(x_i, x_j) = \left| \langle \phi(x_i) | \phi(x_j) \rangle \right|^2$$

### Variational Quantum Circuit Autoencoder
A parameterized quantum circuit compresses $n$ qubits into $n_{\text{latent}}$ qubits. The "trash qubit" approach measures reconstruction quality: if compression is successful, trash qubits collapse to $|0\rangle$. Trained with `scipy.optimize`.

$$\mathcal{L} = 1 - P\!\left(\text{trash} = |0\ldots0\rangle\right)$$

### QAOA Clustering
Formulates balanced 2-cluster partitioning as a QUBO problem. The Quantum Approximate Optimization Algorithm finds cluster assignments. Small or isolated clusters are flagged as anomalies.

$$H = -\sum_{i<j} \frac{D_{ij}}{2}\, Z_i Z_j \;+\; \lambda \left(\sum_i Z_i\right)^{\!2}$$

### QAOA Regression Residuals
Fits a classical regression model, then uses QAOA to solve a binary thresholding optimization. The Hamiltonian encourages labeling high-residual points as anomalies:

$$H = \sum_i |r_i|\, Z_i \;+\; \gamma \sum_{i<j} \frac{1 - Z_i Z_j}{4}$$

### Quantum Distance Estimation
Computes pairwise distances between quantum-encoded data points using state fidelity. A $k$-nearest-neighbor scoring scheme then identifies anomalies as points far from their neighbors.

$$d(x_1, x_2) = \sqrt{1 - \left| \langle \phi(x_1) | \phi(x_2) \rangle \right|^2}$$

## Classical Benchmarks

Two levels of comparison:

- **Level A**: Classical anomaly detection methods — Isolation Forest, One-Class SVM, Local Outlier Factor, Elliptic Envelope, DBSCAN
- **Level B**: For QAOA problems — Simulated Annealing solving the same QUBO formulation vs QAOA vs classical ML

## Method-Problem Mapping

| Method | Tabular | Time Series | Image | Graph/Network |
|--------|---------|-------------|-------|---------------|
| Quantum Kernel + OCSVM | x | x | x | x |
| VQC Autoencoder | x | x | x | |
| QAOA Clustering | x | | | x |
| QAOA Regression | | x | | |
| Quantum Distance k-NN | x | x | x | x |

## Project Structure

```
src/quantum_anomaly_detection/
    circuits/          # Gate-level quantum circuit construction
        feature_maps.py    # ZZ, Pauli, angle-encoding feature maps
        autoencoder.py     # VQC autoencoder (encoder + decoder + trash qubit loss)
        qaoa.py            # QUBO Hamiltonians + QAOA circuit builder
        swap_test.py       # Swap test + fidelity-based distance
        utils.py           # Statevector helpers
    methods/           # High-level quantum anomaly detection pipelines
        quantum_kernel.py      # Kernel matrix + OCSVM
        vqc_autoencoder.py     # Training loop + scoring
        qaoa_clustering.py     # QAOA clustering + anomaly labeling
        qaoa_regression.py     # Regression residuals + QAOA thresholding
        quantum_distance.py    # Quantum distance + k-NN scoring
    classical/         # Classical benchmarks
        benchmarks.py      # IF, OCSVM, LOF, DBSCAN, Elliptic Envelope, SA
    data/              # Dataset loading & preprocessing
        tabular.py         # Credit card fraud, synthetic blobs
        time_series.py     # Synthetic sensor/ECG data
        image.py           # MNIST anomaly setup
        graph.py           # KDD Cup 99 network intrusion
    evaluation/        # Metrics & comparison
        metrics.py
    visualization/     # Plotting functions
        plots.py

notebooks/
    01_tabular.ipynb           # Credit card fraud / synthetic tabular data
    02_time_series.ipynb       # Synthetic sensor anomalies
    03_image.ipynb             # MNIST digit anomaly detection
    04_graph_network.ipynb     # KDD Cup 99 network intrusion

tests/                         # Unit tests for circuits, data, methods
```

## Installation

Requires Python >= 3.10. Uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Clone the repository
git clone https://github.com/hosseinsadeghi/quantum-anomaly-detection.git
cd quantum-anomaly-detection

# Create a virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

### Run the notebooks

```bash
jupyter notebook notebooks/
```

Each notebook is self-contained with:
- A configuration cell at the top (random seed, qubit count, sample sizes) that you can modify
- Background theory for each quantum method
- Circuit construction and visualization
- Training/optimization with progress plots
- Comparison tables and ROC curves

### Use as a library

```python
from quantum_anomaly_detection.circuits.feature_maps import build_zz_feature_map
from quantum_anomaly_detection.methods.quantum_kernel import quantum_kernel_ocsvm

feature_map = build_zz_feature_map(n_qubits=6, reps=2)
predictions, scores, model = quantum_kernel_ocsvm(X_train, X_test, feature_map, nu=0.1)
```

### Run tests

```bash
pytest tests/ -v
```

## Datasets

All datasets auto-download on first use:

- **Tabular**: Synthetic Gaussian blobs with outliers (default), or Credit Card Fraud from OpenML
- **Time Series**: Synthetic sinusoidal signal with spike, frequency-shift, and level-shift anomalies
- **Image**: MNIST digits — one digit as normal, another as anomaly
- **Graph/Network**: KDD Cup 99 network intrusion dataset via scikit-learn

## Technical Notes

- **Qubit budget**: Feature maps use 6-8 qubits (one per PCA component). QAOA uses 12-16 qubits (one per data point in subsample). All circuits run on statevector simulation.
- **Data encoding**: Raw data $\to$ StandardScaler $\to$ PCA $\to$ MinMaxScaler to $[0, \pi]$ $\to$ rotation angles in quantum feature maps.
- **Reproducibility**: Each notebook has a `SEED` variable at the top. Change it to observe different results.

## Dependencies

- [Qiskit](https://qiskit.org/) >= 2.3 — quantum circuit construction and simulation
- [scikit-learn](https://scikit-learn.org/) — classical ML methods, preprocessing, metrics
- NumPy, SciPy, pandas, matplotlib

## License

MIT
