# Quantum Anomaly Detection

A library exploring quantum computing approaches to anomaly detection, comparing them against classical methods across four data types: tabular, time series, image, and graph/network data.

All quantum circuits are built manually at gate level using [Qiskit](https://qiskit.org/), with full circuit visualization and educational explanations in Jupyter notebooks.

## Quantum Methods

### Quantum Kernel + One-Class SVM
Encodes classical data into quantum states via parameterized feature maps (ZZ, Pauli). The quantum kernel $K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$ is computed using statevector fidelity, then fed to a One-Class SVM for anomaly detection.

### Variational Quantum Circuit Autoencoder
A parameterized quantum circuit compresses $n$ qubits into $n_\text{latent}$ qubits. The "trash qubit" approach measures reconstruction quality: if compression is successful, trash qubits collapse to $|0\rangle$. The loss is $\mathcal{L} = 1 - P(\text{trash} = |0 \dots 0\rangle)$, trained with `scipy.optimize`.

### QAOA Clustering
Formulates balanced 2-cluster partitioning as a QUBO problem. The Quantum Approximate Optimization Algorithm finds cluster assignments. Small or isolated clusters are flagged as anomalies. The Hamiltonian includes a separation cost and a balance penalty:

$$H = -\sum_{i < j} \frac{D_{ij}}{2} Z_i Z_j + \lambda \Big(\sum_i Z_i\Big)^2$$

### QAOA Regression Residuals
Fits a classical regression model, then uses QAOA to solve a binary thresholding optimization. The Hamiltonian encourages labeling high-residual points as anomalies, with a smoothness penalty:

$$H = \sum_i |r_i| Z_i + \gamma \sum_{i < j} \frac{1 - Z_i Z_j}{4}$$

### Quantum Distance Estimation
Computes pairwise distances between quantum-encoded data points using state fidelity: $d(x_1, x_2) = \sqrt{1 - |\langle \phi(x_1) | \phi(x_2) \rangle|^2}$. A $k$-nearest-neighbor scoring scheme then identifies anomalies as points far from their neighbors.

## Data Preparation by Method and Problem Type

All quantum methods share a common preprocessing backbone — **StandardScaler → PCA → MinMaxScaler to [0, π]** — but differ in what happens before and after that pipeline depending on the data type and the method's encoding strategy.

### Quantum Kernel + One-Class SVM

| Data Type | Preparation |
|-----------|-------------|
| **Tabular** | Raw features (e.g. 30 credit-card columns) → StandardScaler → PCA to 6 components → MinMaxScaler to [0, π]. Each component becomes a rotation angle in a ZZ feature map with full entanglement. |
| **Time Series** | Each sliding window (64 points) is converted to 8 hand-crafted statistical features (mean, std, max, min, skewness, kurtosis, top-2 FFT magnitudes) → StandardScaler → PCA to 6 components → [0, π]. ZZ feature map encodes pairwise feature interactions. |
| **Image** | 28×28 MNIST pixels flattened to 784 values, divided by 255 → StandardScaler → PCA to 8 components → [0, π]. ZZ feature map captures correlations between principal components of digit shape. |
| **Graph/Network** | 41 KDD Cup 99 features with categorical columns (protocol, service, flag) label-encoded to numeric → StandardScaler → PCA to 8 components → [0, π]. ZZ feature map applied as with tabular data. |

### VQC Autoencoder

| Data Type | Preparation |
|-----------|-------------|
| **Tabular** | Same StandardScaler → PCA(6) → [0, π] pipeline. Each value is loaded via a single Ry(xᵢ) rotation per qubit (no entanglement in the data layer). The encoder/decoder layers then use parameterized Ry/Rz gates with linear CX entanglement. |
| **Time Series** | Window feature extraction (8 stats) → StandardScaler → PCA(8) → [0, π] → Ry encoding. The autoencoder compresses 8 qubits down to a latent space; trash-qubit probability serves as anomaly score. |
| **Image** | Pixel normalization → StandardScaler → PCA(8) → [0, π] → Ry encoding. Same trash-qubit compression approach on the PCA representation of digit images. |

### QAOA Clustering

| Data Type | Preparation |
|-----------|-------------|
| **Tabular** | StandardScaler → PCA(6) → [0, π]. A Euclidean distance matrix is computed between a subsample of data points (~12–16). Each point gets one qubit; the QUBO Hamiltonian uses pairwise distances as coupling strengths for a balanced 2-cluster partition. |
| **Graph/Network** | Label encoding → StandardScaler → PCA(8) → [0, π]. Same distance-matrix QUBO formulation on a subsample. Small or isolated clusters are flagged as anomalies. |

### QAOA Regression Residuals

| Data Type | Preparation |
|-----------|-------------|
| **Time Series** | Window features extracted and scaled to [0, π] as above. A classical regression model predicts the next window from the previous one. Absolute residuals |rᵢ| become the per-qubit cost coefficients in the QUBO Hamiltonian, where one qubit per point decides normal vs anomaly. |

### Quantum Distance k-NN

| Data Type | Preparation |
|-----------|-------------|
| **Tabular** | StandardScaler → PCA(6) → [0, π]. An angle-encoding feature map (H → Ry(xᵢ) → linear CX chain → Rz(xᵢ)) embeds each point into a quantum state. Pairwise fidelity d(x₁,x₂) = √(1 − |⟨φ(x₁)\|φ(x₂)⟩|²) is computed via statevector overlap, then k-NN scoring identifies outliers. |
| **Time Series** | Window feature extraction → StandardScaler → PCA(8) → [0, π] → angle encoding. Same fidelity-based distance and k-NN scoring. |
| **Image** | Pixel normalization → StandardScaler → PCA(8) → [0, π] → angle encoding. Quantum distances capture similarity between PCA-compressed digit representations. |
| **Graph/Network** | Label encoding → StandardScaler → PCA(8) → [0, π] → angle encoding. Same quantum distance k-NN pipeline on preprocessed network features. |

## Classical Benchmarks

Two levels of comparison:

- **Level A**: Classical anomaly detection methods — Isolation Forest, One-Class SVM, Local Outlier Factor, Elliptic Envelope, DBSCAN
- **Level B**: For QAOA problems — Simulated Annealing solving the same QUBO formulation vs QAOA vs classical ML

## Method-Problem Mapping

| Method | Tabular | Time Series | Image | Graph/Network |
|--------|---------|-------------|-------|---------------|
| Quantum Kernel + One-Class SVM | x | x | x | x |
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
        quantum_kernel.py      # Kernel matrix + One-Class SVM
        vqc_autoencoder.py     # Training loop + scoring
        qaoa_clustering.py     # QAOA clustering + anomaly labeling
        qaoa_regression.py     # Regression residuals + QAOA thresholding
        quantum_distance.py    # Quantum distance + k-NN scoring
    classical/         # Classical benchmarks
        benchmarks.py      # IF, One-Class SVM, LOF, DBSCAN, Elliptic Envelope, SA
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

## References

1. <a id="ref-havlicek"></a> V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," *Nature* 567, 209–212 (2019). [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)
2. <a id="ref-schuld-kernel"></a> M. Schuld, "Supervised quantum machine learning models are kernel methods," (2021). [arXiv:2101.11020](https://arxiv.org/abs/2101.11020)
3. <a id="ref-romero"></a> J. Romero, J. P. Olson, A. Aspuru-Guzik, "Quantum autoencoders for efficient compression of quantum data," *Quantum Sci. Technol.* 2, 045001 (2017). [arXiv:1612.02806](https://arxiv.org/abs/1612.02806)
4. <a id="ref-farhi"></a> E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization Algorithm," (2014). [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
5. <a id="ref-otterbach"></a> J. S. Otterbach et al., "Unsupervised Machine Learning on a Hybrid Quantum Computer," (2017). [arXiv:1712.05771](https://arxiv.org/abs/1712.05771)
6. <a id="ref-lloyd-qdist"></a> S. Lloyd, M. Mohseni, P. Rebentrost, "Quantum algorithms for supervised and unsupervised machine learning," (2013). [arXiv:1307.0411](https://arxiv.org/abs/1307.0411)
7. <a id="ref-schuld-encoding"></a> M. Schuld, F. Petruccione, "Machine Learning with Quantum Computers," Springer (2021). [arXiv:2101.11020](https://arxiv.org/abs/2101.11020)

| Method | Key References |
|--------|---------------|
| Quantum Kernel + One-Class SVM | Havlíček et al. [[1]](#ref-havlicek), Schuld [[2]](#ref-schuld-kernel) |
| VQC Autoencoder | Romero et al. [[3]](#ref-romero) |
| QAOA Clustering | Farhi et al. [[4]](#ref-farhi), Otterbach et al. [[5]](#ref-otterbach) |
| QAOA Regression Residuals | Farhi et al. [[4]](#ref-farhi) |
| Quantum Distance k-NN | Lloyd et al. [[6]](#ref-lloyd-qdist) |
| Data Encoding / Feature Maps | Havlíček et al. [[1]](#ref-havlicek), Schuld & Petruccione [[7]](#ref-schuld-encoding) |

## License

MIT
