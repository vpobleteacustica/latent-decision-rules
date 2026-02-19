# latent-decision-rules

**Heuristic centroid-based classification with radial rejection in VAE latent space for ecoacoustic monitoring.**

This repository implements a deterministic geometric decision rule operating on learned latent embeddings produced by a Variational Autoencoder (VAE).

The method consists of:

1. Computing class centroids in latent space.
2. Estimating class-specific radial thresholds using a percentile hyperparameter \( q \).
3. Assigning a sample to its nearest centroid if it lies inside the corresponding confidence region.
4. Otherwise returning `"NO_DETECT"`.

This approach was developed as part of a thesis project on amphibian ecoacoustic monitoring and focuses exclusively on the decision mechanism, not on modifying the VAE architecture.

## Mathematical Formulation

For each class $c$:

Let  

$$
Z_c \in \mathbb{R}^{N_c \times D}
$$

be the matrix of latent embeddings belonging to class $c$, where:

- $N_c$ is the number of training samples in class $c$,
- $D$ is the dimensionality of the latent space,
- each row $z_i \in \mathbb{R}^D$ is a latent embedding vector.

### Class Centroid

The class centroid is defined as the empirical mean:

$$
\mu_c = \frac{1}{N_c} \sum_{i=1}^{N_c} z_i
$$

where $\mu_c \in \mathbb{R}^D$ represents the geometric prototype of class $c$ in latent space.

### Radial Distances

For each training embedding of class $c$, we compute its Euclidean distance to the centroid:

$$
d_i = \| z_i - \mu_c \|_2
$$

where $\|\cdot\|_2$ denotes the Euclidean norm.

These distances characterize the dispersion of class $c$ in latent space.

### Radial Threshold (Percentile-Based)

The class-specific decision radius is defined as:

$$
r_c = \text{quantile}(\{ d_i \}, q)
$$

where:

- $r_c$ is the radial threshold for class $c$,
- $q \in (0,1)$ is a percentile hyperparameter controlling the size of the acceptance region.

Interpretation of $q$:

- Larger $q$ → larger hypersphere → lower rejection rate  
- Smaller $q$ → tighter hypersphere → stricter detection  

### Empirical Interpretation (ECDF)

The percentile-based radius admits an equivalent statistical interpretation via the empirical cumulative distribution function (ECDF) of the distances.

Define:

$$
F_c(r) = \frac{1}{N_c} \sum_{i=1}^{N_c} \mathbf{1}(d_i \le r)
$$

where $F_c(r)$ represents the empirical probability that a training embedding of class $c$ lies within distance $r$ from its centroid.

The radial threshold can then be written as the inverse ECDF:

$$
r_c = F_c^{-1}(q)
$$

Thus, the acceptance region for class $c$ contains approximately a fraction $q$ of its training embeddings.

This interpretation clarifies that the method fixes an empirical coverage level in latent space rather than selecting an arbitrary geometric radius.

## Inference Rule

At inference time, given a latent embedding  

$$
z \in \mathbb{R}^D,
$$

we apply a three-stage geometric decision procedure in the learned latent space.

### 1) Nearest-Centroid Selection

We compute the Euclidean distance between $z$ and each class centroid:

$$
d_c(z) = \| z - \mu_c \|_2
$$

The candidate class $c^{*}$ is selected as:

$$
c^{*} = \arg\min_c \; d_c(z)
$$

This step identifies the geometrically closest class prototype.

### 2) Radial Acceptance Test

Each class defines a confidence region represented by a hypersphere centered at $\mu_c$ with radius $r_c$.

The embedding $z$ is accepted as belonging to class $c^{*}$ only if:  

$$d_{c^{*}}(z) \le r_{c^*}$$

If $d_{c^{*}}(z) > r_{c^{*}}$ the sample lies outside all class-specific confidence regions and is rejected.

### 3) Final Decision Function

The prediction function is defined as:

$$
\hat{y}(z) =
\begin{array}{ll}
c^{*} & \text{if } d_{c^{*}}(z) \le r_{c^{{*}}} \\
\text{NO\_DETECT} & \text{otherwise}
\end{array}
$$

where $\hat{y}(z)$ denotes the predicted label produced by the radial decision rule.

This formulation explicitly introduces a rejection mechanism, enabling open-set behavior while preserving geometric interpretability.

## Repository Structure

```text
latent-decision-rules/
│
├── data/
│   ├── embeddings/          # Precomputed latent embeddings (train/val/test)
│   └── README.md            # Description of expected data format
│
├── scripts/
│   ├── 01_compute_centroids.py
│   ├── 02_compute_radial_thresholds.py
│   ├── 03_classify_with_radial_rule.py
│   ├── 04_q_sweep_experiment.py
│   ├── 21_summarize_q_sweep.py
│   └── 22_plot_q_sweep.py
│
├── outputs/
│   ├── q_sweep_*/
│   │   ├── predictions/
│   │   ├── summary/
│   │   └── plots/
│
├── environment.yml
├── requirements.txt
└── README.md
```

## Data Format

data/embeddings/ must contain precomputed latent embeddings extracted from a trained VAE.

Embeddings should be organized into:
	•	train
	•	val
	•	test

Each file must include:
	•	latent vector (dimension ( D ))
	•	true class label

The VAE architecture is assumed fixed and external to this repository.

## Experimental Pipeline

The core scripts implement the full experimental workflow:
	•	01_compute_centroids.py
Computes class centroids from training embeddings.
	•	02_compute_radial_thresholds.py
Estimates class-specific thresholds ( r_c ) using quantile parameter ( q ).
	•	03_classify_with_radial_rule.py
Applies the geometric decision rule with rejection.
	•	04_q_sweep_experiment.py
Runs systematic evaluation across multiple ( q ) values.
	•	21_summarize_q_sweep.py
Aggregates metrics per split and per class.
	•	22_plot_q_sweep.py
Generates performance curves and distance distribution plots.

## Outputs

The outputs/ directory contains automatically generated experimental results:
	•	Metrics per split (train / val / test)
	•	Per-class accuracy
	•	Distance distributions
	•	Performance curves as a function of \( q \)

## Reproducibility

Reproducible environments are provided via:
	•	environment.yml (Conda)
	•	requirements.txt (pip)

## Limitations and Scope

This repository focuses exclusively on a deterministic geometric decision rule applied to fixed latent embeddings produced by a pretrained VAE.

The current implementation:
	•	Does not modify or retrain the VAE architecture.
	•	Assumes class distributions are reasonably clustered in latent space.
	•	Uses a percentile-based radial threshold rather than probabilistic calibration.
	•	Does not incorporate discriminative classifiers or Bayesian decision rules.

The method is therefore best interpreted as:
	•	A post-hoc geometric decision mechanism,
	•	Operating independently from representation learning,
	•	Providing a transparent rejection-aware classification baseline.

Future extensions may explore:
	•	Probabilistic calibration of acceptance regions,
	•	Alternative distance metrics,
	•	Adaptive or class-dependent quantile selection,
	•	Comparison with discriminative or MAP-based decision rules.
