# Scaling Laws Experiment Design

Our goal is to predict, for a FLOPs budget of `1e19`, the compute-optimal model size, a concrete hyperparameter configuration achieving that size, and the associated final training loss, while using no more than `2e18` FLOPs of exploratory API queries.

We use a two-stage procedure. First, we run a small pilot study to determine a stable architecture family and a learning-rate policy that works well across scales. Second, we spend the remaining budget on a structured scaling sweep over compute budgets and model sizes, fit scaling laws to the best runs, and extrapolate to `1e19`.

## 1. Pilot Study: Architecture Shape and Learning Rate

We first reserve a small portion of the exploration budget for pilot runs. The purpose of this stage is not to fit the scaling law, but to decide:

- a stable learning-rate policy under a fixed scheduler family,
- a preferred architecture family at fixed parameter count,
- a consistent rule for mapping target parameter count to concrete hyperparameters.

In this stage, we compare a small number of models at the same approximate parameter count but with different depth/width tradeoffs. For example, we compare relatively deeper and narrower models with wider and shallower models while keeping the feed-forward ratio fixed at `d_ff = 4 * d_model` and choosing `num_heads` so that `d_model` is divisible by `num_heads`. We then choose the architecture family that gives the best loss most consistently.

For optimization, we use a warmup-stable-decay (WSD) learning-rate schedule throughout the experiment. In the pilot stage, we run a small base-learning-rate sweep on representative small models and either choose one base LR that works consistently well or a simple monotonic rule if the best LR drifts with model size. We then keep the scheduler family, warmup ratio, stable region, decay shape, optimizer, and batch size fixed for the main sweep so that the scaling-law experiments remain comparable. We do not use muP; instead, we choose the architecture family and learning-rate policy empirically from pilot runs and then hold that policy fixed.

## 2. Main Scaling Sweep

After the pilot stage, we spend the remaining FLOPs budget on a set of runs across several compute budgets `C`. We choose 3 to 4 budgets spaced geometrically below the `2e18` cap, since geometric spacing gives better leverage for fitting a power law than many nearly redundant budgets.

At each budget `C_i`, we evaluate only a small number of model sizes `N` to stay within budget. Rather than scanning a large grid, we choose 2 to 3 parameter counts centered around the current estimate of the optimum. For the smallest budget, we use a coarse initial bracket. After observing the first few minima, we fit a provisional scaling law and use it to choose parameter counts for the higher-budget runs. This adaptive design uses the exploration budget much more efficiently than an exhaustive sweep.

For each compute budget, we record the final training loss for each tested model size and select the minimum-loss run. This produces empirical compute-optimal points `(C_i, N_opt(C_i))`. We also record the corresponding minimum loss values `(C_i, L_opt(C_i))`.

## 3. Scaling-Law Fitting

We fit a power law for optimal model size as a function of compute:

`N_opt(C) = a * C^alpha`

Equivalently, we fit a straight line in log-log space:

`log N_opt = log a + alpha * log C`

This gives a direct extrapolation to the target compute budget `C = 1e19`.

We also fit a separate law for the best achievable loss as a function of compute using the compute-optimal runs. We use a simple low-variance functional form and keep the fitting method fixed across all runs. The point of this second fit is to predict the final training loss at `1e19` once the model is trained at the predicted optimal size.

## 4. Converting Predicted Model Size to Hyperparameters

Once we obtain the predicted optimal parameter count `N_opt(1e19)`, we convert it into a concrete Transformer configuration using the architecture family selected in the pilot stage. We keep the same structural rules used throughout the sweep:

- fixed `d_ff / d_model` ratio,
- valid `num_heads` dividing `d_model`,
- fixed context length and other non-swept settings,
- batch size constrained to `128` or `256` as required by the assignment.

We then choose `d_model` and `num_layers` so that the resulting model size is as close as possible to the predicted optimal parameter count. If two nearby configurations are possible, we choose the one more consistent with the pilot-stage preference.

## 5. Learning Rate Treatment

Learning rate is treated as an optimization hyperparameter, not part of model size. We do not assume that exactly the same learning rate is optimal for every model size. Instead, we use the pilot stage to identify a stable learning-rate policy under the WSD schedule. If the best base LR varies only weakly with scale, we fix one value globally. If it varies systematically, we use a simple rule such as a smaller base LR for larger models and keep that rule fixed throughout the main sweep. This avoids spending the full exploration budget on repeated LR tuning while keeping the optimization setup principled and reproducible.

## 6. Concrete Experiment Roadmap

The following roadmap gives a concrete plan that stays below the `2e18` exploration cap while leaving a small safety margin.

```mermaid
flowchart TD
    A[Phase 0: pick fixed recipe<br/>optimizer, WSD family, batch size 128 or 256] --> B[Phase 1: pilot LR sweep<br/>small models, 2-3 base LRs]
    B --> C[Phase 2: pilot shape sweep<br/>same parameter count, deeper vs wider]
    C --> D[Choose architecture family<br/>and LR policy]
    D --> E[Phase 3: main sweep at C1<br/>coarse N bracket]
    E --> F[Fit provisional N_opt(C)]
    F --> G[Phase 4: main sweep at C2, C3, C4<br/>adaptive N choices near predicted optimum]
    G --> H[Fit final scaling laws<br/>N_opt(C) and L_opt(C)]
    H --> I[Extrapolate to 1e19 FLOPs]
    I --> J[Map predicted N to concrete hyperparameters]
```

### 6.1 Fixed Settings

The following settings are held fixed unless a run is clearly unstable:

- scheduler family: WSD
- feed-forward ratio: `d_ff = 4 * d_model`
- valid head count with `d_model % num_heads == 0`
- batch size: `128` or `256`
- optimizer type, warmup style, and decay style

### 6.2 FLOPs Budget Sheet

| Phase | Purpose | Per-run compute | Number of runs | Phase total |
| --- | --- | ---: | ---: | ---: |
| Pilot A | Base LR sweep at one small model size | `1e16` | 4 | `4e16` |
| Pilot B | Base LR sweep at a second small model size | `3e16` | 3 | `9e16` |
| Pilot C | Depth-vs-width comparison at fixed parameter count | `3e16` | 2 | `6e16` |
| Main C1 | First scaling point with coarse parameter bracket | `5e16` | 3 | `1.5e17` |
| Main C2 | Second scaling point | `1e17` | 3 | `3e17` |
| Main C3 | Third scaling point | `2e17` | 3 | `6e17` |
| Main C4 | Fourth scaling point near budget limit | `3e17` | 2 | `6e17` |
| Total | Full exploration plan |  | 20 | `1.84e18` |

This plan leaves roughly `1.6e17` FLOPs of margin under the hard `2e18` cap in case one pilot run needs to be repeated or one extra confirmation run is needed near the predicted optimum.

### 6.3 Pilot Run Sheet

The pilot stage is meant to determine a modeling recipe, not to fit the final scaling law.

| Run group | Budget | Model-size target | Variables changed | Goal |
| --- | ---: | --- | --- | --- |
| Pilot A | `1e16` | small | base LR across 3 values, plus 1 confirmation run | find a plausible LR range under WSD |
| Pilot B | `3e16` | small-to-medium | base LR across 3 values | check whether best LR shifts with size |
| Pilot C | `3e16` | fixed parameter count | depth/width tradeoff | pick an architecture family |

Example choices for the base LR sweep are `{1e-4, 3e-4, 6e-4}`. If the best LR moves with model size, we replace a single LR with a simple rule such as choosing a slightly smaller base LR for the larger model family.

### 6.4 Main Sweep Sheet

The main sweep fits the scaling law. At each compute budget, we evaluate only a few parameter counts centered around the currently estimated optimum.

| Stage | Compute budget | Candidate model sizes | Selection rule |
| --- | ---: | --- | --- |
| Main C1 | `5e16` | coarse bracket, for example low / mid / high | use a wide bracket to locate the optimum region |
| Main C2 | `1e17` | around provisional `N_opt(C)` | choose 3 sizes around the predicted optimum |
| Main C3 | `2e17` | around updated `N_opt(C)` | refine the bracket using the first two minima |
| Main C4 | `3e17` | 2 final candidates near predicted optimum | use the best current fit to place the last runs |

A practical parameter-bracketing rule is:

- for the first scaling point, choose 3 parameter counts spaced roughly geometrically,
- after that, fit a provisional power law and evaluate approximately `{0.7x, 1.0x, 1.4x}` around the current predicted optimum,
- if the optimum falls on the edge of the bracket, shift the next bracket in that direction.

## 7. Fitting Outputs

At the end of the sweep, we fit:

- `N_opt(C) = a * C^alpha`
- `L_opt(C)` using the minimum-loss run at each compute budget

We then extrapolate both quantities to `C = 1e19`. Finally, we convert the predicted `N_opt(1e19)` into a concrete Transformer configuration by choosing the nearest feasible `(d_model, num_layers, d_ff, num_heads)` combination under the fixed architecture family.

## 8. Why This Design

This design is intended to balance three competing goals:

- stay within the strict `2e18` FLOPs exploration budget,
- obtain enough variation in compute to fit a meaningful scaling law,
- avoid wasting runs on repeated retuning of architecture and optimization hyperparameters.

The key idea is to separate the problem into two parts: first determine a stable modeling recipe on small runs, then use that recipe to study how optimal parameter count scales with compute. This gives a defensible, reproducible methodology for predicting the best model size and loss at `1e19`.
