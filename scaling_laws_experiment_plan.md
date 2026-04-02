# Scaling Laws Experiment Plan

This document gives a corrected and operational experiment plan for `Problem (scaling_laws)`. The goal is to use at most `2e18` FLOPs of exploratory runs to predict, at `1e19` FLOPs:

- the compute-optimal model size `N_opt(1e19)`,
- the corresponding optimal dataset size `D_opt(1e19)`,
- a concrete hyperparameter configuration that realizes the predicted model size,
- the associated final training loss.

The plan uses:

- `muP` only to make learning-rate transfer more reliable,
- a fixed WSD schedule family for all training runs,
- a small architecture pilot to choose the model family,
- a Chinchilla-style scaling-law fit for `L(N, D)`,
- the compute relation `C ≈ 6ND` to recover the best `(N, D)` pair under a target FLOPs budget.

## 1. Core Principle

The experiment is split into four stages:

1. lock a fixed training recipe,
2. use `muP` to narrow the learning-rate search on one small model and validate once at a larger model,
3. run a small architecture-shape pilot at fixed parameter count,
4. collect a small but structured set of `(N, D, L)` observations under the fixed recipe and fit a scaling law.

The important correction is that `muP` is used only to support learning-rate transfer. It is not assumed to determine the globally best depth-width allocation by itself. Architecture choice is still validated empirically.

## 2. Fixed Recipe Sheet

These settings should remain fixed for the whole main experiment unless a run is clearly unstable.

| Setting | Planned value | Notes |
| --- | --- | --- |
| Model family | decoder-only Transformer | use the provided training stack |
| Batch size | `128` | satisfies assignment constraint |
| Scheduler family | WSD | keep one scheduler family for all runs |
| WSD warmup fraction | `5%` | keep fixed if exposed by API |
| WSD stable fraction | `85%` | keep fixed if exposed by API |
| WSD decay fraction | `10%` | keep fixed if exposed by API |
| Optimizer | fixed | use API default or one chosen optimizer |
| Context length | fixed | use one value throughout |
| FFN ratio | `d_ff = 4 * d_model` | fixed architecture rule |
| Head rule | `d_model % num_heads == 0` | choose valid divisors only |
| LR policy | determined by Stage 1 | one base LR or a simple size-based rule |

Batch size is not treated as a target of optimization. It should be fixed at `128` or, if needed, `256`. This keeps the experiment focused on model size, data size, and loss.

## 3. Roadmap Graph

```text
Step 0  Lock fixed recipe
   |
   v
Step 1  muP LR pilot on one small model
   |
   v
Step 2  Validate top LR choice on one larger model
   |
   v
Step 3  Architecture-shape pilot at fixed parameter count
   |
   v
Choose final training recipe:
WSD family + LR policy + architecture family + fixed batch size
   |
   v
Step 4  Main N-D data collection runs
   |
   v
Fit scaling law L(N, D)
   |
   v
Use C = 6ND to solve for optimal (N, D) at 1e19 FLOPs
   |
   v
Predict final loss and map N to concrete hyperparameters
```

## 4. Budget Sheet

This plan keeps the total exploratory budget below `2e18` FLOPs.

| Stage | Purpose | Per-run compute | Runs | Stage total |
| --- | --- | ---: | ---: | ---: |
| Step 1 | `muP` LR pilot on one small model | `1e16` | 5 | `5e16` |
| Step 2 | LR validation on one larger model | `3e16` | 2 | `6e16` |
| Step 3 | architecture-shape pilot | `3e16` | 3 | `9e16` |
| Step 4A | main data collection, low band | `5e16` | 4 | `2e17` |
| Step 4B | main data collection, mid band | `1e17` | 4 | `4e17` |
| Step 4C | main data collection, upper-mid band | `2e17` | 4 | `8e17` |
| Step 4D | main data collection, near budget cap | `3e17` | 2 | `6e17` |
| Total | full exploration plan |  | 24 | `1.89e18` |

This leaves about `1.1e17` FLOPs of margin below the hard `2e18` cap.

## 5. Stage-by-Stage Plan

### Step 0. Lock the fixed recipe

Before any scaling runs, define one common recipe.

| Item | Decision |
| --- | --- |
| Batch size | `128` |
| Scheduler family | WSD |
| Optimizer | fixed |
| `d_ff` rule | `d_ff = 4 * d_model` |
| Head-count rule | choose valid divisor of `d_model` |
| Variables to fit scaling law | `N`, `D`, final loss `L` |
| Held fixed in main study | optimizer family, WSD shape, batch size, context length |

Output of this step:

- one fixed training recipe,
- one way to map target parameter count to architecture.

### Step 1. `muP` learning-rate pilot on one small model

Purpose: use `muP` to cheaply identify a transferable LR range.

| Run ID | Compute budget | Model size | Data size | Base LR | Notes |
| --- | ---: | --- | --- | --- | --- |
| A1 | `1e16` | small | chosen by budget | `5e-5` | very conservative |
| A2 | `1e16` | small | chosen by budget | `1e-4` | low |
| A3 | `1e16` | small | chosen by budget | `3e-4` | medium |
| A4 | `1e16` | small | chosen by budget | `6e-4` | high |
| A5 | `1e16` | small | chosen by budget | `1e-3` | aggressive |

Decision rule after Step 1:

- discard unstable or clearly poor LRs,
- keep the best 1 to 2 LR candidates.

### Step 2. Validate the LR choice on one larger model

Purpose: avoid relying on one small model alone.

| Run ID | Compute budget | Model size | Data size | Base LR |
| --- | ---: | --- | --- | --- |
| B1 | `3e16` | small-to-medium | chosen by budget | best LR candidate 1 |
| B2 | `3e16` | small-to-medium | chosen by budget | best LR candidate 2 |

Decision rule after Step 2:

- if the same LR wins, fix one global base LR,
- if the larger model prefers a smaller LR, define a simple LR policy such as using the smaller winning LR for all larger models.

This is the only place where `muP` is used directly in the plan.

### Step 3. Architecture-shape pilot at fixed parameter count

Purpose: choose the model family without assuming `muP` solves architecture allocation.

| Run ID | Compute budget | Parameter target | Shape | Example interpretation |
| --- | ---: | --- | --- | --- |
| C1 | `3e16` | fixed target | deep-narrow | more layers, smaller `d_model` |
| C2 | `3e16` | fixed target | balanced | intermediate depth and width |
| C3 | `3e16` | fixed target | wide-shallow | fewer layers, larger `d_model` |

Decision rule after Step 3:

- choose the shape with the lowest final loss,
- use that shape family for the rest of the experiment.

## 6. Main Scaling-Law Data Collection

The main study fits a Chinchilla-style loss law `L(N, D)` under the fixed recipe. WSD is not itself the scaling law; it is simply the common training schedule used while collecting data.

### 6.1 Main idea

We collect a structured set of runs over model size `N` and dataset size `D`, observe final loss `L`, and fit a law of the form:

`L(N, D) = E + A / N^alpha + B / D^beta`

or another equivalent low-variance parameterization.

After fitting the law, we use the compute constraint

`C ≈ 6ND`

to solve for the best `(N, D)` pair at the target compute budget `C = 1e19`.

### 6.2 Main collection graph

```text
Choose a fixed recipe
   |
   v
Run a small grid of (N, D) pairs
   |
   v
Record final loss L for each pair
   |
   v
Fit L(N, D)
   |
   v
Apply C = 6ND
   |
   v
Find (N_opt, D_opt) at 1e19 FLOPs
```

### 6.3 Main data-collection sheet

The main collection is organized in four compute bands. Within each band, vary `N` and `D` enough to provide signal for the `L(N, D)` fit.

| Band | Per-run compute | Runs | Main purpose |
| --- | ---: | ---: | --- |
| 4A | `5e16` | 4 | dense low-cost coverage |
| 4B | `1e17` | 4 | mid-scale trend estimation |
| 4C | `2e17` | 4 | stronger signal near useful range |
| 4D | `3e17` | 2 | anchor near exploration limit |

### 6.4 Main run template

Each run is one `(N, D)` pair under the fixed recipe.

| Field | Chosen by |
| --- | --- |
| Model size `N` | selected from geometric size ladder |
| Dataset size `D` | chosen so run hits target compute band |
| Compute `C` | approximately `6ND` |
| Batch size | fixed at `128` |
| Scheduler | WSD |
| Architecture family | winner from Step 3 |
| LR | winner or simple LR policy from Steps 1-2 |

### 6.5 Concrete run sheet

Use four model-size levels and several data-size levels. The exact values should be chosen from configurations the API can realize cleanly, but the structure below should be followed.

| Run ID | Compute band | Model size level | Data size level | Purpose |
| --- | --- | --- | --- | --- |
| D1 | `5e16` | `N1` | `D2` | low-band reference |
| D2 | `5e16` | `N2` | `D1` | trade model for data |
| D3 | `5e16` | `N2` | `D2` | balanced reference |
| D4 | `5e16` | `N3` | `D1` | larger model, less data |
| E1 | `1e17` | `N1` | `D3` | more data at small model |
| E2 | `1e17` | `N2` | `D2` | balanced mid-band |
| E3 | `1e17` | `N3` | `D1` | larger model tradeoff |
| E4 | `1e17` | `N3` | `D2` | model-heavy mid-band |
| F1 | `2e17` | `N2` | `D3` | data-heavy upper-mid |
| F2 | `2e17` | `N3` | `D2` | balanced upper-mid |
| F3 | `2e17` | `N4` | `D1` | model-heavy upper-mid |
| F4 | `2e17` | `N4` | `D2` | largest model in repeatable band |
| G1 | `3e17` | `N3` | `D3` | near-cap balanced anchor |
| G2 | `3e17` | `N4` | `D2` | near-cap model-heavy anchor |

Here `N1 < N2 < N3 < N4` are geometrically spaced model-size targets and `D1 < D2 < D3` are geometrically spaced data-size targets. The exact values can be chosen once the available model configurations are known.

## 7. Analysis Sheet

After all runs finish, create the following table.

| Run ID | Model size `N` | Dataset size `D` | Compute `C` | Final loss `L` |
| --- | ---: | ---: | ---: | ---: |
| D1 | fill after runs | fill after runs | fill after runs | fill after runs |
| D2 | fill after runs | fill after runs | fill after runs | fill after runs |
| D3 | fill after runs | fill after runs | fill after runs | fill after runs |
| D4 | fill after runs | fill after runs | fill after runs | fill after runs |
| E1 | fill after runs | fill after runs | fill after runs | fill after runs |
| E2 | fill after runs | fill after runs | fill after runs | fill after runs |
| E3 | fill after runs | fill after runs | fill after runs | fill after runs |
| E4 | fill after runs | fill after runs | fill after runs | fill after runs |
| F1 | fill after runs | fill after runs | fill after runs | fill after runs |
| F2 | fill after runs | fill after runs | fill after runs | fill after runs |
| F3 | fill after runs | fill after runs | fill after runs | fill after runs |
| F4 | fill after runs | fill after runs | fill after runs | fill after runs |
| G1 | fill after runs | fill after runs | fill after runs | fill after runs |
| G2 | fill after runs | fill after runs | fill after runs | fill after runs |

Then fit the loss law `L(N, D)`.

## 8. Final prediction step

After fitting `L(N, D)`:

1. solve for the best `(N, D)` pair under `C = 1e19`,
2. record `N_opt(1e19)` and `D_opt(1e19)`,
3. compute the predicted final loss at that pair,
4. map `N_opt(1e19)` to the nearest feasible Transformer configuration.

Use the architecture family chosen in Step 3 and choose the nearest feasible tuple:

- `d_model`,
- `num_layers`,
- `d_ff = 4 * d_model`,
- `num_heads` dividing `d_model`.

## 9. What is explicitly not optimized

The following quantities are not major optimization targets in this plan:

- batch size,
- scheduler family,
- optimizer family,
- architecture family after Step 3.

They are fixed to keep the scaling-law fit interpretable and to avoid wasting the limited `2e18` exploratory budget.

## 10. Deliverables checklist

| Deliverable | What to include |
| --- | --- |
| Method write-up | `muP` LR pilot, architecture pilot, WSD recipe, scaling-law fit |
| Data table | observed `(N, D, C, L)` values |
| Scaling-law description | functional form, fitting method, assumptions |
| Final prediction | `N_opt(1e19)`, `D_opt(1e19)`, predicted loss |
| Final hyperparameters | nearest feasible Transformer configuration |
