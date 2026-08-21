

````markdown
# URADES
## A Physics-Informed Hierarchical Framework for Rapid Screening of Nb-Based Refractory Alloys

URADES is a **hierarchical, physics-informed analytical framework** developed for the rapid screening and design of Nb-based refractory alloys.

The framework combines a **Global Viability Index (GVI)** with case-specific analytical models to evaluate candidate alloys across three Nb-based refractory alloy regimes. It provides a computational route from alloy composition to viability screening, DBTT prediction or embrittlement classification, and inverse alloy design.

URADES is intended for **rapid preliminary screening** before detailed experimental characterization, CALPHAD analysis, or other high-fidelity computational methods.

---

## Framework at a Glance

```text
                         Alloy composition
                                |
                                v
                     Composition validation
                                |
                                v
                      Case identification
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
          CASE 1             CASE 2             CASE 3
     Nb engineering       Nb-matrix RCCAs     Nb-based RHEAs
          alloys                                  /
     multi-principal alloys
             |                  |                  |
             +------------------+------------------+
                                |
                                v
                  GLOBAL VIABILITY INDEX (GVI)
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
        S_VEC x S_delta   S_SR x S_VEC x S_delta   S_VEC x S_delta
             |                  |                  |
             +------------------+------------------+
                                |
                           GVI >= 0.5
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
            IAS                 SR                 EI
             |                  |                  |
             v                  v                  v
           DBTT               DBTT          Embrittlement
          prediction        prediction       classification
                                |
                                v
                       Candidate screening
                                |
                                v
                         Inverse design
````

---

# 1. Motivation

Nb-based refractory alloys are attractive for high-temperature structural applications because of their high melting temperatures and useful high-temperature mechanical properties.

However, their practical application is strongly constrained by low-temperature brittleness and the associated ductile-to-brittle transition temperature (DBTT).

The compositional design space of Nb-based refractory alloys is large, while experimentally measured DBTT datasets are comparatively limited.

This creates a challenge for alloy screening:

* exhaustive experimental testing is expensive;
* available experimental datasets are relatively small;
* different Nb-based alloy regimes do not necessarily follow the same composition-property relationship;
* purely data-driven models can become difficult to justify when the available dataset is small and heterogeneous.

URADES addresses this problem using a **hierarchical physics-informed analytical approach** rather than treating the entire Nb-based alloy space as a single statistical problem.

---

# 2. URADES Philosophy

URADES is built around three principles.

### 1. Regime-specific modelling

Different Nb-based refractory alloy regimes are treated using different physical assumptions.

### 2. Global viability screening

A common **Global Viability Index (GVI)** is used across all three cases to assess whether a composition lies within the defined viable BCC-based screening region.

### 3. Explicit analytical relationships

The final property prediction or embrittlement classification is obtained from explicit composition-based equations rather than a purely black-box machine-learning model.

The resulting framework is:

> **Global viability screening + case-specific physics-informed prediction/classification**

---

# 3. Three URADES Cases

URADES divides the Nb-based alloy space into three regimes.

| Case       | Alloy regime                            | Global viability layer | Case-specific model              | Output                       |
| ---------- | --------------------------------------- | ---------------------- | -------------------------------- | ---------------------------- |
| **Case 1** | Nb engineering alloys                   | GVI                    | Independent Alloying Shift (IAS) | DBTT                         |
| **Case 2** | Nb-matrix RCCAs                         | GVI                    | Sponge Ratio (SR) framework      | DBTT                         |
| **Case 3** | Nb-based RHEAs / multi-principal alloys | GVI                    | Embrittlement Index (EI)         | Embrittlement classification |

The **GVI is applied to all three cases**.

The GVI framework is global, while its mathematical form is adapted to each case by including the survival terms relevant to that alloy regime.

---

# 4. Global Viability Index (GVI)

The **Global Viability Index (GVI)** is the global viability-screening layer of URADES.

It combines sigmoid survival scores associated with:

* embrittler-buffer balance;
* valence electron concentration (VEC);
* atomic-size mismatch (δ).

The general GVI framework is:

**GVI = S_SR × S_VEC × S_δ**

where:

**S_SR = 1 / [1 + exp(8 × (SR_W − 1.5))]**

**S_VEC = 1 / [1 + exp(15 × (VEC − 5.3))]**

**S_δ = 1 / [1 + exp(10 × (δ − 6.5))]**

Each survival score approaches unity when its corresponding descriptor is within its favourable range and decreases as the descriptor approaches or exceeds its calibrated threshold.

---

## 4.1 Sponge Ratio Used in GVI

The embrittler-buffer ratio used by the GVI is:

**SR_W = [W + 0.077 × Mo] / [Hf + Zr + Ti + 1]**

where the elemental concentrations are expressed in wt.%.

The `+1` denominator offset prevents divergence when the buffer-element concentration approaches zero.

---

## 4.2 Valence Electron Concentration

VEC is calculated from atomic-percent composition:

**VEC = Σ(x_i × VEC_i)**

where:

* `x_i` is the atomic fraction of element `i`;
* `VEC_i` is the elemental VEC.

---

## 4.3 Atomic-Size Mismatch

The atomic-size mismatch is calculated as:

**δ = 100 × [Σ x_i × (1 − r_i / r̄)²]^(1/2)**

where:

**r̄ = Σ(x_i × r_i)**

and `r_i` is the atomic radius of element `i`.

---

# 5. Case-Specific Application of GVI

GVI is **global across URADES**, but not every survival term is physically relevant to every alloy regime.

Therefore, the global GVI framework is applied using the appropriate survival terms for each case.

## Case 1

For Nb engineering alloys:

**GVI_Case1 = S_VEC × S_δ**

The `S_SR` term is not included because the embrittler-buffer competition represented by `SR_W` is not considered the governing viability criterion for dilute Nb engineering alloys.

---

## Case 2

For Nb-matrix RCCAs:

**GVI_Case2 = S_SR × S_VEC × S_δ**

The `S_SR` term is included because embrittler-buffer competition is relevant to the Nb-matrix RCCA regime.

---

## Case 3

For Nb-based RHEAs and multi-principal alloys:

**GVI_Case3 = S_VEC × S_δ**

For Case 3, embrittler-buffer competition is represented separately through the Embrittlement Index.

---

## 5.1 GVI Screening Criterion

A common viability criterion is applied across all three cases:

**GVI >= 0.5**

A composition satisfying this condition passes the GVI viability screen and proceeds to the corresponding case-specific model.

Compositions with:

**GVI < 0.5**

are flagged for further verification, such as CALPHAD analysis, or rejected from the subsequent screening pathway according to the defined workflow.

---

# 6. Case 1: Independent Alloying Shift (IAS)

Case 1 represents Nb-dominated engineering alloys in which Nb remains the dominant matrix element.

The underlying assumption is that individual alloying additions behave approximately as independent perturbations to the Nb matrix.

The reference DBTT is:

**DBTT_Nb = −150 °C**

The Independent Alloying Shift model is:

**DBTT = −150 + 8W + 15Mo − 5V − 2Ti + Zr + 0.5Hf**

where alloying concentrations are expressed in wt.% and DBTT is expressed in °C.

The model can therefore be viewed as:

**DBTT = DBTT_Nb + Σ(ΔDBTT_i)**

where each alloying element produces an individual shift relative to the Nb reference state.

### Case 1 workflow

```text
Alloy composition
       |
       v
Global Viability Index
       |
       v
GVI screening
       |
       v
Independent Alloying Shift
       |
       v
Predicted DBTT
```

---

# 7. Case 2: Sponge Ratio Framework

Case 2 represents Nb-matrix refractory complex concentrated alloys (RCCAs).

In this regime, the relative balance between embrittling and buffering elements becomes important.

The Case 2 framework combines:

1. GVI viability screening;
2. a composition-dependent alloying contribution;
3. the Sponge Ratio;
4. DBTT prediction.

---

## 7.1 Sponge Ratio

The Sponge Ratio is defined as:

**SR = [W + 0.077Mo] / [Hf + Zr + Ti + 1]**

where elemental concentrations are expressed in wt.%.

The numerator represents the principal embrittler contribution, with Mo included through the calibrated coefficient.

The denominator represents the buffering contribution of Hf, Zr and Ti.

---

## 7.2 Case 2 DBTT Model

The Case 2 DBTT relationship is:

**DBTT = −150 + ΔT_alloy × (1 + SR)**

where:

**ΔT_alloy = k_W × W + k_Mo × Mo + k_B × (Hf + Zr + Ti)**

The coefficients `k_W`, `k_Mo`, and `k_B` are calibrated using the Case 2 dataset.

The model therefore combines an alloying-induced DBTT shift with amplification associated with the embrittler-buffer balance.

### Case 2 workflow

```text
Alloy composition
       |
       v
Global Viability Index
       |
       v
GVI screening
       |
       v
Sponge Ratio
       |
       v
Alloying contribution
       |
       v
Predicted DBTT
```

---

# 8. Case 3: Embrittlement Index (EI)

Case 3 represents Nb-based refractory high-entropy alloys and other multi-principal refractory alloys.

In this regime, the alloy composition is no longer treated as a dilute perturbation of a dominant Nb matrix.

The embrittlement tendency is represented using an Embrittlement Index:

**EI = [W + 0.48Mo] / [Hf + Zr + Ti + 1]**

The EI is used to classify the alloy according to its embrittlement/DBTT-risk regime.

Importantly:

**EI ≠ DBTT**

EI is therefore a **classification descriptor**, not a direct DBTT prediction equation.

### Case 3 workflow

```text
Alloy composition
       |
       v
Global Viability Index
       |
       v
GVI screening
       |
       v
Embrittlement Index
       |
       v
Embrittlement / DBTT-risk classification
```

---

# 9. The α Transition

One of the key findings examined within URADES is the different role of Mo in the embrittler contribution between alloy regimes.

The parameter `α` controls the Mo contribution to the embrittler term.

For the general Case 2 Sponge Ratio:

**SR = [W + α × Mo] / [Hf + Zr + Ti + 1]**

The calibrated Case 2 formulation corresponds to:

**α_Case2 = 0**

For Case 3, the corresponding EI formulation uses:

**α_Case3 = 0.48**

The repository includes a sensitivity analysis that sweeps `α` across a defined range for the Case 2 and Case 3 datasets independently.

The analysis demonstrates the regime-dependent change in the contribution of Mo and identifies the corresponding optimum/supported α values.

The sensitivity analysis is provided in:

```text
analysis/alpha_sensitivity.py
```

---

# 10. Inverse Alloy Design

URADES can also be used in the inverse direction.

Instead of asking:

> "What DBTT does this composition produce?"

the inverse-design workflow asks:

> "Which compositions satisfy the required property and viability constraints?"

The search procedure evaluates a defined composition space and applies sequential screening criteria.

```text
Composition space
       |
       v
Composition validity
       |
       v
Case identification
       |
       v
GVI screening
       |
       v
Boundary conditions
       |
       v
Property constraints
       |
       +---- DBTT target
       |
       +---- Yield-strength target
       |
       +---- Density target
       |
       v
Candidate ranking
```

The inverse-design implementation is provided in:

```text
inverse_design/search.py
```

A demonstration search is included to illustrate the complete screening workflow.

---

# 11. Validation

URADES is validated against experimentally reported literature data and, where available, independent CALPHAD information for phase-viability assessment.

Validation is separated from the core model implementation.

This allows the user to distinguish between:

* the model itself;
* the data used to calibrate or test it;
* the validation procedure;
* the resulting performance metrics.

---

## 11.1 Case 1 Validation

The Case 1 validation script:

```text
validation/validate_case1.py
```

evaluates the IAS model against the Case 1 dataset.

The analysis includes:

* measured DBTT;
* predicted DBTT;
* absolute error;
* parity plot;
* R²;
* MAE.

The final validation is intended to reproduce the reported Case 1 results.

---

## 11.2 Case 2 Validation

The Case 2 validation script:

```text
validation/validate_case2.py
```

evaluates the Sponge Ratio framework.

The analysis includes:

* measured DBTT;
* predicted DBTT;
* Sponge Ratio;
* absolute error;
* parity plot;
* Leave-One-Out Cross-Validation (LOOCV);
* R²;
* MAE.

The LOOCV implementation uses the defined Case 2 validation dataset and exclusion criteria reported in the associated study.

---

## 11.3 Case 3 Validation

The Case 3 validation script:

```text
validation/validate_case3.py
```

evaluates the EI-based classification.

The analysis includes:

* Embrittlement Index;
* predicted classification;
* experimental classification;
* confusion matrix;
* classification accuracy.

The Case 3 framework is therefore evaluated as a classification model rather than as a direct DBTT regression model.

---

## 11.4 GVI Validation

The GVI validation script:

```text
validation/validate_gvi.py
```

compares GVI predictions against the available CALPHAD-based phase-stability information.

The validation includes the defined CALPHAD alloy tables and explicitly identifies cases where the GVI and CALPHAD classifications differ.

The repository also documents the relevant boundary-condition interpretation for cases where a discrepancy occurs.

---

# 12. Validation Metrics

For regression-based models, the following metrics are used.

### Mean Absolute Error

**MAE = (1/N) × Σ|y_i − ŷ_i|**

### Root Mean Squared Error

**RMSE = [(1/N) × Σ(y_i − ŷ_i)²]^(1/2)**

### Coefficient of Determination

**R² = 1 − [Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²]**

For classification-based models, classification accuracy and confusion matrices are used.

Training-set performance and cross-validated performance are kept separate.

---

# 13. Validation Summary

The validation scripts reproduce the principal validation results associated with the URADES study.

| Component        | Validation approach             | Primary metrics            |
| ---------------- | ------------------------------- | -------------------------- |
| **Case 1 – IAS** | Literature DBTT dataset         | R², MAE, RMSE              |
| **Case 2 – SR**  | LOOCV + literature DBTT dataset | R², MAE, RMSE              |
| **Case 3 – EI**  | Classification validation       | Accuracy, confusion matrix |
| **GVI**          | CALPHAD comparison              | Agreement / classification |

The numerical results are generated directly by the validation scripts rather than manually entered into the prediction engine.

---

# 14. Machine-Learning Benchmarking

Machine-learning models were investigated during development as independent benchmarks.

They are not part of the core URADES framework.

The benchmark models include:

* Random Forest (RF);
* Gaussian Process Regression (GPR);
* XGBoost in the development study.

These models are kept separate from the core repository because the primary purpose of URADES is the physics-informed analytical framework.

The ML benchmarking can be added as a separate component in a future repository version.

Importantly, URADES-derived quantities such as GVI, SR, EI, or predicted DBTT are not used as input features when evaluating whether conventional ML can independently reproduce the target property.

This avoids information leakage from the proposed framework into the benchmark models.

---

# 15. Reproducibility

The repository is designed to reproduce the computational workflow using the files contained within the repository.

The core implementation does not require external Excel spreadsheets or hidden development files.

The repository contains:

* the final URADES engine;
* the three experimental datasets;
* CALPHAD validation data;
* validation scripts;
* α-sensitivity analysis;
* inverse-design search;
* demonstration scripts;
* Streamlit application.

All final numerical parameters and model equations should be traceable to the associated study and documented validation procedure.

---

# 16. Repository Structure

```text
URADES/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── urades/
│   ├── __init__.py
│   ├── core.py
│   └── data.py
│
├── validation/
│   ├── validate_case1.py
│   ├── validate_case2.py
│   ├── validate_case3.py
│   └── validate_gvi.py
│
├── analysis/
│   └── alpha_sensitivity.py
│
├── inverse_design/
│   └── search.py
│
├── app/
│   └── app.py
│
└── examples/
    └── demo.py
```

---

# 17. Core Package

The main computational engine is contained in:

```text
urades/core.py
```

The final implementation contains the following major components:

```text
atomic_to_weight()
identify_case()
check_boundary_conditions()

calc_GVI()

predict_case1()
predict_case2()
predict_case3()

run_URADES()
```

The core engine contains the final physics-based framework only.

Machine-learning benchmarking, exploratory parameter searches, plotting, and development-history calculations are kept outside the core model.

---

# 18. Data

The datasets required by the final framework are stored in:

```text
urades/data.py
```

The module contains:

* Case 1 dataset;
* Case 2 dataset;
* Case 3 dataset;
* CALPHAD data used for GVI validation.

The datasets are kept in a single location so that all validation and demonstration scripts use the same source data.

This provides a single source of truth for the repository.

Each dataset should retain its literature/source information where applicable.

---

# 19. Example

A minimal demonstration is provided in:

```text
examples/demo.py
```

The demonstration runs representative compositions through the complete URADES workflow.

Example compositions include:

* Cb-752;
* C-103;
* V1;
* a representative RHEA.

The example demonstrates automatic routing of an input composition through:

```text
Composition
     |
     v
Case identification
     |
     v
GVI
     |
     v
Boundary conditions
     |
     v
Case-specific model
     |
     v
Final result
```

The example is intended to provide a rapid demonstration of the framework.

---

# 20. Streamlit Application

A simple interactive interface is provided in:

```text
app/app.py
```

The application allows the user to specify alloy composition and automatically evaluates:

* alloy case;
* boundary-condition status;
* GVI;
* case-specific prediction/classification.

The application uses the same functions as the core URADES implementation.

The Streamlit interface does not contain a separate copy of the scientific equations.

This ensures that the application and command-line calculations use the same underlying model.

---

# 21. Running the Repository

Clone the repository:

```bash
git clone <repository-url>
cd URADES
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the demonstration:

```bash
python examples/demo.py
```

Run Case 1 validation:

```bash
python validation/validate_case1.py
```

Run Case 2 validation:

```bash
python validation/validate_case2.py
```

Run Case 3 validation:

```bash
python validation/validate_case3.py
```

Run GVI validation:

```bash
python validation/validate_gvi.py
```

Run the α-sensitivity analysis:

```bash
python analysis/alpha_sensitivity.py
```

Run the inverse-design search:

```bash
python inverse_design/search.py
```

Launch the Streamlit application:

```bash
streamlit run app/app.py
```

---

# 22. Limitations

URADES is a **screening framework**, not a replacement for experimental or high-fidelity computational methods.

Important limitations include:

### Limited dataset size

The available experimental datasets are small compared with conventional machine-learning datasets.

### Dataset heterogeneity

Literature DBTT measurements can differ in:

* alloy processing history;
* heat treatment;
* microstructure;
* specimen geometry;
* testing conditions;
* measurement methodology.

### Model applicability

Each analytical model is developed for a defined compositional regime.

Extrapolation outside the applicable composition range should therefore be treated cautiously.

### Microstructural effects

URADES is primarily composition-based and does not explicitly model every microstructural factor affecting DBTT and mechanical behaviour.

### Phase stability

GVI provides a rapid viability screen. It does not replace CALPHAD or experimental phase identification.

### Experimental verification

A favourable URADES result identifies a candidate for further investigation. It does not guarantee experimental success.

---

# 23. Intended Use

URADES is intended for the early-stage screening and design of Nb-based refractory alloys.

A typical workflow is:

```text
Large composition space
        |
        v
      URADES
        |
        v
Global viability screening
        |
        v
Case-specific prediction
        |
        v
Candidate ranking
        |
        v
Inverse design
        |
        v
CALPHAD / detailed modelling
        |
        v
Experimental validation
```

The objective is to reduce the number of compositions requiring detailed investigation while retaining an interpretable connection between composition and predicted behaviour.

---

# 24. Scientific Positioning

URADES can be described as a:

> **Hierarchical physics-informed analytical framework for rapid screening of Nb-based refractory alloys.**

The framework is:

* **physics-informed** because it uses composition-derived descriptors and physically motivated relationships;
* **analytical** because the principal models are explicit equations;
* **hierarchical** because different alloy regimes are treated using different physical assumptions;
* **interpretable** because the model structure and descriptors can be directly examined;
* **screening-oriented** because it is intended to identify promising compositions before detailed evaluation.

URADES is not intended to be classified as:

* a purely machine-learning model;
* a CALPHAD framework;
* a first-principles method;
* a replacement for experimental characterization.

---

# 25. Development and Repository Scope

The original development of URADES involved:

* descriptor calculations;
* multiple model formulations;
* parameter fitting;
* sensitivity studies;
* validation;
* LOOCV analysis;
* CALPHAD comparison;
* inverse-design searches;
* machine-learning benchmarking;
* application development.

The public repository contains the **cleaned final implementation and the analyses required to understand and reproduce the final framework**.

Development-only scripts, temporary calculations, superseded implementations, unrelated scripts, and unvalidated models are intentionally excluded.

Examples of excluded development material include:

* obsolete grid-search implementations;
* broken intermediate scripts;
* unrelated exploratory code;
* external-file-dependent development scripts;
* high-temperature yield-strength models that remain under development;
* preliminary ML benchmarking code.

---

# 26. Citation

If you use URADES, its models, datasets, or associated analyses in academic work, please cite the corresponding publication:

> **[URADES publication citation to be added]**

---

# 27. License

This project is distributed under the license provided in:

```text
LICENSE
```

Please check the individual data sources and cited literature when redistributing datasets derived from external publications.

---

# 28. Author

**Srivathsan (SC23B047)**

**Department of Aerospace Engineering, Indian Institute of Space Science and Technology, ISRO / DoS**

Research interests:

* Refractory alloys
* Computational materials science
* Materials informatics
* Nb-based alloy design
* Alloy screening
* Physics-informed modelling

---

# 29. Repository Status

**Status: Research / Reproducible Implementation**

URADES is provided as a research framework for rapid screening and computational design of Nb-based refractory alloys.

The repository is intended to provide a transparent connection between:

```text
Composition
    |
    v
Global Viability Index (GVI)
    |
    v
Case-specific model
    |
    v
Prediction / classification
    |
    v
Candidate screening
```

and the corresponding experimental and CALPHAD validation.

```



