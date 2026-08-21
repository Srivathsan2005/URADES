


# URADES
## Unified Refractory Alloy Descriptor and Embrittlement Screener

URADES is a **hierarchical, physics-informed analytical framework** for rapid screening of Nb-based refractory alloys.

The framework combines a **Global Viability Index (GVI)** for BCC phase-stability screening with three case-specific analytical models for DBTT prediction and embrittlement classification.

URADES is designed as a **rapid screening tool** for identifying potentially viable alloy compositions before detailed CALPHAD analysis and experimental validation.

---

## Framework Overview


                         Alloy composition
                                |
                                v
                    Unit conversion / validation
                                |
                                v
                    Case identification
                                |
                                v
                  Boundary-condition check
                         /           \
                      FAIL            PASS
                       |                |
                       v                v
                   REJECTED           GVI
                                      |
                           +----------+----------+
                           |          |          |
                           v          v          v
                        CASE 1     CASE 2     CASE 3
                        IAS        SR model   EI classifier
                           |          |          |
                           v          v          v
                      DBTT & YS   DBTT & YS   EI / Zone & YS
                           |          |          |
                           +----------+----------+
                                      |
                                      v
                             Screening result


The three alloy regimes are treated separately because the underlying modelling assumptions differ with increasing compositional complexity.

---

# 1. Motivation

Nb-based refractory alloys are promising candidates for high-temperature structural applications because of their high melting temperatures and useful high-temperature mechanical properties.

A major limitation, however, is their susceptibility to low-temperature brittleness and the associated ductile-to-brittle transition temperature (DBTT).

The compositional space of Nb-based refractory alloys is large, while experimentally available DBTT data are comparatively limited.

URADES addresses this screening problem through a hierarchical analytical approach in which:

* alloy compositions are first classified into three compositional regimes;
* boundary conditions are checked;
* BCC viability is assessed using the Global Viability Index;
* a case-specific analytical model is then applied;
* candidate compositions can subsequently be explored through inverse design.

The framework is intended to provide a transparent alternative to treating the entire Nb-based refractory alloy space as a single empirical model.

---

# 2. The Three URADES Cases

URADES separates the Nb-based alloy space into three regimes.

| Case       | Alloy regime                 | Composition basis | Model                            | Output             |
| ---------- | ---------------------------- | ----------------- | -------------------------------- | ------------------ |
| **Case 1** | Dilute Nb engineering alloys | wt% for model     | Independent Alloying Shift (IAS) | DBTT               |
| **Case 2** | Nb-matrix RCCAs              | at%               | Sponge Ratio (SR) model          | DBTT               |
| **Case 3** | Nb-based RHEAs               | at%               | Embrittlement Index (EI)         | Embrittlement zone |

The case is automatically identified from Nb concentration.

```text
Case 1:
Nb >= 79 wt%

Case 2:
Nb < 79 wt% and Nb >= 50 at%

Case 3:
Nb < 50 at%
```

The classifier internally converts between atomic and weight percentages where required.

---

# 3. Global Viability Index (GVI)

The **Global Viability Index (GVI)** is the common phase-stability screening layer of URADES.

GVI is used across **all three cases**.

The global framework is constructed from three sigmoid survival terms:

```text
S_VEC
S_delta
S_SR
```

The terms included in the final GVI depend on the alloy regime.

### Case 1

```text
GVI = S_VEC × S_delta
```

### Case 2

```text
GVI = S_SR × S_VEC × S_delta
```

### Case 3

```text
GVI = S_VEC × S_delta
```

Thus, GVI is a **global descriptor framework**, while the Sponge Ratio survival term is activated specifically for Case 2.

---

## 3.1 VEC Survival Score

The VEC survival score is:

```text
S_VEC = 1 / [1 + exp(15 × (VEC − 5.3))]
```

The VEC threshold used by the model is:

```text
VEC threshold = 5.3
```

VEC is calculated from atomic-fraction-weighted elemental VEC values:

```text
VEC = Σ(x_i × VEC_i)
```

where `x_i` is the atomic fraction of element `i`.

---

## 3.2 Atomic-Size Mismatch Survival Score

The atomic-size mismatch is calculated as:

```text
δ = 100 × [Σ x_i × (1 − r_i / r̄)^2]^(1/2)
```

with:

```text
r̄ = Σ(x_i × r_i)
```

The corresponding survival score is:

```text
S_delta = 1 / [1 + exp(10 × (δ − 6.5))]
```

The calibrated δ threshold is:

```text
δ threshold = 6.5 %
```

---

## 3.3 Sponge Ratio Survival Score

For the GVI phase-stability calculation, the embrittler-buffer ratio is calculated using **weight percent**:

```text
SR_W =
(W + 0.077 × Mo)/(Hf + Zr + Ti + 1)
```

The corresponding survival score is:

```text
S_SR = 1 / [1 + exp(8 × (SR_W − 1.5))]
```

The threshold used for this survival function is:

```text
SR_W threshold = 1.5
```

### Important distinction

The `SR_W` descriptor used inside **GVI** is calculated in **wt%**.

This is distinct from the **Case 2 predictive Sponge Ratio**, which operates on **at%** composition.

---

# 4. GVI Screening Criterion

The common GVI threshold is:

```text
GVI >= 0.5
```

A composition satisfying this condition receives:

```text
PASS
```

A composition with:

```text
GVI < 0.5
```

is flagged as:

```text
SECONDARY PHASE RISK — verify with CALPHAD
```

A low GVI is therefore a **screening flag**, not an automatic computational hard rejection.

Boundary-condition violations are handled separately and result in rejection.

---

# 5. CALPHAD Verification Trigger

The GVI implementation also contains a documented CALPHAD verification condition.

If all three conditions are simultaneously satisfied:

```text
Mo > 5 at%
Hf > 5 at%
Zr > 5 at%
```

the composition is flagged for additional CALPHAD verification regardless of its GVI value.

This condition was introduced based on the CALPHAD validation dataset.

A documented example is:

```text
Nb-5W-10Mo-5Hf-5Zr-5Ti
```

which receives a high GVI value but is identified by CALPHAD as a multiphase alloy.

This demonstrates that GVI is intended as a **rapid screening descriptor**, rather than a replacement for detailed phase-equilibrium calculations.

---

# 6. Case 1 — Independent Alloying Shift (IAS)

Case 1 represents dilute Nb engineering alloys where Nb remains the dominant matrix element.

The model treats individual alloying additions as approximately independent shifts relative to the DBTT of pure Nb.

The baseline value is:

```text
DBTT_Nb = −150 °C
```

The final IAS model implemented in URADES is:

```text
DBTT = −150 + 8W + 15Mo − 5V − 2Ti + 1Zr + 0.5Hf
    
```

All alloying concentrations in this equation are expressed in **wt%**.

The corresponding coefficients are:

| Element |  DBTT shift |
| ------- | ----------: |
| W       |   +8 °C/wt% |
| Mo      |  +15 °C/wt% |
| V       |   −5 °C/wt% |
| Ti      |   −2 °C/wt% |
| Zr      |   +1 °C/wt% |
| Hf      | +0.5 °C/wt% |

The Case 1 dataset contains:

```text
n = 23 alloys
```

with the reported validation metrics:

```text
R²  = 0.856
MAE = 15.9 °C
```

The implementation automatically converts the input composition to wt% before evaluating the IAS model.

---

# 7. Case 2 — Nb-Matrix RCCAs

Case 2 represents Nb-matrix refractory complex concentrated alloys (RCCAs).

The model combines:

1. an alloying-induced DBTT shift;
2. the Sponge Ratio;
3. a multiplicative amplification of the alloying shift.

All Case 2 model concentrations are expressed in **at%**.

---

## 7.1 Alloying Contribution

The model calculates:

```text
ΔT_alloy = 2.244W + 7.899Mo + 1.723(Hf + Zr + Ti)
```

where all concentrations are in at%.

---

## 7.2 Sponge Ratio

The Case 2 predictive Sponge Ratio is:

```text
SR =
W/(Hf + Zr + Ti + 1)
```

The calibrated Mo contribution to this multiplier is:

```text
α = 0
```

Therefore Mo contributes to the direct alloying term but does not enter the Sponge Ratio multiplier in the Case 2 formulation.

---

## 7.3 DBTT Prediction

The Case 2 model is:

```text
DBTT = −150 + ΔT_alloy × (1 + SR)
```

The implementation reports:

```text
LOOCV R² = 0.871
MAE       = 26 °C
```

with:

```text
n = 10
```

for the validated LOOCV dataset.

The full Case 2 dataset contains 11 alloys, with `Nb-40Mo-10Ti` excluded from the reported LOOCV because its Mo concentration exceeds the validated Case 2 limit.

---

# 8. Case 3 — Embrittlement Index (EI)

Case 3 represents Nb-based refractory high-entropy alloys and other multi-principal Nb-based refractory alloys.

Unlike Case 1, these compositions are not treated as dilute perturbations of a dominant Nb matrix.

The Case 3 model uses an Embrittlement Index:

```text
EI =
(W + 0.48Mo)/(Hf + Zr + Ti + 1)
```

All concentrations are expressed in **at%**.

The Mo weighting parameter is:

```text
α = 0.48
```

Thus Mo contributes at 48% of the W weighting within the EI formulation.

---

## 8.1 EI Classification

The implemented classification is:

| EI range           | Classification         |
| ------------------ | ---------------------- |
| EI < 0.10          | Ductile Zone           |
| 0.10 <= EI < 0.50  | Transition Zone        |
| 0.50 <= EI < 15.50 | Brittle Zone           |
| EI >= 15.50        | Confirmed Brittle Zone |

The Case 3 dataset contains:

```text
n = 25 alloys
```

with a reported leave-one-out classification accuracy of:

```text
84% (21/25)
```

Importantly:

```text
EI is a classification descriptor.
EI is not a direct DBTT prediction.
```

---

# 9. The Mo Weighting Transition

A central analysis within URADES concerns the change in the calibrated Mo contribution between the RCCA and RHEA regimes.

The general embrittler expression can be represented as:

```text
W + αMo
```

The final implementations use:

```text
Case 2:
α = 0

Case 3:
α = 0.48
```

Therefore:

```text
RCCA regime  → α = 0
RHEA regime  → α = 0.48
```

The repository includes an α-sensitivity analysis to examine this parameter across the two alloy regimes.

The analysis is located in:

```text
analysis/alpha_sensitivity.py
```

---

# 10. Boundary Conditions

URADES applies explicit composition limits before the corresponding property model is evaluated.

## Case 1 limits

The following limits are expressed in wt%:

| Element | Maximum |
| ------- | ------: |
| W       |  20 wt% |
| Mo      |  10 wt% |
| Hf      |  10 wt% |
| Zr      |   5 wt% |
| Ti      |  10 wt% |

---

## Case 2 limits

The following limits are expressed in at%:

| Quantity     |  Maximum |
| ------------ | -------: |
| W            |   15 at% |
| Mo           |   18 at% |
| Hf           | 22.4 at% |
| Zr           |  8.5 at% |
| Ti           |   10 at% |
| Hf + Zr + Ti | 22.4 at% |

---

## Case 3 limits

The following limits are expressed in at%:

| Element | Maximum |
| ------- | ------: |
| W       |  20 at% |
| Mo      |  20 at% |
| Hf      |  20 at% |
| Zr      |  33 at% |
| Ti      |  33 at% |

Compositions violating the applicable limits are returned as:

```text
REJECTED
```

together with the specific boundary-condition violation.

---

# 11. Complete URADES Pipeline

The main entry point is:

```text
run_URADES()
```

The complete workflow is:

```text
Input composition
       |
       v
Convert composition if required
       |
       v
Identify Case
       |
       v
Check boundary conditions
       |
       +------ fail ------> REJECTED
       |
      pass
       |
       v
Calculate GVI
       |
       +------ GVI < 0.5 ------> FLAGGED
       |
       v
Run case-specific model
       |
       +------------+------------+
       |            |            |
       v            v            v
      IAS           SR           EI
       |            |            |
       v            v            v
     DBTT         DBTT       Classification
       |            |            |
       +------------+------------+
                    |
                    v
              Final result
```

The function returns a structured Python dictionary containing the case, GVI information, status, and case-specific results.

---

# 12. Additional Calculated Quantities

The core implementation also provides supporting quantities used for screening and design.

These include:

* rule-of-mixtures melting temperature;
* estimated density;
* Case 1 yield-strength estimate;
* Case 2 yield-strength estimate;
* composition conversions between at% and wt%.

These supporting quantities are implemented in `core.py` and are primarily used by the screening and inverse-design workflow.

They should be regarded as **screening estimates**, rather than replacements for experimentally measured properties.

---

# 13. Inverse Alloy Design

URADES can be used in the forward direction:

```text
Composition → Predicted property
```

but it can also be used in the inverse direction:

```text
Property requirements → Candidate compositions
```

The inverse-design workflow searches a defined composition space and applies sequential screening constraints.

```text
Composition search space
          |
          v
Case identification
          |
          v
Boundary-condition screening
          |
          v
GVI screening
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

The implementation is provided in:

```text
inverse_design/search.py
```

---

# 14. Validation

The repository separates the final computational framework from its validation scripts.

The validation structure is:

```text
validation/
├── validate_case1.py
├── validate_case2.py
├── validate_case3.py
└── validate_gvi.py
```

This allows the model implementation and the validation procedure to remain independently inspectable.

---

## Case 1

Validation against the 23-alloy literature dataset.

Reported metrics:

```text
R²  = 0.856
MAE = 15.9 °C
n   = 23
```

---

## Case 2

Validation against the Case 2 literature dataset using the defined LOOCV procedure.

Reported metrics:

```text
LOOCV R² = 0.871
MAE      = 26 °C
n        = 10
```

The full dataset contains 11 alloys, with `Nb-40Mo-10Ti` excluded from the reported LOOCV analysis.

---

## Case 3

Validation of the EI classification against 25 reported alloys.

Reported performance:

```text
LOO accuracy = 84%
Correct classifications = 21/25
```

---

## GVI

GVI is additionally compared with CALPHAD results using the documented CALPHAD validation datasets.

The repository contains:

```text
CALPHAD_FULL
CALPHAD_SUMMARY
```

including the documented false-positive case used to establish the additional CALPHAD verification trigger.

---

# 15. Data

All final datasets used by the repository are centralized in:

```text
urades/data.py
```

The module contains:

```text
CASE1_DATA
CASE2_DATA
CASE3_DATA

CALPHAD_FULL
CALPHAD_SUMMARY
```

This provides a **single source of truth** for the data used by the validation and demonstration scripts.

The dataset file documents the composition units, dataset sizes, validation metrics, and CALPHAD information associated with each dataset.

Experimental values should not be modified without updating the corresponding validation analysis.

---

# 16. Core Implementation

The primary computational implementation is contained in:

```text
urades/core.py
```

Major functions include:

```text
atomic_to_weight()
weight_to_atomic()

identify_case()
check_boundary_conditions()

calc_GVI()

predict_case1()
predict_case2()
predict_case3()

run_URADES()
```

Supporting descriptor functions include:

```text
_calc_VEC()
_calc_delta()
_calc_density()
_calc_Tm_ROM()
```

The implementation is deliberately centralized so that the same equations are used by demonstrations, validation, inverse design, and the application.

---

# 17. Example Usage

A simple demonstration is provided in:

```text
examples/demo.py
```

The demonstration is intended to show how a researcher can pass an alloy composition to URADES and obtain:

* automatically identified alloy case;
* Nb content;
* boundary-condition status;
* GVI;
* DBTT prediction or EI classification;
* supporting screening quantities.

A minimal usage pattern is:

```python
from urades.core import run_URADES

composition = { "Nb": 75.4, "Hf": 15.0, "Ti": 5.5, "W": 4.1 }

result = run_URADES(composition, input_unit="at")
```

The actual input unit can be specified as either:

```text
"at"
```

or:

```text
"wt"
```

with the required internal conversions handled by the framework.

---

# 18. Streamlit Application

An interactive Streamlit application is provided in:

```text
app/app.py
```

The application provides an interface for entering alloy compositions and viewing the corresponding URADES screening result.

The application uses the same core functions as the command-line implementation.

Therefore:

```text
Streamlit interface
        |
        v
run_URADES()
        |
        v
Same URADES calculations
```

The scientific equations are not duplicated inside the user interface.

---

# 19. Repository Structure

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

# 20. Installation

Clone the repository:

```bash
git clone <repository-url>
cd URADES
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# 21. Running the Demonstration

Run:

```bash
python examples/demo.py
```

This executes representative alloy compositions through the URADES framework.

---

# 22. Running Validation

### Case 1

```bash
python validation/validate_case1.py
```

### Case 2

```bash
python validation/validate_case2.py
```

### Case 3

```bash
python validation/validate_case3.py
```

### GVI

```bash
python validation/validate_gvi.py
```

---

# 23. Running the α-Sensitivity Analysis

Run:

```bash
python analysis/alpha_sensitivity.py
```

The analysis examines the effect of the Mo weighting parameter α on the corresponding Case 2 and Case 3 model behaviour.

---

# 24. Running the Inverse-Design Search

Run:

```bash
python inverse_design/search.py
```

The script searches the defined composition space and applies the implemented viability and property constraints.

---

# 25. Running the Interactive Application

Launch the Streamlit interface using:

```bash
streamlit run app/app.py
```

---

# 26. Machine-Learning Benchmarking

Machine-learning models were investigated separately during the development of the study.

These include:

* Random Forest;
* Gaussian Process Regression;
* XGBoost.

They are **not part of the core URADES prediction engine**.

The purpose of the repository is to provide the final physics-informed analytical framework rather than to combine the framework with a machine-learning predictor.

Machine-learning benchmarking can therefore be treated as a separate analysis from the core URADES implementation.

---

# 27. Limitations

URADES is a **rapid screening framework** and should not be interpreted as a replacement for detailed materials characterization.

Important limitations include:

### Dataset size

The experimental datasets are relatively small.

### Literature variability

DBTT measurements can be affected by:

* processing history;
* heat treatment;
* microstructure;
* specimen geometry;
* testing conditions;
* measurement methodology.

### Composition-only modelling

The principal models are composition-based and do not explicitly account for every microstructural variable affecting DBTT or embrittlement.

### Phase stability

GVI provides a rapid viability assessment. This GVI methodology do provides considerable understanding about the phase stability, it does not replace CALPHAD or experimental phase identification.

### Extrapolation

The models should be used within their defined compositional boundaries.

### Experimental verification

A composition passing URADES screening should be regarded as a candidate for further investigation, not as an experimentally guaranteed alloy.

---

# 28. Intended Workflow

URADES is intended to be used as an early-stage screening layer:

```text
Large alloy composition space
             |
             v
          URADES
             |
             v
     Boundary conditions
             |
             v
            GVI
             |
             v
     Case-specific model
             |
             v
   Candidate identification
             |
             v
      Inverse design
             |
             v
       CALPHAD analysis
             |
             v
   Experimental validation
```

The purpose is to reduce the number of candidate compositions requiring detailed investigation while retaining an interpretable relationship between composition and predicted behaviour.

---

# 29. Scientific Positioning

URADES is a:

> **Hierarchical physics-informed analytical framework for rapid screening of Nb-based refractory alloys.**

The framework is:

* **Hierarchical** because the Nb-based alloy space is divided into three compositional regimes.
* **Physics-informed** because the descriptors and relationships are motivated by alloy chemistry and phase-stability considerations.
* **Analytical** because the final models are explicit mathematical relationships.
* **Interpretable** because individual composition terms and descriptors can be inspected directly.
* **Screening-oriented** because the framework is designed for rapid candidate evaluation rather than replacing detailed materials analysis.

URADES is not intended to replace:

* CALPHAD;
* first-principles calculations;
* experimental characterization;
* detailed microstructural modelling.

---

# 30. What Is Included

The public repository contains the cleaned implementation required to reproduce the final URADES workflow:

* final model implementation;
* centralized datasets;
* GVI calculation;
* boundary-condition checks;
* Case 1 IAS model;
* Case 2 SR model;
* Case 3 EI classifier;
* validation scripts;
* GVI/CALPHAD comparison;
* α-sensitivity analysis;
* inverse-design search;
* demonstration code;
* Streamlit application.

Development-only and superseded scripts are intentionally excluded.

---

# 31. What Is Not Included

The following are not part of the final core repository:

* obsolete early grid-search implementations;
* broken development scripts;
* scripts dependent on external undocumented/experimental files;
* superseded model formulations;
* partially validated high-temperature property models;
* preliminary machine-learning benchmarking scripts.

This keeps the repository focused on the final URADES framework rather than the entire history of its development.

---

# 32. Reproducibility

The repository is structured so that the principal calculations are centralized and reproducible.

The relationship between the main components is:

```text
data.py
   |
   v
core.py
   |
   +-------- validation/
   |
   +-------- analysis/
   |
   +-------- inverse_design/
   |
   +-------- examples/
   |
   +-------- app/
```

All of these components use the same core implementation rather than maintaining separate copies of the model equations.

This reduces the possibility of different scripts silently using different model definitions.

---

# 33. Citation

If you use URADES, its models, datasets, or associated analyses in academic work, please cite the corresponding publication:

> **[URADES publication citation to be added]**

---

# 34. License

This project is distributed under the license provided in:

```text
LICENSE
```

Please consult the original sources when redistributing literature-derived datasets.

---

# 35. Author

**Srivathsan (SC23B047)**

**Department of Aerospace Engineering, Indian Institute of Space Science and Technology, ISRO / DoS**

Research interests:

* Refractory alloys
* Nb-based alloy design
* Materials informatics
* Computational materials science
* Physics-informed modelling
* Alloy screening

---

# 36. Repository Status

**Research / Reproducible Implementation**

URADES provides a computational framework connecting alloy composition to:

```text
Composition
    |
    v
Case identification
    |
    v
Boundary-condition screening
    |
    v
Global Viability Index
    |
    v
Case-specific analytical model
    |
    v
DBTT prediction / embrittlement classification
    |
    v
Candidate screening
```

The framework is intended to support rapid computational screening of Nb-based refractory alloy compositions prior to detailed CALPHAD and experimental investigation.

```




