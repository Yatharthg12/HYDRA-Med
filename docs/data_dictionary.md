# Data Dictionary and Transformations

The dashboard exposes a searchable entry for every original field plus the
complete descriptions parsed from `IDS_mapping.csv`. This document groups
fields by their implemented treatment.

## Metadata and target

| Field | Treatment |
|---|---|
| `encounter_id` | Preserved for joins/dashboard; excluded from predictors |
| `patient_nbr` | Preserved for grouping/dashboard; excluded from predictors |
| `readmitted` | `<30` → 1; `NO` and `>30` → 0; original field not predicted |

## Demographic and encounter context

| Fields | Treatment |
|---|---|
| `race`, `gender` | Categorical; missing becomes `Unknown` |
| `age` | Interval retained as `age_group`; numeric midpoint is modelled |
| `weight` | Removed after auditable >96% missingness |
| `payer_code`, `medical_specialty` | Retained; missing becomes `Unknown`; rare training values become `Other` |
| `admission_type_id` | Replaced as predictor by readable mapped description |
| `discharge_disposition_id` | Mapped for eligibility and discharge-time category |
| `admission_source_id` | Replaced as predictor by readable mapped description |

End-of-life eligibility searches mapped discharge text for expiration, death,
or hospice language. The numeric IDs are not hard-coded.

## Numeric utilization and treatment counts

`time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`,
`number_outpatient`, `number_emergency`, `number_inpatient`, and
`number_diagnoses` are converted to numeric values, median-imputed from training
statistics, and standardized.

## Diagnoses

`diag_1`, `diag_2`, and `diag_3` map reproducibly:

| ICD-9 range | Category |
|---|---|
| 250.xx | Diabetes |
| 390–459 or 785 | Circulatory |
| 460–519 or 786 | Respiratory |
| 520–579 or 787 | Digestive |
| 580–629 or 788 | Genitourinary |
| 710–739 | Musculoskeletal |
| 800–999 | Injury |
| 140–239 | Neoplasm |
| V codes, E codes, and remaining valid codes | Other |
| Missing/unparseable | Unknown |

## Laboratory and treatment indicators

| Field | Treatment |
|---|---|
| `max_glu_serum` | Missing means `Not Measured` |
| `A1Cresult` | Missing means `Not Measured` |
| `change` | Categorical medication-change indicator |
| `diabetesMed` | Categorical diabetes-medication indicator |

## Medication statuses

The 23 source fields are:

`metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`,
`acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`,
`rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`,
`examide`, `citoglipton`, `insulin`, `glyburide-metformin`,
`glipizide-metformin`, `glimepiride-pioglitazone`,
`metformin-rosiglitazone`, and `metformin-pioglitazone`.

For each, `Up`, `Down`, or `Steady` produces an active indicator of 1; `No`
produces 0. These compact indicators enter the predictive matrix and determine
`received_medication` graph edges.

## Derived fields

Derived fields include `target`, `age_group`, `age_midpoint`, three diagnosis
categories, three readable mapping descriptions, and 23 medication-active
indicators. They do not alter the preserved raw CSV.
