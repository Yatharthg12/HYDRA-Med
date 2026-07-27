# Dataset layout

`raw/diabetic_data.csv` and `raw/IDS_mapping.csv` are the original UCI Diabetes
130-US Hospitals files supplied with this repository. They are never modified by
the experiment pipeline.

`processed/diabetic_data_reduced.csv` is the reproducible 20,000-row,
target-stratified sample (`random_state=42`). If it is absent, the runner recreates
it from the full file while retaining all original columns. `split_assignments.csv`
is generated from eligible encounters after the mapped end-of-life/hospice
exclusion.

The data source is the UCI Machine Learning Repository's *Diabetes 130-US
Hospitals for Years 1999-2008* dataset. Verify the source's current redistribution
terms before publishing copies; this project does not invent or grant a dataset
license.
