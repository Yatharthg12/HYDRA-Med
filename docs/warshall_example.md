# Warshall Reachability Example

This component is a five-node mathematical demonstration. It is not applied to
the complete healthcare graph and is not the relationship-robustness simulation.

## Entities and starting matrix

`P = Patient`, `D = Disease`, `M = Medication`, `L = Lab Test`,
`C = Complication`.

| From \ To | P | D | M | L | C |
|---|---:|---:|---:|---:|---:|
| P | 0 | 1 | 0 | 1 | 0 |
| D | 0 | 0 | 1 | 0 | 1 |
| M | 0 | 0 | 0 | 0 | 0 |
| L | 0 | 1 | 0 | 0 | 0 |
| C | 0 | 0 | 0 | 0 | 0 |

The update is:

```text
T[i][j] = T[i][j] OR (T[i][k] AND T[k][j])
```

The implementation stores `T^(0)` plus one matrix after considering each of the
five possible intermediates.

## Iterations and discoveries

### T^(0): direct relationships

Patient directly reaches Disease and Lab Test. Disease directly reaches
Medication and Complication. Lab Test directly reaches Disease.

### T^(1): Patient as intermediate

No node reaches Patient, so Patient cannot produce a new indirect path.

### T^(2): Disease as intermediate

Four indirect pairs appear:

- Patient → Medication: Patient reaches Disease, which reaches Medication.
- Patient → Complication: Patient reaches Disease, which reaches Complication.
- Lab Test → Medication: Lab Test reaches Disease, which reaches Medication.
- Lab Test → Complication: Lab Test reaches Disease, which reaches Complication.

### T^(3): Medication as intermediate

Medication has no outgoing relationships, so no pair is added.

### T^(4): Lab Test as intermediate

Patient already reaches everything that Lab Test can reach after `T^(2)`, so no
new pair is added.

### T^(5): Complication as intermediate

Complication has no outgoing relationships, so no pair is added.

## Verified final closure

| From \ To | P | D | M | L | C |
|---|---:|---:|---:|---:|---:|
| P | 0 | 1 | 1 | 1 | 1 |
| D | 0 | 0 | 1 | 0 | 1 |
| M | 0 | 0 | 0 | 0 | 0 |
| L | 0 | 1 | 1 | 0 | 1 |
| C | 0 | 0 | 0 | 0 | 0 |

The program asserts this matrix. It proves that Patient reaches Disease,
Medication, Lab Test, and Complication; Disease reaches Medication and
Complication; and Lab Test reaches Disease, Medication, and Complication.

Warshall requires `O(V³)` time. Storing the matrix requires `O(V²)` memory in
this implementation. Computing closure on tens of thousands of heterogeneous
nodes would be both unnecessary for prediction and computationally infeasible;
the complete research graph instead uses sparse local relationships.

## Dashboard interaction contract

The `/warshall` page initializes at `T^(0)`. Its slider, Previous and Next
buttons, numeric counter, selected matrix, intermediate-node banner,
explanation, calculations, and compact directed graph are all rendered from the
same `currentStep` value. Both slider `input` and `change` events render the
selected step. Previous is disabled at step 0 and Next at step 5.

Only cells added in the selected iteration receive the new-path highlight.
Known reachability, unreachable pairs, and diagonal cells use separate legend
states. Reset returns to direct relationships. Auto-play advances through the
six states and can be paused. The controls use native buttons and a labelled
range input, and Left/Right arrow navigation is supported when the matrix panel
has focus.

For every new cell, the page displays a plain-English path and the previous-step
logical calculation. For example:

```text
Patient can reach Medication through Disease.

T[Patient, Medication] =
T_previous[Patient, Medication] OR
(T_previous[Patient, Disease] AND T_previous[Disease, Medication])
= 0 OR (1 AND 1) = 1
```

The page uses only local HTML, CSS, JavaScript, and SVG resources.
