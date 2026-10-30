# Ultra experiment results

All numbers below are from the controlled stochastic contact-outcome benchmark,
not Genesis physics. Seed 37; 12,000 candidate training rows; 6,400 A/B/C/D
evaluation objects; 2,000 ablation objects.

| Configuration | Known correct-bin | Known first-grasp | Mean attempts | Mean time |
|---|---:|---:|---:|---:|
| A | 88.8% | 53.8% | 1.67 | 18.21 s |
| B | 90.0% | 61.8% | 1.54 | 16.64 s |
| C | 90.5% | 62.1% | 1.55 | 17.72 s |
| D | 90.0% | 61.9% | 1.52 | 17.39 s |

Open-set unknown false acceptance was A/B/C 100.0% versus D 5.3%, while
known-shape correct-bin success in that scene remained 97.7% for all four.
Combined-unseen success was A 49.0%, B 54.0%, C 52.5%, D 50.0%; this is the
main counterexample showing that re-observation and conservative holds are not
universally beneficial.

Verification: 13 tests passed; offline RGB-D E2E passed; PDF 12 pages rendered;
H.264 replay video 64.0 s at 1280x720/24 fps. Genesis qualification did not pass
because torch/genesis could not be installed in the managed container.
