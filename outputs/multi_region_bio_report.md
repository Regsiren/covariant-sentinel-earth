# BIO_05 Multi-Region Flash Drought Benchmark (2015–2024)

- **Window:** 2015-01-01 → 2024-12-31 (10 years × 3 regions = 30 site-years)

### East_Anglia_UK (Temperate Maritime)

- **Flash drought events:** 12

| Detector | Recall | Precision | F1 | Mean Lead (d) | Alert Rate (%) | False Alert Days | Lead/100 FA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `legacy_30d_soil_z` | 0.667 | 0.129 | 0.216 | 9.12 | 16.32 | 477 | 1.91 |
| `bio05_thinning_or_veto` | 0.833 | 0.125 | 0.217 | 17.10 | 18.56 | 614 | 2.79 |
| `bio05_veto_only` | 0.250 | 0.048 | 0.080 | 20.00 | 7.86 | 269 | 7.43 |
| `bio05_drying_thinning_or_veto` | 0.667 | 0.116 | 0.198 | 13.50 | 4.60 | 126 | 10.71 |
| `bio05_drying_veto_only` | 0.250 | 0.100 | 0.143 | 20.00 | 1.53 | 41 | 48.78 |
| `bio05_twostage_confirm` | 0.750 | 0.113 | 0.196 | 15.00 | 14.37 | 402 | 3.73 |

- **Event dates:** 2015-04-16, 2016-07-19, 2017-06-18, 2018-06-19, 2019-05-25, 2019-08-27, 2020-04-15, 2020-08-09, 2022-05-01, 2022-06-19, 2023-06-12, 2024-08-11

### Iowa_US_CornBelt (Continental)

- **Flash drought events:** 18

| Detector | Recall | Precision | F1 | Mean Lead (d) | Alert Rate (%) | False Alert Days | Lead/100 FA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `legacy_30d_soil_z` | 0.667 | 0.203 | 0.312 | 12.08 | 13.00 | 305 | 3.96 |
| `bio05_thinning_or_veto` | 0.667 | 0.203 | 0.312 | 14.58 | 22.39 | 749 | 1.95 |
| `bio05_veto_only` | 0.222 | 0.103 | 0.140 | 13.25 | 13.22 | 469 | 2.83 |
| `bio05_drying_thinning_or_veto` | 0.444 | 0.129 | 0.200 | 13.25 | 4.19 | 127 | 10.43 |
| `bio05_drying_veto_only` | 0.167 | 0.097 | 0.122 | 12.33 | 1.72 | 60 | 20.56 |
| `bio05_twostage_confirm` | 0.722 | 0.210 | 0.325 | 13.85 | 11.85 | 329 | 4.21 |

- **Event dates:** 2015-10-17, 2016-07-28, 2017-06-06, 2017-08-01, 2018-07-30, 2019-07-04, 2020-07-05, 2021-04-28, 2021-07-25, 2021-09-10, 2021-12-04, 2022-07-20, 2022-10-01, 2023-05-24, 2023-08-20, 2023-11-16, 2024-03-12, 2024-09-03

### Castile_Spain (Semi-Arid Mediterranean)

- **Flash drought events:** 8

| Detector | Recall | Precision | F1 | Mean Lead (d) | Alert Rate (%) | False Alert Days | Lead/100 FA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `legacy_30d_soil_z` | 0.875 | 0.115 | 0.203 | 17.29 | 16.89 | 530 | 3.26 |
| `bio05_thinning_or_veto` | 1.000 | 0.250 | 0.400 | 16.75 | 43.20 | 1443 | 1.16 |
| `bio05_veto_only` | 1.000 | 0.143 | 0.250 | 16.50 | 27.51 | 917 | 1.80 |
| `bio05_drying_thinning_or_veto` | 0.750 | 0.081 | 0.146 | 14.83 | 15.58 | 483 | 3.07 |
| `bio05_drying_veto_only` | 0.750 | 0.092 | 0.164 | 14.50 | 10.38 | 315 | 4.60 |
| `bio05_twostage_confirm` | 1.000 | 0.129 | 0.229 | 11.62 | 27.95 | 884 | 1.32 |

- **Event dates:** 2017-07-22, 2018-06-16, 2019-02-17, 2019-09-30, 2021-03-24, 2022-09-17, 2023-02-20, 2023-09-24

## Pooled 30 Site-Year Summary

- **Total flash drought events:** 38
- **Events per region:** East_Anglia_UK=12, Iowa_US_CornBelt=18, Castile_Spain=8

### Pooled 30 Site-Year Summary Table

- **Flash drought events:** 38

| Detector | Recall | Precision | F1 | Mean Lead (d) | Alert Rate (%) | False Alert Days | Lead/100 FA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `legacy_30d_soil_z` | 0.711 | 0.148 | 0.245 | 12.56 | 15.40 | 1312 | 0.96 |
| `bio05_thinning_or_veto` | 0.789 | 0.175 | 0.287 | 16.00 | 28.05 | 2806 | 0.57 |
| `bio05_veto_only` | 0.395 | 0.095 | 0.153 | 16.33 | 16.20 | 1655 | 0.99 |
| `bio05_drying_thinning_or_veto` | 0.579 | 0.107 | 0.181 | 13.77 | 8.12 | 736 | 1.87 |
| `bio05_drying_veto_only` | 0.316 | 0.095 | 0.146 | 15.33 | 4.54 | 416 | 3.69 |
| `bio05_twostage_confirm` | 0.789 | 0.147 | 0.248 | 13.60 | 18.06 | 1615 | 0.84 |

## 95% Site-Year Block Bootstrap CI & Paired Delta Table

- **Blocks:** 30 site-years
- **Bootstrap replicates:** 1000 (seed=42)
- **Baseline:** `legacy_30d_soil_z`

### Detector metrics (point [95% CI])

| Detector | Recall | Precision | F1 | Mean Lead (d) |
|---|---|---|---|---|
| `legacy_30d_soil_z` | 0.711 [0.564, 0.849] | 0.145 [0.105, 0.186] | 0.241 [0.178, 0.299] | 12.56 [10.15, 15.08] |
| `bio05_thinning_or_veto` | 0.789 [0.667, 0.896] | 0.155 [0.109, 0.202] | 0.260 [0.190, 0.327] | 16.00 [13.52, 18.21] |
| `bio05_veto_only` | 0.395 [0.242, 0.543] | 0.085 [0.045, 0.124] | 0.140 [0.075, 0.196] | 16.33 [13.19, 18.64] |
| `bio05_drying_thinning_or_veto` | 0.579 [0.421, 0.718] | 0.104 [0.067, 0.142] | 0.177 [0.115, 0.234] | 13.77 [10.40, 16.39] |
| `bio05_drying_veto_only` | 0.316 [0.172, 0.457] | 0.092 [0.046, 0.138] | 0.142 [0.073, 0.207] | 15.33 [11.14, 19.50] |
| `bio05_twostage_confirm` | 0.789 [0.676, 0.895] | 0.142 [0.103, 0.183] | 0.241 [0.180, 0.300] | 13.60 [10.79, 16.39] |

### Paired deltas vs `legacy_30d_soil_z` (point [95% CI])

| Detector | Δ Recall | Δ F1 | Δ Mean Lead (d) |
|---|---|---|---|
| `bio05_thinning_or_veto` | 0.079 [-0.079, 0.220] | 0.019 [-0.034, 0.080] | 3.44 [0.15, 6.55] |
| `bio05_veto_only` | -0.316 [-0.500, -0.150] | -0.102 [-0.163, -0.044] | 3.78 [0.55, 7.02] |
| `bio05_drying_thinning_or_veto` | -0.132 [-0.333, 0.051] | -0.064 [-0.116, -0.010] | 1.22 [-2.00, 4.20] |
| `bio05_drying_veto_only` | -0.395 [-0.567, -0.250] | -0.099 [-0.157, -0.042] | 2.78 [-1.44, 7.59] |
| `bio05_twostage_confirm` | 0.079 [-0.070, 0.222] | -0.000 [-0.047, 0.051] | 1.04 [-2.18, 4.55] |
