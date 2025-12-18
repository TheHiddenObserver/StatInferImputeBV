# [Statistical inference for regression with imputed binary covariates with application to emotion recognition] - Code and Data Repository

## Overview

This repository contains the **implementation code** and **research datasets** for the paper titled "Statistical inference for regression with imputed binary covariates with application to emotion recognition" published in Annals of Applied Statistics, including:

- code for model estimation and inference,
- example notebooks,
- and the datasets used about livestream emotion analysis.


## Publication Information
**Title**: Statistical inference for regression with imputed binary covariates with application to emotion recognition 

**Authors**: Ziqian Lin, Danyang Huang, Ziyu Xiong, and Hansheng Wang

**Venue**: Annals of Applied Statistics (2025)  

**DOI**: [10.1214/24-AOAS1961](10.1214/24-AOAS1961)  

**Preprint**: [arXiv:2408.09619](https://arxiv.org/abs/2408.09619)

## Data Description

```markdown
data/
├── data_full_final.csv   # The full data
├── pilot_final.csv       # The pilot data
├── W_pilot_final.npy     # The extracted feature vectors
└── README.md             # Detailed data documentation
```

## Code Description

``` markdown
codes/
├── calculate_weight.py  # Compute the optimal weights for weighted estimator
├── cov_impute.py        # Compute the covariance estimator
├── fit_model.py         # Fit the imputed and regression model
├── real_data.ipynb      # Implementation using the demostrated data
├── simulator.py         # Simulators for the algorithm
└── utils.py             # Utility functions
```

## Quick Start

You can run the demo notebook via

```bash
jupyter lab codes/real_data.ipynb
```

## Environment

If your environment does not work, try the following version of dependencies:

```markdown
jupyter-client==7.3.4
jupyter-core==4.10.0
numpy==1.22.4
pandas==1.4.2
scikit-learn==1.1.1
statsmodels==0.14.4
```


## Citation 

If you use this code or data in your research, please cite our paper:

**APA style:**

Lin, Z., Huang, D., Xiong, Z., & Wang, H. (2025). Statistical inference for regression with imputed binary covariates with application to emotion recognition. Annals of Applied Statistics, 19(1), 329-350.

**BibTex**

```bibtex
@article{lin2025statistical,
  title={Statistical inference for regression with imputed binary covariates with application to emotion recognition},
  author={Lin, Ziqian and Huang, Danyang and Xiong, Ziyu and Wang, Hansheng},
  journal={The Annals of Applied Statistics},
  volume={19},
  number={1},
  pages={329--350},
  year={2025},
  publisher={Institute of Mathematical Statistics}
}
```


## Contact
If you have questions, you can contact the first author Ziqian Lin (linziqian@stu.pku.edu.cn) or the corresponding author Danyang Huang (dyhuang89@126.com)
