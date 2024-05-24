# SSC_prediction_framework
This is the code for [AI for Learning Deformation Behavior of a Material: Predicting Stress-Strain Curves 4000x Faster Than Simulations](https://ieeexplore.ieee.org/abstract/document/10191138)

## Installation Requirements

## Source Files
### SSC_prediction
- train.py is the code for MLP model training.
- autoencoder.py is the code for autoencoder model training.
- cross_validation.py is the test code for the whole framework.
- ssc_pred.py can predict stress-strain curve for any given orientation (ox and oy)
  - `do_prediction(ox=0, oy=0)`

### SSC_predictor

This is the code for the webtool. 

main.py includes the main code for ssc prediction. 

## Running the code
If you want to predict a stress-strain curve using this framework, please change the parameters (ox and oy) in function `do_prediction` in ssc_pred.py. Then, run `python ssc_pred.py`. The output stress-strain curve file is image.jpg.

## Developer Team
The code was developed by Vishu Gupta from the [CUCIS](http://cucis.ece.northwestern.edu/index.html) group at the Electrical and Computer Engineering Department at Northwestern University.

## Publication
1. Mao, Yuwei, Shahriyar Keshavarz, Vishu Gupta, Andrew CE Reid, Wei-keng Liao, Alok Choudhary, and Ankit Agrawal. "Ai for learning deformation behavior of a material: predicting stress-strain curves 4000x faster than simulations." In 2023 International Joint Conference on Neural Networks (IJCNN), pp. 1-8. IEEE, 2023.[PDF](https://ieeexplore.ieee.org/abstract/document/10191138)

## Disclaimer
The research code shared in this repository is shared without any support or guarantee on its quality. However, please do raise an issue if you find anything wrong and I will try my best to address it.

email: yuweimao2019@u.northwestern.edu

Copyright (C) 2023, Northwestern University.

See COPYRIGHT notice in top-level directory.

## Funding Support
This work is supported in part by the following grants: NIST award 70NANB19H005; DOE awards DE-SC0019358, DE-SC0021399; NSF award CMMI-2053929; and Northwestern Center for Nanocombinatorics.
