# SSC_prediction_framework
This is the code for [AI for Learning Deformation Behavior of a Material: Predicting Stress-Strain Curves 4000x Faster Than Simulations](https://ieeexplore.ieee.org/abstract/document/10191138)

## SSC_prediction
- train.py is the code for MLP model training.
- autoencoder.py is the code for autoencoder model training.
- cross_validation.py is the test code for the whole framework.
- ssc_pred.py can predict stress-strain curve for any given orientation (ox and oy)
  - `do_prediction(ox=0, oy=0)`
  
If you want to predict stress-strain curve using this framework, please change the parameters (ox and oy) in function `do_prediction` in ssc_pred.py. Then, run `python ssc_pred.py`. The output stress-strain curve file is image.jpg.

## SSC_predictor

This is the code for the webtool. 

main.py includes the main code for ssc prediction. 

- The function `do_predict(ox=0, oy=0)` can predict and plot stress-strain curve for any orientation number ox and oy.
