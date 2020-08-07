# xECG Project Repository

**Explaining ECG Biometrics: Is It All In The QRS?**    
João Ribeiro Pinto and Jaime S. Cardoso    
*INESC TEC and Universidade do Porto, Portugal* 
joao.t.pinto@inesctec.pt

## Summary
This repository contains the code used for our paper on intepretability for ECG biometrics. In this work, we implemented the model proposed in (1) for ECG biometrics in PyTorch and trained it on the PTB (2) and UofTDB (3) databases, with varying number of subjects. We then applied interpretability tools from Captum (4) to understand how the model behaves on these diverse settings. We find that the QRS complex is the most relevant part of the ECG for identification in smaller sets of subjects and on-the-person signals. Nevertheless, when considering more challenging off-the-person contexts and larger populations, the model uses information from the different ECG waveforms more evenly.  

If you want to know more about this, or if you use our code, please refer to:

> J. R. Pinto and J. S. Cardoso, "Explaining ECG Biometrics: Is It All In The QRS?", in *Proceedings of the 19th International Conference of the Biometrics Special Interest Group (BIOSIG),* 2020.


## Description
This repository includes these python scripts:
1. *prepare_data.py* - 
2. *utils.py* - 


## Setup
To run our code, download this repository or 


## Acknowledgements




## References
(1) Paper chapter
(2)
(3)
