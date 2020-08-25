# xECG Project Repository

**Explaining ECG Biometrics: Is It All In The QRS?**    
João Ribeiro Pinto and Jaime S. Cardoso    
*INESC TEC and Universidade do Porto, Portugal*   
joao.t.pinto@inesctec.pt

## Summary
This repository contains the code used for our paper on intepretability for ECG biometrics. In this work, we implemented the model proposed in (1) for ECG biometrics in PyTorch and trained it on the PTB (2,3) and UofTDB (4) databases, with varying number of subjects. We then applied interpretability tools from Captum (5) to understand how the model behaves on these diverse settings. We find that the QRS complex is the most relevant part of the ECG for identification in smaller sets of subjects and on-the-person signals. Nevertheless, when considering more challenging off-the-person contexts and larger populations, the model uses information from the different ECG waveforms more evenly.  

If you want to know more about this, or if you use our code, check out our paper:    
**J. R. Pinto and J. S. Cardoso, "Explaining ECG Biometrics: Is It All In The QRS?", in *Proceedings of the 19th International Conference of the Biometrics Special Interest Group (BIOSIG),* 2020.**    
[[bib]](https://github.com/jtrpinto/xECG/blob/master/citation.bib) [[pdf]](https://jtrpinto.github.io/files/pdf/jpinto2020biosig.pdf)

## Description
This repository includes the python scripts used to train, test, and interpret the models with PTB and UofTDB data. The *models* directory includes trained models with PTB, the *results* directory includes the test scores of each trained model, the *plots* directory includes explanation figures from the first two subjects of each database, and the *peak_locations* directory includes some annotations on the R-peaks of the first two subjects of each database and a script to label more.

To ensure the PTB and UofTDB data is not redistributed, especially UofTDB, this repository includes limited trained models, test scores, and explanation plots. Nevertheless, anyone with access to the data and this code should be able to replicate our results exactly:
1. Use *prepare_data.py* to transform the raw databases in prepared data samples;
2. Use *train_model_X.py* to train a model;
3. Use *test_model_X.py* to obtain test predictions with the trained model;
4. Use *interpret_X.py* to compute explanations using the interpretability tools;
5. Use *get_plots.py* to generate explanation plots of the signals.

Do not forget to set the needed variables at the beginning of each script.

#### Model training details
We trained the model over a maximum of 2500 (PTB) or 5000 (UofTDB) epochs with batch size 2N, using the Adam optimiser with initial learning rate 10^(-3). Early stopping was used, with patience of 100 (PTB) or 250 (UofTDB) epochs, based on loss values obtained in 10% of training data used for validation. As in (1), random permutations were used as data augmentation for model regularisation, as well as dropout, before the last fully-connected layer (p=0.2 for PTB and p=0.5 for UofTDB). For UofTDB, L2 weight regularisation was also used, with lambda=10^(-3).

## Setup
To run our code, download or clone this repository and use *requirements.txt* to set up a pip virtual environment with the needed dependencies.

You will also need the data from the PTB and UofTDB databases. The PTB database is quickly accessible at [Physionet](https://physionet.org/content/ptbdb/1.0.0/). To get the UofTDB data, you should contact the [BioSec.Lab at the University of Toronto](https://www.comm.utoronto.ca/~biometrics/). 

## Acknowledgements
This work was financed by the ERDF - European Regional Development Fund through the Operational Programme for Competitiveness and Internationalization - COMPETE 2020 Programme and by National Funds through the Portuguese funding agency, FCT - Fundação para a Ciência e a Tecnologia within project "POCI-01-0145-FEDER-030707", and within the PhD grant "SFRH/BD/137720/2018". The authors wish to thank the creators and administrators of the PTB (Physikalisch-Technische Bundesanstalt, Germany) and UofTDB (University of Toronto, Canada) databases, which have been essential for this work.

## References
(1) Pinto, J. R.; Cardoso, J. S.; Lourenço, A.: Deep Neural Networks For Biometric Identification Based On Non-Intrusive ECG Acquisitions. In: The Biometric Computing: Recognition and Registration, chapter 11, pp. 217–234. CRC Press, 2019.  
(2) Bousseljot, R.; Kreiseler, D.; Schnabel, A.: Nutzung der EKG-Signaldatenbank CARDIODAT der PTB ̈uber das Internet. Biomedizinische Technik, 40(1), 1995.   
(3) Goldberger, A.; Amaral, L.; Glass, L.; Hausdorff, J.; Ivanov, P. C.; Mark, R.; Stanley, H. E.: PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23):e215–e220, 2000.   
(4) Wahabi, S.; Pouryayevali, S.; Hari, S.; Hatzinakos, D.: On Evaluating ECG Biometric Systems: Session-Dependence and Body Posture. IEEE Transactions on Information Forensics and Security, 9(11):2002–2013, Nov 2014.   
(5) Kokhlikyan, N.; Miglani, V.; Martin, M.; Wang, E.; Reynolds, J.; Melnikov, A.; Lunova, N.; Reblitz-Richardson, O.: PyTorch Captum. https://github.com/pytorch/captum, 2019.
