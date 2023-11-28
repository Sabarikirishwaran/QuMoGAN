# QuMoGAN
Quantum Small Molecule Drug Generative Adversarial Network (QuMoGAN)

A Hybrid quantum-classical algorithm for generating small molecular drugs using QGAN

## Dependencies
* **python>=3.7**
* **pytorch>=0.4.1**: https://pytorch.org
* **rdkit**: https://www.rdkit.org
* **pennylane**
* **tensorflow==1.15**

## Dataset
* Run bash script `data/gdb9_generater.sh` to download gdb database and then run `data/sparse_molecular_dataset.py` to generate molecular graph dataset used to train the model.

## Training
```
python QuMoGAN_main.py
```

## Prediction
To run the model against test dataset, make sure the model is fully trained first, then run inference.py
```
python inference.py
```
