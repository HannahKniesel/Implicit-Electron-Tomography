# Implicit-Electron-Tomography

This is the official git repository of 

> ## ***"Clean Implicit 3D Structure from Noisy 2D STEM Images", CVPR 2022***  

https://arxiv.org/abs/2203.15434

*Abstract: STEM acquire 2D images of a 3D sample on the scale of individual cell components.
Unfortunately, these 2D images can be too noisy to be fused into a useful 3D structure and facilitating good denoisers is challenging due to the lack of clean-noisy pairs.
Additionally, representing detailed 3D structure can be difficult even for clean data when using regular 3D grids.
Addressing these two limitations, we suggest a differentiable image formation model for STEM, allowing to learn a joint model of 2D sensor noise in STEM together with an implicit 3D model.
We show, that the combination of these models outperforms both individually, as well as several baselines on synthetic and real data.
We show, that the combination of these models are able to successfully disentangle 3D signal and noise without supervision and outperform at the same time several baselines on synthetic and real data.*


<img src="./Images/QualitativeResults.png" alt="Qualitative Results on Synthetic Data" width="1000"/>


This repository contains all code regarding the synthetic data generation process and the reconstruction, as presented in the paper.

---
## Citation
---
If you find this code useful, please consider citing us: 

     @inproceedings{kniesel2022stem,
       title={Clean Implicit 3D Structure from Noisy 2D STEM Images},
       author={Kniesel, Hannah and Ropinski, Timo and Bergner, Tim and Devan, Kavitha Shaga and Read, Clarissa and Walther, Paul and Ritschel, Tobias and Hermosilla, Pedro},
       booktitle={Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},
       year={2022}
    }

---
## Getting Started 
---
Create new conda environment:

    conda create -n Implicit-Electron-Tomography python=3.9.
    conda activate Implicit-Electron-Tomography
    conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
    conda install --file requirements.txt
    conda install pip
    pip install transformations==2021.6.6

You are ready to go

---
## Reconstructions
---
For closer details on reconstructions, see the README.md in the ./Reconstruction/ directory.

---
## Synthetic Data Generation
---
For closer details on synthetic data generation see the README.md in the ./SyntheticDataGeneration/ directory.

---
## Data for Download
---
You can find all training data and micrographs as presented in the paper here: https://viscom.datasets.uni-ulm.de/Implicit-Electron-Tomography/dataset.zip

Or here: https://cove.thecvf.com/datasets/760

The directory structure is as follows: 

```
_Data                           ...contains micrographs and training data
│   Real_CovidInfectedCell
│   Real_Nanoparticles
│   Synthetic

```
All this data has a size of approximately 5GB.


You can find all pretrained-models and reconstructions as presented in the paper here: https://viscom.datasets.uni-ulm.de/Implicit-Electron-Tomography/models_and_results.zip 

The directory structure is as follows: 
```
ModelsAndResults  
│
└───Reconstruction_Data
│   │
│   └───_PretrainedModels               ...contains pretrained models 
│   │   │   NoiseModel
│   │   │   Real_CovidInfectedCell
│   │   │   Synthetic 
│   └───_Reconstructions                ...contains results of pretrained models
│   │   │   Real_CovidInfectedCell      
|   |   |   Synthetic
│   
└───SyntheticDataGen_Data
    │   Cell                            ...contains synthetic phantom volume
    │   NoiseModel                      ...contains pretrained noise model 
    │   VirusPDB                        ...contains virus density map (PDB 6mid)
```


All this data has a size of approximately 15GB.


Please unpack all data of the "Reconstruction_Data" and "_Data" to the "Reconstruction" directory. 
And please unpack all data of the "SyntheticDataGen_Data" to the "SyntheticDataGeneration" directory.
