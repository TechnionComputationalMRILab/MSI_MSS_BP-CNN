# Exploring the Interplay Between Colorectal Cancer Subtypes Genomic Variants and Cellular Morphology: A Deep-Learning Approach
---
Hadar Hezi,
Daniel Shats,
Daniel Gurevich,
Yosef E. Maruvka,
Moti Freiman
---
## Abstract
Molecular subtypes of colorectal cancer (CRC) significantly influence treatment
decisions. While convolutional neural networks (CNNs) have recently been introduced
for automated CRC subtype identification using H&E stained histopathological images,
the correlation between CRC subtype genomic variants and their corresponding cellular
morphology expressed by their imaging phenotypes is yet to be fully explored. The goal
of this study was to determine such correlations by incorporating genomic variants in
CNN models for CRC subtype classification from H&E images. We utilized the publicly
available TCGA-CRC-DX dataset, which comprises whole slide images from 360
CRC-diagnosed patients (260 for training and 100 for testing). This dataset also
provides information on CRC subtype classifications and genomic variations. We
trained CNN models for CRC subtype classification that account for potential
correlation between genomic variations within CRC subtypes and their corresponding
cellular morphology patterns. We assessed the interplay between CRC subtypes’
genomic variations and cellular morphology patterns by evaluating the CRC subtype
classification accuracy of the different models in a stratified 5-fold cross-validation
experimental setup using the area under the ROC curve (AUROC) and average
precision (AP) as the performance metrics. The CNN models that account for potential
correlation between genomic variations within CRC subtypes and their cellular
morphology pattern achieved superior accuracy compared to the baseline CNN
classification model that does not account for genomic variations when using either
single-nucleotide-polymorphism (SNP) molecular features (AUROC: 0.824±0.02 vs.
0.761±0.04, p<0.05, AP: 0.652±0.06 vs. 0.58±0.08) or CpG-Island methylation
phenotype (CIMP) molecular features (AUROC: 0.834±0.01 vs. 0.787±0.03, p<0.05,
AP: 0.687±0.02 vs. 0.64±0.05). Combining the CNN models account for variations in
CIMP and SNP further improved classification accuracy (AUROC: 0.847±0.01 vs.
0.787±0.03, p=0.01, AP: 0.68±0.02 vs. 0.64±0.05). The improved accuracy of CNN
models for CRC subtype classification that account for potential correlation between
genomic variations within CRC subtypes and their corresponding cellular morphology
as expressed by H&E imaging phenotypes may elucidate the biological cues impacting
cancer histopathological imaging phenotypes. Moreover, considering CRC subtypes
genomic variations has the potential to improve the accuracy of deep-learning models in
discerning cancer subtype from histopathological imaging data.

## Requirements
1. Python $\geq$ 3.6
2. GPU
3. pyTorch $\geq$ 1.9
   
## User Guide
1. Download the TCGA CRC [data](https://doi.org/10.5281/zenodo.3832231)
2. Clone this repository
3. Train and test a new model using run_model.py
4. To load this paper's results - load the relevant roc_out files using Pickle. The file contains the patients' MSI probabilities and their labels. It is organized as a dictionary with the keys 'labels' and 'probs'. 
