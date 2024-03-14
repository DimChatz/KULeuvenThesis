# KULeuvenThesis
My thesis for the Advanced Master in Artificial Intelligence from KU Leuven

---

# Versioning
## V-0.0.1 
### 27/2 - 1/3
Tried to make foundation work -> failed, some functions missing
Created berts dataset preprocessing + simple model -> failed, wrong data former for torch

## V-0.1.1 
### 2/3 - 4/3
Debugging -> made simple model work -> overfitting af
added weigths -> still overfitting af

## V-0.1.2 
### 5/3 - 6/3
Added callbacks for early stopping and reducing LR
Added training visualizer for loss and accuracy
Added F1 score
Fixed Model training and preprocessing
create more complex model -> is shit af -> not same as paper
Issue with cuda -> Fixed

## V-1-1-1 
### 7/3
Fixed Cuda
Recleared dataset


## V-2-1-1 
### 11/3
Added Gaussian per channel
Added PTB-XL preprocessing
Fixed CNN model of paper with calcs and padding

### 12/3 
checked for duplicate patients in berts data
fixed ptb pipeline and visualization
fixed all train-test pipeline

### 13/3
Run ptb tests and finished its correct preproc

### 14/3
overhauled Berts pipeline

---

## To do
- Preproc Berts
- Add visualization of preproc
- Add visualization of training
- Add F1 per class
- Create CNN
○ Check other models
○ Handle Foundation
- Add accuracy
- Add F1
- Add LR reduction callback 
- Add early stoppying callbacks
- Add ptbxl data
○ Add cross-validation
○ Add experiment tracker
○ Add model weight loader
- check missing data of Bert
- check duplicate patients for Bert
○ Add regularization
- fix gaussian norms to be per channel
○ ablation study
? Add explainable AI
? Add PTBXL all classes
- Add savepaths in visualizations
- Add mean and sigma efficient calculation



