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
Created plot for berts data to check weird sigma and mean
Trained pretrained model

## V-3-1-1
### 18/3 - 22/3
added tracking
addded regularization
found issues with gaussian norm and fixed
did data stats 
added confusion matrix
checked a bit literature about statistics
added checkpoints in data for possible ablation
changed preproc PTB for all type of exps


# V 4-0-1
### 24/3 - 31/3
**DID**
added working images and data for wnb
trained on rhythm
added data table for data stats


# 5-1-1
### 15-22/4
added gaussian augmentations
overhauled pipeline
fixed learning rate reset bug
fixed loss visualization
created 1 more CNN, and 1 transformer and LSTM

# 5-2-1
### 22-29/4
added swin transformer
added test for attia paper
---

## To do
- Preproc Berts
- Add visualization of preproc
- Add visualization of training
- Add F1 per class
- Create CNN
- Check other models
- Add accuracy
- Add F1
- Add LR reduction callback 
- Add early stoppying callbacks
- Add ptbxl data
- Add cross-validation
- Add experiment tracker
- Add model weight loader
- check missing data of Bert
- check duplicate patients for Bert
- Add regularization
- fix gaussian norms to be per channel </br>
○ ablation study
? Add explainable AI
- Add PTBXL rhythm classes
- Add savepaths in visualizations
- Add mean and sigma efficient calculation
- Add Confusion matrix
- Added Gaussian scaling
- Added Gaussian Noise
- fix learning rate
- fix loss visualization







