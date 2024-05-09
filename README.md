# Multimodal Speech Emotion Recognition

This project explores multimodal emotion recognition using the IEMOCAP dataset, exploring the interplay between multimodal features, as well as different model architectures, to enhance accuracy and robustness. Employing the TIM-Net (Temporal-aware bi-directional Multi-scale Network) as the baseline model, additional exploration entails diverse model architectures to discern their effectiveness across various modalities. Subsequently, a novel multimodal model is devised, streamlining individual modality processing through optimised pathways prior to implementing cross-modal attention mechanisms. The multimodal model demonstrates promising performance on the IEMOCAP dataset, achieving an F1 score of 0.729 and accuracy of 0.730, underscoring its potential in multimodal emotion recognition tasks.

Survey paper: https://bit.ly/3OBITDP

Report: https://bit.ly/multimodal-emotion-recognition

## Dataset

I used a multimodal dataset preprocessed by Poria et. al.: https://github.com/soujanyaporia/multimodal-sentiment-analysis/tree/master/dataset

It comprises 120 samples, with 110 timesteps per sample, and textual, audio and visual features processed using CNN, 3D-CNN and openSMILE respectively. 

To streamline the dataset, I merged 'happy' and 'excited' labels, as well as 'angry' and 'frustrated' labels. This resulted in a more imbalanced dataset, with the ‘Angry' class comprising 41.3% of the dataset:

<img width="614" alt="Screenshot 2024-05-09 at 10 50 11 AM" src="https://github.com/vennietweek/multimodal-speech-emotion-recognition/assets/19652161/7b3315c1-4697-4eef-b425-eb4af6de75bb">

## Model Architecture

The architecture of the Multimodal Model uses distinct processing pathways for each modality, before synthesising the information to make predictions:

![Multimodal Model Architecture drawio (2)](https://github.com/vennietweek/multimodal-speech-emotion-recognition/assets/19652161/dfb07924-5a9f-42cf-85d1-aea0ab52cc62)

Text data is processed using a dual-layer bidirectional Long Short-Term Memory (Bi-LSTM). Audio data is processed using Transformer blocks, each consisting of a self-attention mechanism with four heads and a deep feed-forward network with a dimensionality of 2048. Visual data processing is handled by a combination of 1D convolutional neural network (CNN) layers and a multi-head attention mechanism. Cross-modal attention layers are used, specifically between the text and the other two modalities, audio and visual. These layers are designed emphasize the importance of textual data, which was shown to be more informative and robust across different model architectures during the exploratory stage. 

## Evaluation

The multimodal model achieved an accuracy of 0.7307 and an F1 score of 0.7292. 

An examination of the confusion matrix reveals specific trends in class misclassification:

<img width="911" alt="Screenshot 2024-05-09 at 10 48 01 AM" src="https://github.com/vennietweek/multimodal-speech-emotion-recognition/assets/19652161/8afa8f13-f4e4-4322-afde-3095a9417047">

Emotions such as 'Happy' and 'Sad' are well-classified, indicating the model's robustness in recognizing these distinct emotional states. However, a considerable amount of class confusion is observed with the 'Neutral' emotion, where various true labels were misclassified as 'Neutral.' Furthermore, for true labels classified as 'Neutral,' the predominant misclassifications were into the 'Anger' category, reflecting the dataset imbalance.

## Running the Model

1. **System prerequisites**
- `pyenv` is installed. Visit [pyenv](https://github.com/pyenv/pyenv#installation) for installation instructions.
- Python 3.8.18

2. **Install Python 3.8.18 using pyenv**

```
pyenv install 3.8.18
pyenv local 3.8.18
```
 
3. **Create and activate a Python virtual environment**

 ```
 python -m venv venv
 source venv/bin/activate
 ```

4. **Upgrade pip, setuptools and wheel**

 ```
 pip install --upgrade pip setuptools wheel
 ```

5. **Install project dependencies**

Note that this project has been modified to ensure compatibility with MacOS. For non-MacOS environments, try removing tensorflow-macos and tensorflow-metal from requirements.txt.

``` 
pip install -r requirements.txt
```

## Project Files

#### Workbooks
The workbooks consist of the following:
1. **1_timnet**: Initial implementation of TIM-Net baseline model, with multimodal data.
2. **2_exploratory_analysis**: Exploration of early VS late fusion approaches, as well as different model architctures on each modality.
3. **3_multimodal_model**: Design and implementation of a multimodal model.

#### Other files
1. **dataset**: Pre-processed IEMOCAP data
2. **Archive**: Previous notebooks and experimentations (for assignment, interim report etc.)
3. **data_utils**: Data utilities such as data load, one-hot encoding, and data preparation.
4. **model_utils**: Model utilities such as Transformer Blocks and Cross-Modal Attention layers.
