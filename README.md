# Multimodal Speech Emotion Recognition

Survey paper: https://bit.ly/3OBITDP

Currently testing TIM-Net: Official Tensorflow implementation of ICASSP 2023 paper, "Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition". [[paper]](https://arxiv.org/abs/2211.08233) [[code]](https://github.com/Jiaxin-Ye/TIM-Net_SER) 

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:
- `pyenv` is installed on your system. Visit [pyenv](https://github.com/pyenv/pyenv#installation) for installation instructions.
- Python 3.8.18 is required for this project.

### Installation

Follow these steps to get your development env running:

1. **Install Python 3.8.18 using pyenv**

```
pyenv install 3.8.18
pyenv local 3.8.18
```
 
2. **Create and activate a Python virtual environment**

 ```
 python -m venv venv
 source venv/bin/activate
 ```

3. **Upgrade pip, setuptools and wheel**

 ```
 pip install --upgrade pip setuptools wheel
 ```

4. **Install project dependencies**

Note that this project has been modified to ensure compatibility with MacOS. For non-MacOS environments, try removing tensorflow-macos and tensorflow-metal from requirements.txt.

``` 
pip install -r requirements.txt
```

### Load the data and model weights

Download the data and model weights here: 

You can directly move the folders into the root project directory.

The folder structure should be as follows: 

- MFCC
- Models
- main.ipynb
- requirements.txt
- README.md

### Testing the model

Test the model using the following command: 
```
python main.py --mode test --data IEMOCAP  --test_path ./Test_Models/IEMOCAP_16 --split_fold 5 --random_seed 16
```