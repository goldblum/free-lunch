# The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning

This repository contains PyTorch implementations for all experiments described in [The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning](https://github.com/goldblum/free-lunch) by [Micah Goldblum](https://goldblum.github.io/), [Marc Finzi](https://mfinzi.github.io/), [Keefer Rowan](https://cims.nyu.edu/~kjr9750/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).  

## Abstract

No free lunch theorems for supervised learning state that no learner can solve all problems or that all learners achieve exactly the same accuracy on average over a uniform distribution on learning problems.  Accordingly, these theorems are often referenced in support of the notion that individual problems require specially tailored inductive biases. While virtually all uniformly sampled datasets have high complexity, real-world problems disproportionately generate low-complexity data, and we argue that neural network models share this same preference, formalized using Kolmogorov complexity.  Notably, we show that architectures designed for a particular domain, such as computer vision, can compress datasets on a variety of seemingly unrelated domains. Our experiments show that pre-trained and even randomly initialized language models prefer to generate low-complexity sequences.  Whereas no free lunch theorems seemingly indicate that individual problems require specialized learners, we explain how tasks that often require human intervention such as picking an appropriately sized model when labeled data is scarce or plentiful can be automated into a single learning algorithm.  These observations justify the trend in deep learning of unifying seemingly disparate problems with an increasingly small set of machine learning models.

[![Preview](/tree.png)](https://github.com/goldblum/free-lunch)

## Folders

Since the experiments in our paper are diverse, we organize them into individual folders, described below:  


`compress_data/` - This folder contains code for compressing datasets as an introductory example for bounding Kolmogorov complexity. (Section 3.1)  

`llm_sequences/` - We demonstrate that large larnguage models, including GPT-3 and GPT-2, even ones which are randomly initialized, posess a preference for low-complexity numerical sequences. (Sections 4.3 and 4.4)  

`model_selection/` - This folder contains our experiments with polynomial regression and neural networks, showing that a single learner can simultaneously achieve strong performance on large and small datasets, where single architectures typically perform well on only small or large datasets but not both. (Section 5.2)  

`tabular_compression/` - Real world datasets contain structure aligned with that of neural networks.  This folder contains tools for compressing tabular datasets using MLPs and CNNs themselves as well as computing PAC-Bayes generalization bounds.  (Sections 3.2 and 4.2)

`plotting/` - The notebook in this folder generates all plots from our paper.  

## Requirements
- jupyter 1.0.0
- matplotlib 3.6.3
- numpy 1.24.1
- openai 0.27.2
- pandas 1.5.3
- scikit-learn 1.2.0
- seaborn 0.12.2
- torch 2.0.0
- torchaudio 2.0.1
- torchtext 0.15.1
- torchvision 0.14.1
- olive-oil-ml >=0.0.1
- openml
- timm
- git+https://github.com/activatedgeek/tight-pac-bayes.git

## Repo Structure
We include separate README's in each of the experiment-specific folders
```
├── compress_data  
│   ├── README.md  
│   ├── audio_compression.py  
│   ├── text_compression.py  
│   └── utils.py  
├── llm_sequences  
│   ├── README.md  
│   ├── analysis_long.py  
│   ├── gpt3.py  
│   ├── int_c=6_l=200.pickle  
│   ├── int_c=6_l=30.pickle  
│   ├── int_c=6_l=500.pickle  
│   ├── int_c=7_l=100.pickle  
│   ├── int_c=7_l=30.pickle  
│   ├── mingpt/  
│   ├── saved_dicts/  
│   ├── sequence_generation_initialized.py  
│   ├── sequence_generation_initialized_long.py  
│   ├── sequence_generation_pretrained.py  
│   ├── sequence_generation_pretrained_long.py  
│   ├── trees.py  
│   └── utils.py  
├── model_selection  
│   ├── neural_networks  
│   │   ├── README.md  
│   │   ├── train.py  
│   │   ├── train_tradeoff.py  
│   │   └── utils/  
│   └── polynomial_regression  
│       └── polynomial_regression.py  
├── plotting  
│   ├── README.md  
│   ├── data/  
│   ├── plot_ipynb.ipynb  
│   └── utils.py  
├── tabular_compression  
│   ├── README.md  
│   ├── label_compression_image.py
│   ├── label_compression_tabular.py
│   └── pac_bounds_tabular.py
```
