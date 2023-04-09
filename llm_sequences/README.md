# Low-Complexity Bias in Language Models

The code in this folder conducts all language model experiments from the associated paper, both using randomly initialized [GPT-2](https://openai.com/research/better-language-models) models as well as [GPT-3](https://arxiv.org/abs/2005.14165).  GPT-2 code is adapted from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) repo.

Below is a guide to the organization and content of this directory.  The plots from our paper can be generated using `../plotting` directory.

## Expression Tree Sequences and GPT-3
We perform experiments where we evaluate the probabilities assigned by GPT-3 to numerical sequences generated using short expression trees.  
`gpt3.py` - evaluates the probabilities assigned to numerical sequences by GPT-3.  
`trees.py` - generates the numerical sequences.  
The five pickle files in this directory contain such numerical sequences and are automatically loaded by `trees.py`.

## T-Tests
We use the below scripts to conduct t-tests on long sequences showing that GPT-2, either randomly initialized or pre-trained, prefers to generate lower complexity sequences than a uniform prior.  
`sequence_generation_initialized_long.py` - generates numerical sequences with randomly initialized GPT-2 models.  
`sequence_generation_pretrained_long.py` - generates numerical sequences with randomly trained GPT-2 models.  
`analysis_long.py` - performs the associated t-tests.

## Computing Probabilities of Repeated Sequences According to GPT-2
`sequence_generation_initialized.py` - generates numerical sequences which are used to estimate their probabilities under randomly initialized GPT-2 models.  
`sequence_generation_pretrained.py` - evaluates the probabilities assigned to numerical sequences by GPT-2.