# Data Compression

## Run:
The below lines compress LibriSpeech, LibriSpeech (shuffled), and Amazon Review Full as in Section 3.1 of our paper.  We employ these experiments in our introductory demonstration on how to upper bound Kolmogorov complexity.
```bash
python3 audio_compression.py --data_root ./data
python3 audio_compression.py --data_root ./data --shuffle 
python3 text_compression.py --data_root ./data 
```