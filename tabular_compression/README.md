# PAC-Bayes bounds and label compression via neural networks

This folder contains the code to produce the PAC-Bayes bounds
for the tabular and image data, as well as the information necessary for the label compression.

This folder requires the `timm` repo to be installed, as well as
[`olive-oil-ml`](https://github.com/mfinzi/olive-oil-ml) for experiment management and [`pactl`](https://github.com/activatedgeek/tight-pac-bayes) for compression based PAC-Bayes bound calculation.

The file `label_compression_image.py` contains the code for compressing each of the image datasets using a small CNN,
`label_compression_image.py` contains the code for compressing the tabular datasets using an MLP
and `pac_bounds_tabular.py` can be run to produce the PAC-Bayes bounds for compressing tabular data with a convolutional network.

The scripts can be run as follows, but the hyperparameters can be overriden with keyword arguments. These keyword arguments can be found by running e.g. 
`python pac_bounds_tabular.py --help`.

You can run them without any arguments to produce the dataframes which are used in the plotting notebooks.

```bash
python3 label_compression_image.py
python3 label_compression_image.py
python3 pac_bounds_tabular.py
```

