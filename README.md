# Generation of OOD Data for Damage Detection in Buildings
This is the main repository for the the master thesis "Generation of OOD Data for Damage Detection in Buildings".
It contains the code for training the networks mentioned in the thesis and saving the results.

<img align="middle" alt="Python" width="800px" src="etc/GANandVGG.svg" />

# Usage
1. Install dependencies `pip install -r requirements.txt`
2. Save the desired training datasets following the PyTorch dataloader convention (<a href = "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html" >Link</a>).
2. Run `train_vgg13.py`/`train_vgg13gan.py` for training the network classifiers.

Currently the classifier model used for the thesis is VGG13. We encourage further architectures to be explored.
