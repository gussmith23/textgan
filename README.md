# TextGAN Implementation

Please view this file with a Markdown viewer, or [browse this repository on GitHub](https://github.com/gussmith23/textgan) to use its built-in Markdown viewer.

This repository contains my term project for CSE 583: Pattern Recognition from Spring 2018 at Penn State. [paper.pdf](paper.pdf) is the report for this project, and should be treated as the primary document explaining this project. Please consult it for technical details.

This README primarily explains
- How the code is structured
- How to run the code

It does not explain the code itself; again, for that, please read the report at [paper.pdf](paper.pdf).

## Dependencies
**Python 3.**

**Tensorflow and its dependencies.** Instructions for installing TensorFlow can be found [here](https://www.tensorflow.org/install/), but essentially boil down to 
```shell
pip3 install --upgrade tensorflow       # For non-GPU version
pip3 install --upgrade tensorflow-gpu   # For GPU version
``` 
If you would like GPU support, you will also need to follow TensorFlow's instructions for installing GPU dependencies (CUDA and CUDNN).

**(optional) matplotlib and scikit-learn.** These are only needed for running embeddings visualization, and can be installed via:
```shell
pip install -U matplotlib
pip install -U scikit-learn
```

## Files and Directories in this Repository

The core files implementing TextGAN training, roughly in the order they are needed, are:
- [**embeddings-skip-gram.py**](embeddings-skip-gram.py) Embeddings generation script.
- [**discriminator.py**](discriminator.py) Discriminator model.
- [**pretrain-discriminator.py**](pretrain-discriminator.py) Discriminator pretraining script.
- [**generator.py**](generator.py) Generator model.
- [**pretrain-generator.py**](pretrain-generator.py) Generator pretraining script.
- [**textgan.py.**](textgan.py) The TextGAN model and training script.

Utility:
- [**generate-and-score-sentences.py.**](generate-and-score-sentences.py) Script for generating sentences from trained TextGAN model. Optionally computes their BLEU score.
- [**embeddings-visualize.py.**](embeddings-visualize.py) Creates t-SNE visualization of generated embeddings.

Third-party scripts:
- [**bleu.py.**](bleu.py) Implementation of the BLEU metric for evaluation. From TensorFlow's [neural machine translation project](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py).
- [**mmd.py.**](mmd.py) MMD implementation [by Dougal Sutherland](https://github.com/dougalsutherland/opt-mmd/blob/master/gan/mmd.py).
- [**tf_ops.py.**](tf_ops.py) Dependency for mmd.py.

Directories:
- [**data.**](data/README.md) The directory containing our datasets and dataset-getting modules. Please read that directory's README for more information.
- [**saved-checkpoints.**](saved-checkpoints) Contains important checkpoints saved during the training of our model. These checkpoints are used in the demonstrations below.
- [**results.**](results) Contains results used in the paper.

Unused files:
- [**embeddings-cbow.py**](embeddings-cbow.py) Continuous Bag of Words embeddings generation script. Ended up using only the Skip-gram model. However, I keep the CBoW model, as it is allegedly better for smaller datasets. _Warning: it likely will not work in its current state._
