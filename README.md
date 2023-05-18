# Self-Training-of-Halfspaces

This repository contains the implementation of the paper:

**Generalization Guarantees of Self-Training of Halfspaces under Label Noise Corruption**

[Lies Hadjadj](https://orcid.org/0000-0002-7926-656X), [Massih-Reza Amini](http://ama.liglab.fr/~amini/), and [Sana Louhichi](https://cv.hal.science/sana-louhichi?langChosen=fr).



## Abstract
We investigate the generalization properties of a self-training algorithm with halfspaces. The approach learns a list of halfspaces iteratively from labeled and unlabeled training data, in which each iteration consists of two steps: exploration and pruning. In the exploration phase, the halfspace is found sequentially by maximizing the unsigned-margin among  unlabeled examples and then assigning pseudo-labels to those that have a distance higher than the current threshold. These pseudo-labels are allegedly corrupted by noise.
The training set is then augmented with noisy pseudo-labeled examples, and a new classifier is trained.  This process is repeated until no more unlabeled examples remain for pseudo-labeling. In the pruning phase, pseudo-labeled samples that have a distance to the last halfspace greater than the associated  unsigned-margin are then discarded. We prove that the misclassification error of the resulting sequence of classifiers is bounded and show that the resulting semi-supervised approach never degrades performance compared to the  classifier learned using only the initial labeled training set. Experiments carried out on a variety of benchmarks demonstrate the efficiency of the proposed approach compared to state-of-the-art methods.



## Code
Coming soon!
