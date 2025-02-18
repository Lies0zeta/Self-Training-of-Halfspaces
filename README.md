# Self-Training-of-Halfspaces

This is the official repository for the IJCAI 2023 paper:

**Generalization Guarantees of Self-Training of Halfspaces under Label Noise Corruption**

[Lies Hadjadj](https://orcid.org/0000-0002-7926-656X), [Massih-Reza Amini](http://ama.liglab.fr/~amini/), and [Sana Louhichi](https://cv.hal.science/sana-louhichi?langChosen=fr)

*Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23)*

[[Paper]](https://doi.org/10.24963/ijcai.2023/420)

## Abstract
We investigate the generalization properties of a self-training algorithm with halfspaces. The approach learns a list of halfspaces iteratively from labeled and unlabeled training data, in which each iteration consists of two steps: exploration and pruning. In the exploration phase, the halfspace is found sequentially by maximizing the unsigned-margin among unlabeled examples and then assigning pseudo-labels to those that have a distance higher than the current threshold. These pseudo-labels are allegedly corrupted by noise.

The training set is then augmented with noisy pseudo-labeled examples, and a new classifier is trained. This process is repeated until no more unlabeled examples remain for pseudo-labeling. In the pruning phase, pseudo-labeled samples that have a distance to the last halfspace greater than the associated unsigned-margin are then discarded. We prove that the misclassification error of the resulting sequence of classifiers is bounded and show that the resulting semi-supervised approach never degrades performance compared to the classifier learned using only the initial labeled training set. Experiments carried out on a variety of benchmarks demonstrate the efficiency of the proposed approach compared to state-of-the-art methods.

## Key Features

- Novel self-training algorithm for halfspaces with theoretical guarantees
- Two-phase approach: exploration and pruning
- Robust to label noise corruption
- Provable bounded misclassification error
- Non-degradation guarantee compared to supervised learning

## Datasets

We evaluated our method on several benchmark datasets:

- 20-NEWSPAPER
- BANKNOTE
- SPAMBASE
- DELICIOUS
- MEDIAMILL

## Baselines

Our method demonstrates superior performance compared to:
- Label Propagation
- Gaussian Naive Bayes
- Entropy Regularized Logistic Regression

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{ijcai2023p420,
    title     = {Generalization Guarantees of Self-Training of Halfspaces under Label Noise Corruption},
    author    = {Hadjadj, Lies and Amini, Massih-Reza and Louhichi, Sana},
    booktitle = {Proceedings of the Thirty-Second International Joint Conference on
                 Artificial Intelligence, {IJCAI-23}},
    publisher = {International Joint Conferences on Artificial Intelligence Organization},
    editor    = {Edith Elkind},
    pages     = {3777--3785},
    year      = {2023},
    month     = {8},
    note      = {Main Track},
    doi       = {10.24963/ijcai.2023/420},
    url       = {https://doi.org/10.24963/ijcai.2023/420}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue or contact:
- Lies Hadjadj (lies.hadjadj@univ-grenoble-alpes.fr)
