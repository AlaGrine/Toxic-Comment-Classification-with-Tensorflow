# Toxic Comment Classification

## NLP multi-label classification using Tensorflow

<div align="center">
  <img src="https://github.com/AlaGrine/Toxic-Comment-Classification-with-Tensorflow/blob/main/stats/toxic_comments_img.png" >
</div>

### Table of Contents

1. [Project Overview](#overview)
2. [Problem statement](#prbStatement)
3. [Installation](#installation)
4. [File Descriptions](#file_descriptions)
5. [Evaluation metrics and handling unbalanced dataset](#evaluation)
6. [Modelling](#modelling)
7. [Results](#results)
8. [Improvements](#Improvements)
9. [Acknowledgements](#Acknowledgements)

## Project Overview <a name="overview"></a>

Negative online behaviors, like toxic comments, are likely to make people stop expressing themselves and leave a conversation.

Platforms are struggling to facilitate conversations in an effective way, leading many communities to limit or shut down user comments altogether.

## Problem statement <a name="prbStatement"></a>

We used a [Kaggle dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) containing 159,571 comments from Wikipedia’s talk page edits. These comments were labeled for toxic behaviour by human raters.

Our goal is to build a multilabel classification model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.

## Installation <a name="installation"></a>

This project requires Python 3 and the following Python libraries installed:

`tensorflow` ,`pandas`, `numpy`, `scikit-multilearn`, `wordcloud`, `seaborn`, `matplotlib` and `googletrans`

## File Descriptions <a name="file_descriptions"></a>

The main file of the project is `Toxic-Comment-Classification-with-Tensorflow.ipynb`.

The project folder also contains the following:

- `stats` folder: The history and metrics of our models, including f1-score and AUC, are available here (csv files).
- `googletrans` folder: contains the output of the text data augmentation with [googletrans](https://pypi.org/project/googletrans/) API.

## Evaluation metrics and handling unbalanced dataset <a name="evaluation"></a>

To deal with class imbalance, we:

> 1.  Used `stratified` data split instead of the traditional train_test_split.
> 2.  Built a `custom binary cross-entropy loss function with class weights`. The weights are inversely proportional to the number of samples in each class, so that the minority labels (namely 'threat' and 'identity_hate') are given a greater weight.
> 3.  Performed `text data augmentation` to increase the number of minority labels (namely 'threat' and 'identity_hate'). We used the [googletrans](https://pypi.org/project/googletrans/) API to translate the given comment into French, German and Spanish and then back into English (back-translation).
> 4.  Used the the **F1-score** and **AUC** as evaluation metrics instead of the accuracy.

## Modelling <a name="modelling"></a>

We used `Tensorflow` to build the following model structure:

```
Input (text) -> TextVectorization -> Embedding -> Layers -> Output (sigmoid)
```

where the `Layers` component is either Dense Layer, LSTM, GRU or 1D Convolutional Neural Network.

## Results<a name="results"></a>

<div align="center">
  <img src="https://github.com/AlaGrine/Toxic-Comment-Classification-with-Tensorflow/blob/main/stats/AUC_dense_model_.png" >
</div>

Applying our `custom binary cross-entropy loss function with class weights` and `text data augmentation` (using googletrans API) resulted in a **3%** increase in AUC for the dense model.

Both techniques were used for the other models.

<div align="center">
  <img src="https://github.com/AlaGrine/Toxic-Comment-Classification-with-Tensorflow/blob/main/stats/AUC_per_model_.png" >
</div>

The `GRU` model performed best with an AUC of **97.4%**, **0.1%** ahead of the `LSTM` model.

## Improvements <a name="Improvements"></a>

We have been building our own embedding layer.

We can take advantage of Transfer Learning and use a pre-trained embedding layer, such as the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) or [BERT](https://tfhub.dev/google/collections/bert/1) available on the TensorFlow Hub.

## Acknowledgements <a name="Acknowledgements"></a>

Must give credit to [Kaggle](https://kaggle.com) for the dataset.
