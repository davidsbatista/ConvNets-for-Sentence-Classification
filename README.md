## Sentence Classification with Convolutional Neural Networks

This repository contains experiments done with Keras implementations of the Convolutional Neural
Networks to perform sentence classification based on the paper of [Kim (2014)](https://www.aclweb.org/anthology/D14-1181.pdf)  

I also wrote a blog post about it, you can read it here:
- http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/

I did experiments on 2 datasets:

-  TREC (http://cogcomp.cs.illinois.edu/Data/QA/QC/)
-  Stanford Sentiment Treebank (https://nlp.stanford.edu/sentiment/) (fine-grained-labels)

Some considerations:

- I used GloVe vectors of dimensionality 50
- The random embeddings have a dimensionality of 300
- I ran the training for 10 epochs 


# Results

## Accuracy

| Model                                               | TREC       |  SST-1  |
| ----------------------------------------------------|:----------:| -------:|
| CNN w/ random word embeddings                       | 0.918      |  0.432  |
| CNN w/ pre-trained static word embeddings           | 0.896      |  0.484  |
| CNN w/ pre-trained dynamic word embeddings          | 0.919      |  0.432  |
| CNN multichannel dynamic and static word embeddings | 0.921      |  0.406  |


The results are in line with the ones reported in the paper, a bit lower, but within the expected.
Since this is a multi-label dataset I also report the results per class.

## Precision/Recall/F1

### TREC

 __CNN w/ random word embeddings__

             precision    recall  f1-score   support

        ABBR       1.00      0.22      0.36         9
        DESC       0.74      0.22      0.34       138
        ENTY       0.55      0.88      0.68        94
         HUM       0.84      0.86      0.85        65
         LOC       0.91      0.93      0.92        81
         NUM       0.61      0.84      0.70       113

__CNN w/ pre-trained static word embeddings__

          precision    recall  f1-score   support

        ABBR       1.00      0.56      0.71         9
        DESC       0.57      0.09      0.15       138
        ENTY       0.32      0.89      0.47        94
         HUM       0.94      0.69      0.80        65
         LOC       0.75      0.73      0.74        81
         NUM       0.98      0.71      0.82       113

__CNN w/ pre-trained dynamic word embeddings__

         precision    recall  f1-score   support

        ABBR       0.75      0.67      0.71         9
        DESC       0.76      0.22      0.35       138
        ENTY       0.39      0.82      0.52        94
         HUM       0.94      0.77      0.85        65
         LOC       0.74      0.90      0.82        81
         NUM       0.96      0.85      0.90       113


__CNN multichannel dynamic and static word embeddings__

               precision    recall  f1-score   support

        ABBR       0.88      0.78      0.82         9
        DESC       0.64      0.26      0.37       138
        ENTY       0.38      0.79      0.51        94
         HUM       0.84      0.83      0.84        65
         LOC       0.87      0.89      0.88        81
         NUM       1.00      0.81      0.90       113

---


### SST-1

 __CNN w/ random word embeddings__

               precision    recall  f1-score   support

     negative       0.22      0.19      0.20       405
      neutral       0.53      0.72      0.61      1155
     positive       0.19      0.11      0.14       424
very positive       0.07      0.02      0.03       112
very_negative       0.05      0.01      0.01       114


__CNN w/ pre-trained static word embeddings__

               precision    recall  f1-score   support

     negative       0.13      0.03      0.05       405
      neutral       0.52      0.91      0.66      1155
     positive       0.13      0.03      0.05       424
very positive       0.00      0.00      0.00       112
very_negative       0.00      0.00      0.00       114


__CNN w/ pre-trained dynamic word embeddings__

               precision    recall  f1-score   support

     negative       0.15      0.10      0.12       405
      neutral       0.52      0.76      0.61      1155
     positive       0.18      0.08      0.11       424
very positive       0.12      0.03      0.04       112
very_negative       0.04      0.01      0.01       114


__CNN multichannel dynamic and static word embeddings__

               precision    recall  f1-score   support

     negative       0.16      0.11      0.13       405
      neutral       0.52      0.65      0.58      1155
     positive       0.23      0.22      0.22       424
very positive       0.02      0.01      0.01       112
very_negative       0.12      0.03      0.04       114

