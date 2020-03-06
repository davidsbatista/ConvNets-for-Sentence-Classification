# Sentence Classification with Convolutional Neural Networks

Code for blog post:
- http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/

I only did experiments on 2 datasets:

-  TREC (http://cogcomp.cs.illinois.edu/Data/QA/QC/)
-  Stanford Sentiment Treebank (https://nlp.stanford.edu/sentiment/)

Other considerations:

- I used GloVe vectors of dimensionality 50
- Experiments with the TREC dataset only ran for 10 epochs 


## Results (accuracy)

| Model            | TREC       |  SST-1  |
| -----------------|:----------:| -------:|
| CNN-rand         | 0.918      |         |
| CNN-static       | 0.896      |         |
| CNN-non-static   | 0.919      |         |
| CNN-multichannel | 0.921      |         |


## Results (precision/recall/f1)

### TREC

- CNN-rand

             precision    recall  f1-score   support

        ABBR       1.00      0.22      0.36         9
        DESC       0.74      0.22      0.34       138
        ENTY       0.55      0.88      0.68        94
         HUM       0.84      0.86      0.85        65
         LOC       0.91      0.93      0.92        81
         NUM       0.61      0.84      0.70       113

- CNN-static

          precision    recall  f1-score   support

        ABBR       1.00      0.56      0.71         9
        DESC       0.57      0.09      0.15       138
        ENTY       0.32      0.89      0.47        94
         HUM       0.94      0.69      0.80        65
         LOC       0.75      0.73      0.74        81
         NUM       0.98      0.71      0.82       113

- CNN-non-static

         precision    recall  f1-score   support

        ABBR       0.75      0.67      0.71         9
        DESC       0.76      0.22      0.35       138
        ENTY       0.39      0.82      0.52        94
         HUM       0.94      0.77      0.85        65
         LOC       0.74      0.90      0.82        81
         NUM       0.96      0.85      0.90       113


- CNN-multichannel

               precision    recall  f1-score   support

        ABBR       0.88      0.78      0.82         9
        DESC       0.64      0.26      0.37       138
        ENTY       0.38      0.79      0.51        94
         HUM       0.84      0.83      0.84        65
         LOC       0.87      0.89      0.88        81
         NUM       1.00      0.81      0.90       113

---

### T
