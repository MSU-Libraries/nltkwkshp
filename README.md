
# NLTK Workshop 
Updated March, 2016

Introductory code to practice using Python's Natural Language Toolkit (NLTK), much of which is taken from the excellent [NLTK Book](http://www.nltk.org/book/). 

Begin by importing NLTK along with all the resources used in the NLTK book -- this second part assumes that you have already downloaded the book resources. If you haven't, first enter `nltk.download()` and select the "book" resources for download.


```python
import nltk
from nltk.book import *
```

## Basic Text Data


```python
text1
```


```python
len(text1)
```


```python
len(set(text1))
```


```python
from __future__ import division
len(text1) / len(set(text1))
```


```python
text1.tokens[:25]
```


```python
text1.generate()
```

## Collocations
Find "collocations", that is, word combinations that occur more often than would be expected by chance.


```python
text1.collocations()
```

A more involved approach allows users to make use of additional functionality.


```python
from nltk.collocations import *
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
finder.nbest(trigram_measures.pmi, 10)
```


```python
finder.apply_freq_filter(3)
finder.nbest(trigram_measures.pmi, 10)
```

## Concordances
Count words and see them in context.


```python
text1.concordance("whale")
```


```python
text1.count("monster")
```


```python
text1.similar("monstrous")
```


```python
text1.count("whale") / len(text1) * 100
```


```python
fdist = FreqDist(text1)
```


```python
fdist
```


```python
fdist.items()[:50]
```


```python
fdist.most_common(50)
```

### Filtering Word Lists


```python
all_words = [w.lower() for w in text1 if w.isalpha()]
fdist_words = FreqDist(all_words)
```


```python
fdist_words.items()[:50]
```


```python
from nltk.corpus import stopwords
filtered_words = [w for w in lowercase_words if w not in stopwords.words('english')]
fdist_filtered_words = FreqDist(filtered_words)
fdist_filtered_words.items()[:50]
```


```python
[word for word in set(filtered_words) if len(word) > 15]
```


```python
import re
[(word, filtered_words.count(word)) for word in set(filtered_words) if re.search('^un.*ly$', word)]
```

## Classifying Text

Assigning text to categories algorithmically.


```python
nltk.pos_tag(sent2)
```

[Reference](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) for part of speech tags.


```python
nltk.corpus.brown.tagged_paras()
```

Supervised machine learning for gender classification of names.


```python
def gender_features(word):
    return {'last_letter': word[-1]}
```


```python
from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
```


```python
labeled_names
```


```python
import random
random.shuffle(labeled_names)
```


```python
labeled_names
```


```python
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
```


```python
classifier.classify(gender_features('Neo'))
```


```python
classifier.classify(gender_features('Trinity'))
```


```python
nltk.classify.accuracy(classifier, test_set)
```


```python
classifier.show_most_informative_features(5)
```


```python
def gender_features(word):
    return {'last_letter': word[-1]}
```


```python
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
```


```python
classifier.show_most_informative_features(5)
```
