## **[The 5th Social Media Mining for Health Applications](https://healthlanguageprocessing.org/smm4h-sharedtask-2020/)** Workshop, COLING 2020.

### **Task 5**: **Automatic classification of tweets reporting a birth defect pregnancy outcome**

This new, multi-class classification task involves distinguishing three classes of tweets that mention birth defects: “defect” tweets refer to the user’s child and indicate that he/she has the birth defect mentioned in the tweet (annotated as “1”); “possible defect” tweets are ambiguous about whether someone is the user’s child and/or has the birth defect mentioned in the tweet (annotated as “2”); “non-defect” tweets merely mention birth defects (annotated as “3”).

* Training data: 18,397 tweets (953 “defect” tweets; 956 “possible defect” tweets; 16,488 “non-defect” tweets)
* Test data: 4,602 tweets
* Evaluation metric: micro-averaged F1-score for the “defect” and “possible defect” classes

|                | Training      | Validation    | test  |
| -------------- | ------------- |:-------------:| -----:|
|Max tweet Length|62| 61 |  77  |
|Max Hashtags    |22| 13      |   23  |
|Avg Hashtags    |2.1| 2.1      |  2.2  |
|tweets with atleast one Hashtag|4277|  1086     | 1303  |
|Number of Tweets|14717| 3680 |   4372 |

**Note** : Some of the lines of test data are corrupted due to which length is more than 280 for those we assigned length as 0.

## Proposed Architectures

### Single View
![Single-View](https://github.com/Saichethan/SMM4H/blob/master/images/Single%20View.png)

### Two View
![Two-View](https://github.com/Saichethan/SMM4H/blob/master/images/Two%20View.png)



## Results

Results on Validation data are given below

|                | P      | R   | F1  |
| -------------- | ------------- |:-------------:| -----:|
|Single View|0.52| 0.63 |  0.57  |
|Two View|0.50| 0.78      |   0.61  |


## Run

```
$ python3 preprocess.py
# for data statistics and embeddings, invoked automatically when TwoView/SingleView.py executed 

$ python3 TwoView.py # for two view

$ python3 TwoView.py # for single view, use dense1 instead of dense

$ python3 SMM4H2020Task5_EvaluationScript.py $input $output

```

## Citation

```
@inproceedings{reddy-2020-detecting,
    title = "Detecting Tweets Reporting Birth Defect Pregnancy Outcome Using Two-View {CNN} {RNN} Based Architecture",
    author = "Reddy, Saichethan",
    booktitle = "Proceedings of the Fifth Social Media Mining for Health Applications Workshop {\&} Shared Task",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.smm4h-1.21",
    pages = "125--127",
    abstract = "This research work addresses a new multi-class classification task (fifth task) provided at the fifth Social Media Mining for Health Applications (SMM4H) workshop. This automatic tweet classification task involves distinguishing three classes of tweets that mention birth defects. We propose a novel two view based CNN-BiGRU based architectures for this task. Experimental evaluation of our proposed architecture over the validation set gives encouraging results as it improves by approximately 7{\%} over our single view model for the fifth task. Code of our proposed framework is made available on Github",
}
```
