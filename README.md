## **The 5th Social Media Mining for Health Applications** Workshop, of COLING 2020.

### **Task 5**: **Automatic classification of tweets reporting a birth defect pregnancy outcome**

This new, multi-class classification task involves distinguishing three classes of tweets that mention birth defects: “defect” tweets refer to the user’s child and indicate that he/she has the birth defect mentioned in the tweet (annotated as “1”); “possible defect” tweets are ambiguous about whether someone is the user’s child and/or has the birth defect mentioned in the tweet (annotated as “2”); “non-defect” tweets merely mention birth defects (annotated as “3”).

* Training data: 18,397 tweets (953 “defect” tweets; 956 “possible defect” tweets; 16,488 “non-defect” tweets)
* Test data: 4,602 tweets
* Evaluation metric: micro-averaged F1-score for the “defect” and “possible defect” classes

|                | Training      | Validation    | test  |
| -------------- | ------------- |:-------------:| -----:|
|Max tweet Length|62| 61 |  699  |
|Max Hashtags    |22| 13      |   23  |
|Avg Hashtags    |2.1| 2.1      |  2.2  |
|tweets with atleast one Hashtag|4277|  1086     | 1303  |
|Number of Tweets|14717| 3680 |   4372 |


## Proposed Architectures

### Multi View
![Multi-View](https://github.com/Saichethan/SMM4H/blob/master/images/MV.png)
