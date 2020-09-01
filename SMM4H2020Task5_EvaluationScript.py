#!/usr/bin/env python
import sys
import logging as log
import os.path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# as per the metadata file, input and output directories are the arguments
[_, input_dir, output_dir] = sys.argv
#output_dir = 'C:\\Users\\iflores\\Documents\\Projects\\SMM4H2020\\Datasets_Eval_Scripts\\EvalScripts\\SMM4H20scoringTask5'
#output_dir = '/tmp/'
#input_dir=''

log.warning("Start scoring subtask 5")
print("Start scoring subtask 5")

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

#df = pd.read_csv(os.path.join(input_dir, 'ref', 'task5_validation.tsv'))
dtruth = pd.read_csv(os.path.join(input_dir, 'ref', 'task5_validation.tsv'), sep='\t', encoding= 'unicode_escape')
#dtruth = pd.read_csv('C:\\Users\\iflores\\Documents\\Projects\\SMM4H2020\\Datasets_Eval_Scripts\\ValidationFiles\\SMM4H20ValidationST5\\task5_validation.tsv', sep='\t')
assert 'tweet_id' in dtruth.columns, "I was expecting the column tweet_id to be in the tsv file of the test set for task 5."
assert 'label' in dtruth.columns, "I was expecting the column label to be in the tsv file of the test set for task 5."

dtruth.set_index('tweet_id', inplace=True)

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

submission_dir = os.path.join(input_dir, 'res')
submission_path = os.path.join(submission_dir, 'prediction_task5.tsv')
#submission_path = 'C:\\Users\\iflores\\Documents\\Projects\\SMM4H2020\\Datasets_Eval_Scripts\\EvalScripts\\SMM4H20scoringTask5\\prediction_task5.tsv'
if not os.path.exists(submission_path):
    log.fatal('Expected submission named prediction_task5.tsv, found files: ' + submission_path)
    raise Exception('Expected submission named prediction_task5.tsv, found files: ' + submission_path)

dpred = pd.read_csv(submission_path, sep='\t', encoding= 'unicode_escape')
assert 'tweet_id' in dpred.columns, "I was expecting the column tweet_id to be in the tsv file of the test set for task 5, it was not found."
assert 'Class' in dpred.columns, "I was expecting the column Class containing the predictions of the classifier to evaluate to be in the tsv file submitted for task 5, it was not found."

dpred.set_index('tweet_id', inplace=True)

assert len(dtruth) == len(dpred), "The number of tweets predicted " + str(len(dpred)) + " is not equal to the number of tweets annotated in the test set " + str(len(dtruth))
assert dtruth.index.equals(dpred.index), "The tweet IDs in the test set do not correspond to the tweet IDs in the set of tweets predicted"

dEval = pd.concat([dtruth, dpred], axis=1, join='inner')
dEval.to_csv('/tmp/out.tsv', sep='\t')
#dEval.to_csv('C:\\Users\\iflores\\Documents\\Projects\\SMM4H2020\\Datasets_Eval_Scripts\\EvalScripts\\SMM4H20scoringTask5\\out.tsv', sep='\t')
tp = 0.
fp = 0.
fn = 0.
tn = 0.
for index, row in dEval.iterrows():
    #print(str(row["label"]), str(row["Class"]))
    if (str(row["label"]) == '1' and str(row["Class"]) == '1') or (str(row["label"]) == '2' and str(row["Class"]) == '2'):
        tp += 1.
    if (str(row["label"]) == '1' and not str(row["Class"]) == '1') or (str(row["label"]) == '2' and not str(row["Class"]) == '2'):
        fp += 1.
    if (str(row["label"]) == '3' and not str(row["Class"]) == '3') or (str(row["label"]) == '2' and str(row["Class"]) == '1') or (str(row["label"]) == '1' and str(row["Class"]) == '2'):
        fn += 1.
    if str(row["label"]) == '3' and str(row["Class"]) == '3':
        tn += 1.
#print(f"tp: {tp}, fp: {fp}, fn: {fn}, tn:{tn}")
prec = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = (2*prec*rec)/(prec+rec)
#print(f"prec: {prec}, rec: {rec}, f1: {f1}")

cf = confusion_matrix(list(dEval['label']), list(dEval['Class']))
cr = classification_report(list(dEval['label']), list(dEval['Class']),digits=4)
#prec = precision_score(list(dEval['label']), list(dEval['Class']), pos_label=1, average='micro')
#rec = recall_score(list(dEval['label']), list(dEval['Class']), pos_label=1, average='micro')
#f1 = f1_score(list(dEval['label']), list(dEval['Class']), pos_label=1, average='micro')
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(cf, index = ['1', '2', '3'],
                  columns = ['1', '2', '3'])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

print(cf)
print(cr)
log.warning("scores computed subtask 5.")
print("scores computed subtask 5.")

# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write("Task7F: " + str(f1)+"\n")
    output_file.write("Task7P: " + str(prec)+"\n")
    output_file.write("Task7R: " + str(rec)+"\n")
    output_file.flush()

log.warning("output file written subtask 5.")
print("output file written subtask 5.")
