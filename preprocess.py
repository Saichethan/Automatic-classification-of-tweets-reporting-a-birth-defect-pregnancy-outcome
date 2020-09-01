import os
import csv
import re,string
import numpy as np
import gensim
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import pickle


def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@', '#']
    hashtag_prefixes = '#'
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    hashtags = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes :
                words.append(word)
            elif word[0] is hashtag_prefixes:
            	hashtags.append(word[1:])
    sentence = ' '.join(words)
    return (sentence, hashtags)





def process(mode):

	print("\n")
	print(mode.title()," data stats")
	base_dir = "/home/reddy/Task5/"
	prefix = "data/task5_"
	file_name = base_dir + prefix + str(mode) + ".tsv"


	maxlen = 0
	max_hash = 0
	avg_hash = 0
	count_hash = 0
	count = 0
	sentences = []
	hashtags = []
	labels = []
	tweetId = []
	userId = []
	original_tweet = []

	with open(file_name, encoding='mac_roman') as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		next(tsvreader) #skip headers
		for line in tsvreader:
			count = count + 1
			tweetId.append(line[0]) #tweet ID
			userId.append(line[1]) #user ID
			original_tweet.append(line[2]) #tweet
			#print(mode)
			try:
				labels.append(int(line[3])-1) #classes
			except:
				pass
			tweet = line[2][:]
			tweet, hashtag = strip_all_entities(strip_links(tweet.lower()))
			hashtags.append(hashtag) #list of hashtags
			sentences.append(tweet) #processed tweets
			templen = len(tweet.split(" "))
			
			if templen > 140:
				templen = 0

			maxlen = max(maxlen, templen)			
			max_hash = max(max_hash, len(hashtag))
			if(len(hashtag)>0):
				avg_hash = avg_hash + len(hashtag)
				count_hash = count_hash + 1
	
	print("Max tweet Length: ",maxlen)		
	print("Max Hashtags: ", max_hash)
	print("Avg Hashtag: ", avg_hash/count_hash) #in tweets having atleast one tweet
	print("No of tweets with atleast one Hashtag: ", count_hash)
	print("Total no. of tweets: ", count)

	
	labels = np.asarray(labels)
	userId = np.asarray(userId)
	original_tweet = np.asarray(original_tweet)
	tweetId = np.asarray(tweetId)

	return tweetId, userId, original_tweet, sentences, hashtags, labels


process("test")

def load_data():


	size = 300
	test_tweetId, test_userId, test_original_tweet, test_sentences, test_hashtags, test_labels = process("test")
	train_tweetId, train_userId, train_original_tweet, train_sentences, train_hashtags, train_labels = process("training")
	val_tweetId, val_userId, val_original_tweet, val_sentences, val_hashtags, val_labels = process("validation")
	

	#xx = input("?")
	total = train_sentences + train_hashtags + val_sentences + val_hashtags + test_sentences + test_hashtags  # word matrix from both tweet and hashtags
	
	tokenizer = Tokenizer(oov_token="<OOV>") #oov_token="<OOV>"
	tokenizer.fit_on_texts(total)
	vocabulary = tokenizer.word_index

	train_sequences = tokenizer.texts_to_sequences(train_sentences)
	train_hashseq = tokenizer.texts_to_sequences(train_hashtags)

	val_sequences = tokenizer.texts_to_sequences(val_sentences)
	val_hashseq = tokenizer.texts_to_sequences(val_hashtags)

	test_sequences = tokenizer.texts_to_sequences(test_sentences)
	test_hashseq = tokenizer.texts_to_sequences(test_hashtags)

	train_tweet = pad_sequences(train_sequences,padding="post",truncating="post",maxlen=75)
	train_hash = pad_sequences(train_hashseq,padding="post",truncating="post",maxlen=3)	

	val_tweet = pad_sequences(val_sequences,padding="post",truncating="post",maxlen=75)
	val_hash = pad_sequences(val_hashseq,padding="post",truncating="post",maxlen=3)	

	test_tweet = pad_sequences(test_sequences,padding="post",truncating="post",maxlen=75)
	test_hash = pad_sequences(test_hashseq,padding="post",truncating="post",maxlen=3)	
	
	word_matrix = pickle.load(open('saves/word_matrix.np', 'rb'))
	vocab_size = len(vocabulary)+1
	
	b = 0
	"""
	embeddings_dictionary = dict()
	glove_file = open("/home/reddy/Long_v1/assets/glove.6B.300d.txt")
	for line in glove_file:
		records = line.split()
		word = records[0]
		vector_dim = np.asarray(records[1:],dtype="float")
		embeddings_dictionary[word] = vector_dim
	glove_file.close() 


	word_matrix = np.zeros((len(vocabulary)+1, size)) 
	model = gensim.models.KeyedVectors.load_word2vec_format('/home/reddy/clss/GoogleNews-vectors-negative300.bin', binary=True)

	print("Creating word matrix....")
	
	print("vocab_size: ", vocab_size)
	for word, i in vocabulary.items():
		try:
			word_matrix[i] = embeddings_dictionary[word] #model.wv[word.lower()] #
		except KeyError:
			# if a word is not include in the vocabulary, it's word embedding will be set by random.
			word_matrix[i] = np.random.uniform(-0.25,0.25,size)
			b+=1	
	print('there are %d words not in model'%b)
	#np.ndarray.dump(word_matrix, open('saves/word_matrix.np', 'wb'))
	"""
	



	return vocab_size, train_tweet, train_hash, train_labels, val_tweet, val_hash, val_labels, test_tweet, test_hash, test_labels, word_matrix, val_tweetId, val_userId, val_original_tweet 

