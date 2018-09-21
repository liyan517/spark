import re
import sys

from collections import Counter

import math
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.functions import *
from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType

review_file = sys.argv[1]
stopword_file = sys.argv[2]
query_file = sys.argv[3]


def load_stopwords(stopword_file):
    with open(stopword_file) as f:
        lines = f.read().splitlines()
    return lines

stopwords = load_stopwords(stopword_file)
def split(doc):
    return list([word for word in (re.split(r'[^\w]+',str.lower(doc))) if word not in stopwords])

spark = SparkSession \
    .builder \
    .appName("Assignment 1 - TF-IDF") \
    .getOrCreate()
    #.config("spark.some.config.option", "some-value") \

df = spark.read.json(review_file)
num_doc = df.count()

df = df.withColumn('documentID', sf.concat(sf.col('reviewerID'), sf.lit('_'), sf.col('asin')))
df = df.withColumn('document', sf.concat(sf.col('reviewText'), sf.lit(' '), sf.col('summary')))

doc_df = df[['documentID', 'document']]
doc_df = doc_df.withColumn("words", sf.udf(split, ArrayType(StringType()))('document'))


def count_TF(words):
    counts = Counter(words)
    return [counts[word] for word in query_words]

def find_similarity(q):
    query_words = re.split(r'[^\w]+',str.lower(q))

    def count_TF(words):
        #count term frequency
        counts = Counter(words)
        return [counts[word] for word in query_words]

    query_df = doc_df.withColumn('query_words',sf.udf(lambda words: [word for word in words if word in query_words],ArrayType(StringType()))('words'))
    tf_df = query_df.withColumn('tf_vectors', sf.udf(count_TF, ArrayType(IntegerType()))('query_words'))
    # split the words into rows.
    doc_word_df = query_df.withColumn('word', sf.explode('query_words'))
    #get the count the number of documents that contains each query word
    words_doc_count = doc_word_df.groupBy('word').agg(sf.count(sf.lit(1)).alias('DF'))
    #get the IDF of the query words
    df_vector = [words_doc_count.filter("word == '{}'".format(word)).first().DF for word in query_words]
    def tf_idf(tf_vector):
        tfidf = []
        tfidf_nor = []
        for i in range(len(tf_vector)):
            if tf_vector[i] == 0 or df_vector[i] == 0:
                tfidf.append(0)
            else:
                data = ((1 + math.log(tf_vector[i])) * math.log(num_doc / df_vector[i]))
                tfidf.append(data)
        S = 0
        for i in range(len(tf_vector)):
            S += pow(tfidf[i], 2)
        for i in range(len(tf_vector)):
            tfidf_nor.append((tfidf[0] / math.sqrt(S)) if S != 0 else 0)
        return tfidf_nor

    tfidf_df = tf_df.withColumn('tfidf', sf.udf(tf_idf)('tf_vectors'))
    tfidf_df = tfidf_df.withColumn('nor_tf_idf', sf.udf(nor_tf_idf)('tfidf'))
    query_tfidf = nor_tf_idf(tf_idf(count_TF(query_words)))

    def cos_similarity(tfidf):
        tfidf_array = np.array(tfidf)
        query_array = np.array(query_tfidf)
        dot_product = np.dot(tfidf_array, query_array)
        print(dot_product)
        norm_product = np.linalg.norm(tfidf_array) * np.linalg.norm(query_array)
        return (dot_product / norm_product).tolist() if norm_product != 0 else 0

    tfidf_df = tfidf_df.withColumn('cos_simi', sf.udf(cos_similarity, DoubleType())('nor_tf_idf'))
    simi_df = tfidf_df[['documentID', 'cos_simi']]
    simi_df.filter(simi_df.cos_simi != 0).orderBy(simi_df.cos_simi.desc()).write.parquet("result")




with open(query_file) as query:
    query_lines = query.read().splitlines()
    find_similarity(query_line for query_line in query_lines)





def tf_idf(tf_vector):
    tfidf = []
    tfidf_nor = []
    for i in range(len(tf_vector)):
        if tf_vector[i] == 0 or df_vector[i] == 0:
            tfidf.append(0)
        else:
            data = ((1 + math.log(tf_vector[i])) * math.log(num_doc / df_vector[i]))
            tfidf.append(data)
    return tfidf
def nor_tf_idf(tf_idf):
    S = 0
    tfidf_nor = []
    for i in range(len(tf_idf)):
        S += pow(tf_idf[i], 2)
    for i in range(len(tf_idf)):
        tfidf_nor.append((tf_idf[i] / math.sqrt(S)) if S != 0 else 0)
    return tfidf_nor






def cos_similarity(tfidf):
    tfidf_array = np.array(tfidf)
    query_array = np.array(query_tfidf)
    dot_product = np.dot(tfidf_array, query_array)
    norm_product = np.linalg.norm(tfidf_array) * np.linalg.norm(query_array)
    return (dot_product / norm_product).tolist() if norm_product != 0 else 0.0
