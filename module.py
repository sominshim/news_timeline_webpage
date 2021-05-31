
import numpy as np
import pandas as pd
import matplotlib
# import matplotlib.pyplot as plt
import pickle
import datetime
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS # sklearn 0.21.1
from collections import Counter
from soynlp.noun import NewsNounExtractor

from datetime import datetime
from soynlp.tokenizer import LTokenizer

# 시각화
import matplotlib.pyplot as plt
import seaborn as sb

# file I/O
import pickle
import pandas as pd

import re # 정규표현식
import numpy as np # 연산
import time # 연산 시간 측정

from collections import Counter
from soynlp.noun import NewsNounExtractor
# 문서 요약
from krwordrank.word import KRWordRank
from krwordrank.sentence import make_vocab_score, MaxScoreTokenizer, keysentence
import re

# 뉴스 링크 크롤링
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
time_list = [] # 각 단계별 처리 시간 저장
process = ['load_data' ,' preprocess - get proper docs A','preprocess - get proper doc B',
           'get category', 'get topic - vectorize', 'get topic-DBSCAN','get main doc']

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings("ignore")


# 벡터화
class Vectorizer:
    def help(self):
        print("******Vectorizer******")
        print("1)get_tfidf_vec(토큰화된 문서(Series),단어 수(int)) : 문서를 tfidf 벡터(x) 와 단어(words)로 반환")
        print("2)get_doc2vec(토큰화된 문서(Series)) : doc2vec 벡터 반환")
        print("3)load_doc2vec_model(토큰화된 문서(Series),모델객체(word2vec_obj)): 저장된 모델로  doc2vec 벡터 반환")
        print("*****************************")

    def get_tfidf_vec(self, query_doc, max_feat=None, min_df = 0, max_df = 1.0):
        query_doc = query_doc.apply(lambda x: ' '.join(x))
        obj = TfidfVectorizer(max_features=max_feat, min_df = min_df, max_df = max_df)  # max_features for lda
        x = obj.fit_transform(query_doc).toarray()
        words = np.array(obj.get_feature_names())
        return x, words

    def get_doc2vec(self, query_doc,
                    dm=1, dbow_words=1, window=8, vector_size=50, alpha=0.025,
                    seed=42, min_count=5, min_alpha=0.025, workers=4, hs=0, negative=10,
                    n_epochs=50, model_name='d2v.model'):
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(query_doc)]
        model = Doc2Vec(
            dm=dm,  # PV-DBOW => 0 / default 1
            dbow_words=dbow_words,  # w2v simultaneous with DBOW d2v / default 0
            window=window,  # distance between the predicted word and context words
            vector_size=vector_size,  # vector size
            alpha=alpha,  # learning-rate
            seed=seed,
            min_count=min_count,  # ignore with freq lower
            min_alpha=min_alpha,  # min learning-rate
            workers=workers,  # multi cpu
            hs=hs,  # hierarchical softmax / default 0
            negative=negative,  # negative sampling / default 5
        )
        model.build_vocab(tagged_data)
        print("corpus_count: ", model.corpus_count)

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print('epoch: ', epoch)
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            model.alpha -= 0.0002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(model_name)
        print("Model Saved")
        model_loaded = Doc2Vec.load(model_name)
        print("Load Model")
        x_doc2vec = []
        for i in range(len(query_doc)):
            x_doc2vec.append(model_loaded.docvecs[i])
        x_doc2vec = np.array(x_doc2vec)
        return x_doc2vec

    def load_doc2vec_model(self, query_doc, model_name):
        print("Load Model")
        model_loaded = Doc2Vec.load(model_name)
        x_doc2vec = []
        for i in range(len(query_doc)):
            x_doc2vec.append(model_loaded.docvecs[i])
        x_doc2vec = np.array(x_doc2vec)
        return x_doc2vec

# 클러스터링 및 시각화
class Get2DPlot:
    def help(self):
        print("******Get2DPlot******")
        print("1)get_2D_vec(벡터(ndarray),벡터 종류(string), 차원축소 방법(string)) : ")
        print("2 차원으로 차원축소된 벡터를 반환, 벡터 종류 = (tfidf, doc2vec) 차원축소 방법 = (TSNE,PCA)")
        print(
            "2)get_cluster_labels(클러스터링 방법(string), min_samples(int), min_range(int), optimal_eps(boolean), eps(float)")
        print('''
            1)에서 받은 벡터를 군집화하고, 라벨 리스트를 반환, 
            optimal_eps = True 면 엡실론 최적화 함수 실행
            min_sample 은 DBSCAN min_sample과 동일, min_range는 평균 변화율 계산 구간의 너비
            ''')
        print("클러스터링 방법은 kmeans, DBSCAN 중 하나 선택. kmeans를 선택할 경우 클러스터 개수를 입력, DBSCAN 은 eps 입력")
        print("3)plot2D(): 2)에서 실행한 군집화 결과를 2차원으로 시각화")
        print("*****************************")

    def __init__(self, learning_rate=200, random_state=10):
        self.learning_rate = learning_rate
        self.random_state = random_state

    def get_2D_vec(self, x, vec_kind='tfidf', reduction_method='TSNE'):
        self.reduction_method = reduction_method
        if vec_kind == 'tfidf':
            self.x_scaled = x
        elif vec_kind == 'doc2vec':
            self.x_scaled = StandardScaler().fit_transform(x)
        else:
            print('vec_kind 는 tfidf 혹은 doc2vec 만 가능하다. 이상한거 넣지 말긔')
            raise NotImplementedError
        if self.reduction_method == 'TSNE':
            t_sne = TSNE(n_components=2, learning_rate=self.learning_rate, init='pca',
                         random_state=self.random_state)
            self.vec = t_sne.fit_transform(self.x_scaled)
        elif self.reduction_method == 'PCA':
            pca = PCA(n_components=2)
            self.vec = pca.fit_transform(self.x_scaled)
        return self.vec

    def get_best_eps(self, vector, min_samples=5, min_range=4):
        l = len(vector)
        nn = NearestNeighbors(n_neighbors=min_samples, metric='cosine').fit(vector)

        distances, indices = nn.kneighbors(vector)
        min_samples -= 1
        candi = sorted(distances[:, min_samples])
        rate1 = []
        for i in range(min_range, l):
            dy1 = candi[i - 1] - candi[i - min_range]
            dy2 = candi[i] - candi[i - (min_range - 1)]
            e = 10 ** (-6)
            rate1.append((dy2 + e) / (dy1 + e))
        index = rate1.index(max(rate1))
        best_eps = candi[index]
        # plt.figure(figsize = (10,5))
        # plt.plot(candi)
        # plt.axhline(y = best_eps, color = 'red')
        return best_eps

    def get_cluster_labels(self, optimal_eps=False, min_samples=5, min_range=4, eps=0.5, cluster_method='kmeans'):
        if optimal_eps == True:
            self.eps = self.get_best_eps(self.x_scaled, min_samples=min_samples, min_range=min_range)
        else:
            self.eps = eps
        self.cluster_method = cluster_method

        if self.cluster_method == 'kmeans':
            print('클러스터 개수를 입력하세요 :')
            self.n_clusters = int(input())
            cluster = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.x_scaled)
            self.cluster_labels = cluster.labels_

        elif self.cluster_method == 'DBSCAN':
            self.cluster_labels = DBSCAN(eps=self.eps, min_samples=min_samples, metric='cosine').fit_predict(
                self.x_scaled)

        elif self.cluster_method == 'OPTICS':
            opt = OPTICS(min_samples=min_samples, max_eps=0.9, metric='cosine')
            opt.fit(self.x_scaled)
            self.cluster_labels = opt.labels_
        else:
            print('cluster method는  kmeans, DBSCAN, OPTICS 중에서만 골라주세용')
            raise NotImplementedError

        vec_pd = np.c_[self.vec, self.cluster_labels]
        self.vec_pd = pd.DataFrame(vec_pd, columns=['x', 'y', 'labels'])

        return self.cluster_labels

    def plot2D(self):
        print(self.reduction_method, self.cluster_method)
        groups = self.vec_pd.groupby('labels')
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group.x,
                    group.y,
                    marker='o',
                    linestyle='',
                    label=name)
        # ax.legend(fontsize=12, loc='upper left') # legend position

        plt.title('%s Plot of %s' % (self.reduction_method, self.cluster_method), fontsize=20)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.show()

# 토큰화 관련 함수
def cleanText(readData):
    nouns_list = {}
    for key, value in readData.items():
        if len(re.findall('[-=+,#/\?:^$.@*\"“”※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》|0-9]', key)) == 0:
            nouns_list[key] = value

    return nouns_list

def remove_common_word(readData, common_list):
    final_nouns_list = {}
    for key, value in readData.items():
        if key not in common_list:
            final_nouns_list[key] = value

    return final_nouns_list

def noun_extractor(cat_df):
    noun_extractor = NewsNounExtractor(
        max_left_length=10,
        max_right_length=7,
        predictor_fnames=None,
        verbose=True
    )

    nouns = noun_extractor.train_extract(cat_df['body_prep'])

    # score 점수에 따라서 분류
    # 빈도수 * 명사점수 기준

    sorted_nouns = sorted(nouns.items(),
                          key=lambda x: -x[1].frequency * x[1].score)

    sorted_by_score = {}

    for i, (word, score) in enumerate(sorted_nouns):
        if score.score != 0:
            sorted_by_score[word] = score.score
            # print('%6s (%.2f)' % (word, score.score), end='')

    nouns_list = cleanText(sorted_by_score)

    # 자주 등장하는 단어
    n = 50
    common_words = Counter(nouns_list.keys()).most_common(n)

    common_list = []
    for i in range(n):
        common_list.append(common_words[i][0])

    final_nouns_list = remove_common_word(nouns_list, common_list)
    # final_nouns_list
    return final_nouns_list

def tokenize(corpus, tokenizer, final_nouns_list):
    tok_sent = []
    for sent in corpus:
        # sent = corpus[i]
        tok_sent.append(tokenizer.tokenize(sent))

    noun_sent = []
    for doc in tok_sent:
        nouns = [word for word in doc if word in final_nouns_list]
        noun_sent.append(nouns)

    token = [doc for doc in noun_sent]
    return token

def search_by_keyword(key=None, data=None):
    key = key
    match_cluster=[]
    for i in data.keys():
        keywords = data.get(i)
        for word in keywords:
            if word==key:
                match_cluster.append(i)
    return match_cluster

# 날짜형식 -> 문자열로 변환
def datetime_to_string(dt_series):
    return [dt.strftime("%Y-%m-%d") for dt in dt_series.tolist()]

def news_summarization(df, col):
    result_summ = []
    for i in range(len(df)):
        ch = re.compile('([.])').split(df[col].iloc[i])
        summary_text = []

        for s in map(lambda a, b: a + b, ch[::2], ch[1::2]):
            summary_text.append(s)

        wordrank_extractor = KRWordRank(
            min_count=5,  # 단어의 최소 출현 빈도수 (그래프 생성 시)
            max_length=10,  # 단어의 최대 길이
            verbose=True
        )

        beta = 0.85  # PageRank의 decaying factor beta
        max_iter = 10
        keywords, rank, graph = wordrank_extractor.extract(summary_text, beta, max_iter, num_keywords=100)

        stopwords = {}  # 뉴스 본문 크기가 작은 관계로 생략
        vocab_score = make_vocab_score(keywords, stopwords, scaling=lambda x: 1)
        tokenizer = MaxScoreTokenizer(vocab_score)

        penalty = lambda x: 0 if 25 <= len(x) <= 80 else 1

        sents = keysentence(
            vocab_score, summary_text, tokenizer.tokenize,
            penalty=penalty,
            diversity=0.3,
            topk=3
        )

        sents = " ".join(sents)
        result_summ.append(sents)

    result_summ = np.array(result_summ).flatten().tolist()
    return result_summ

def timeline(cat_num, keyword, data_path):
    catergories = ['law', 'education', 'welfare', 'traffic', 'accident', 'environment', 'region', 'health']
    cat_df = pd.read_csv(data_path + '/cat_data/{}.csv'.format(catergories[int(cat_num)]))

    # load data
    with open(data_path + '/keyword_data/{}_keywords.pickle'.format(catergories[int(cat_num)]), 'rb') as fr:
        cluster_info = pickle.load(fr)

    # 검색어 입력 시 키워드가 포함된 클러스터 번호 리턴
    match_cluster = search_by_keyword(keyword, cluster_info)
    keyword_df = cat_df[cat_df['labels'].isin(match_cluster)]

    if keyword_df.empty:
        return print("키워드에 해당하는 타임라인이 없습니다.")

    final_nouns_list = noun_extractor(cat_df)

    keyword_df['tokenized_title'] = tokenize(keyword_df['title'], LTokenizer(scores=final_nouns_list), final_nouns_list)
    keyword_df['tokenized_body'] = tokenize(keyword_df['body_prep'], LTokenizer(scores=final_nouns_list),
                                            final_nouns_list)

    df_groupby = keyword_df.groupby('labels').count()
    docs_labels = df_groupby[df_groupby['date'] > 1].index.tolist()
    labels_list = keyword_df['labels'].unique()
    doc1_labels = [x for x in labels_list if x not in docs_labels]

    keyword1_df = keyword_df[keyword_df['labels'].isin(doc1_labels)]
    keyword_over1_df = keyword_df[keyword_df['labels'].isin(docs_labels)]

    # 카테고리별 단어리스트와 TFIDF를 딕셔너리로 저장
    tic = time.time()

    x_cat_dict = {}  # 카테고리별 tfidf 벡터
    word_cat_dict = {}  # 카테고리별 단어

    for cat in docs_labels:
        vec_obj = Vectorizer()
        cat_docs = keyword_over1_df[keyword_over1_df['labels'] == cat]['tokenized_body']

        x_cat, word_cat = vec_obj.get_tfidf_vec(cat_docs)

        x_cat_dict[cat] = x_cat
        word_cat_dict[cat] = word_cat

    toc = time.time()
    print(f'excution time : {toc - tic}')
    time_list.append(toc - tic)


    ######## clustering ###########
    cluster_obj = Get2DPlot()

    tic = time.time()

    topic_label_dict = {}
    for category, tfidf_vec in x_cat_dict.items():
        cluster_obj = Get2DPlot()

        vec_2d = cluster_obj.get_2D_vec(tfidf_vec, 'tfidf', 'PCA')
        topic_label = cluster_obj.get_cluster_labels(True, min_samples=2, min_range=1, cluster_method='OPTICS')
        # print(f'****{category}****')
        # cluster_obj.plot2D()

        topic_label_dict[category] = topic_label
    toc = time.time()
    print(f'excution time : {toc - tic}')
    time_list.append(toc - tic)

    # 2개 이상의 문서를 가진 labels viewpoint 할당
    df_final = pd.DataFrame()

    point = 0
    for cat in docs_labels:
        df_tmp = keyword_df[keyword_df['labels'] == cat].copy()
        df_tmp['viewpoint'] = topic_label_dict[cat] + point

        df_tmp.loc[df_tmp['viewpoint'] == (point - 1), 'viewpoint'] = -1
        point += len(np.unique(topic_label_dict[cat]))

        df_final = pd.concat([df_final, df_tmp])
        # break

    if len(doc1_labels) != 0:
        keyword1_df = keyword_df[keyword_df['labels'].isin(doc1_labels)]
        keyword_over1_df = keyword_df[keyword_df['labels'].isin(docs_labels)]

        # 1개 문서를 가진 labels viewpoint 할당
        i = df_final['viewpoint'].max() + 1
        v_point = list(range(i, i + keyword1_df.shape[0]))
        keyword1_df['viewpoint'] = v_point

        df_final = pd.concat([df_final, keyword1_df])

    df_final = df_final[df_final['viewpoint'] != -1]
    df_final['datetime'] = pd.to_datetime(df_final['date'])

    timeline = df_final.groupby('viewpoint')['datetime', 'title', 'body_prep', 'body_for_summ'].min()
    timeline.sort_values('datetime', ascending=True, inplace=True)
    # timeline['datetime'] = datetime_to_string(timeline['datetime'])

    timeline['body_summ'] = news_summarization(timeline, 'body_for_summ')
    timeline['datetime'] = datetime_to_string(timeline['datetime'])

    date_body_timeline = timeline[['datetime', 'body_summ']].values.tolist()
    date_title_body_timeline = timeline[['datetime', 'title', 'body_summ']].values.tolist()

    dt_article_list = np.array(date_body_timeline).flatten().tolist()
    # dt_title_article_list = np.array(date_title_body_timeline).flatten().tolist()


    return dt_article_list
    # return dt_title_article_list