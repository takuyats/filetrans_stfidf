#!/usr/bin/env python
# coding: utf-8

# In[24]:


#@word クラス
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy
#ライブラリインポート
#基本ライブラリ
import numpy as np
import pandas as pd
#形態素解析
import MeCab
#モデル保存
#from sklearn.externals import joblib


# In[108]:


class StfidfVectorizer(TfidfVectorizer):
    """
    Parameters
    ----------
    corpus:スペース区切りのコーパス
    model：word2vecのmodel
    n：？√nでword2vecの分散表現上での類似語上位
    epoch：スコア更新回数
    word:更新対象の言葉   
    """


    def __init__(self,corpus,model,n,epoch,token_pattern,ngram_range):
        TfidfVectorizer.__init__(self)
        self.token_pattern=token_pattern
        self.ngram_range = ngram_range
        self._corpus = corpus
        self._model = model
        self._n = n
        self._epoch = epoch

    def update_score(self,word):
        X = self.fit_transform(self._corpus)
        wds = self.get_feature_names()        
        S = {}
        for wd in wds:
            #print(wd)
            S[wd] = X[0, self.vocabulary_[wd]] * 10.0
        print("「 {} 」の初期スコア：{}".format(word,S[word]))

        for ep in range(1,self._epoch):
            prev_S = S[word]
            weight_e = self.relevant_word(word,prev_S)
            S[word] = prev_S * 1/(
                             1+np.linalg.norm(self.e(wd))*np.linalg.norm(weight_e)*scipy.spatial.distance.cosine(self.e(word), weight_e)
                             )
        print("「 {}　」の最終スコア：{}".format(word,S[word]))


    def relevant_word(self,wd,prev_S):
        root_n = math.floor(math.sqrt(self._n))
        weightened_expected_value = 0 
        for i in range(1,int(root_n)+1):
            weight=            1/(1-self.all_weight_score(wd,root_n,prev_S))            *self.e(wd)
            weightened_expected_value += weight
        return weightened_expected_value/root_n


    def all_weight_score(self,wd,root_n,prev_S):
        try:
            results = self._model.wv.most_similar(positive=[wd])
            words =[w[0] for w in results]
            return prev_S/self.score_sum(prev_S,words)
        except KeyError:
            print ("not in vocabulary")
            return 0

    def e(self,wd):
        try:
            return model.wv[wd]
        except KeyError:
            return 0

    def score_sum(self,prev_S,words):
        scoresum = 0
        for word in words:
            #print(word)
            try:
                scoresum += S[word]
            except:
                pass
        return scoresum


# In[26]:


#学習データ読み込み
df = pd.read_csv("../tele2/theme/Train_変復調・符号変換.tsv",sep = '\t')
#df.info()


# In[62]:


#評価データ読み込み
df_test = pd.read_csv("../tele2/theme/Eval_変復調・符号変換.tsv",sep = '\t')
#df.info()


# In[27]:


# #形態素解析
mecabTagger = MeCab.Tagger("-Ochasen")
mecabTagger.parse('')


# In[60]:


#形態素解析を実施して、'名詞'の単語を抽出し、半角スペースで区切り１行にする関数
def get_surface(text):
    result = []
    #ストップワードリスト
    #名詞
    stopnounslist=["図","前記","上記","もの","こと","ため","よう","それぞれ","さ","ａ","装置"]
    #動詞
    stopverbslist=["する","られる","れる","なる","できる","せる","ある","いる","おる"]
    #区切り文字
    stopleftonelist=["(",")","[""]","{","}","（","）","［","］","｛","｝","「","」","〔","〕","『","』","｢","｣","【","】","＜","＞","<",">","%","+","-","/","'","\"","`","*",":",";",".","，",",","。","｡","÷","？","｜","|"]

    node = mecabTagger.parseToNode(text)
    while node:
        if node.feature.split(",")[0] in('名詞') and node.feature.split(",")[1] in('一般') and node.surface.isdigit() == False and node.surface not in(stopnounslist) and node.surface not in(stopleftonelist):
            result.append(node.surface)
        #elif node.feature.split(",")[0] in('動詞') and node.feature.split(",")[6] not in(stopleftonelist) and node.feature.split(",")[6] not in(stopverbslist):
            #result.append(node.feature.split(",")[6])
        node = node.next
    return " ".join(result)


# In[75]:


#学習データ用コーパス作成
text_tokenized = []
for text in df['text']:
    text_tokenized.append(get_surface(text))

df['text_tokenized'] = text_tokenized

#print(df['text_tokenized'][1:11])
#print(df['text_tokenized'][201:211])
#print(df['text_tokenized'][401:411])
#print(df['text_tokenized'][601:611])
#print(df['text_tokenized'][801:811])
#df['text_tokenized'].to_csv('学習データ_形態素解析.csv')


# In[63]:


#評価データ用コーパス作成
text_tokenized = []
for text in df_test['text']:
    text_tokenized.append(get_surface(text))

df_test['text_tokenized'] = text_tokenized
#print(df['text_tokenized'][1:11])
#print(df['text_tokenized'][201:211])
#print(df['text_tokenized'][401:411])
#print(df['text_tokenized'][601:611])
#print(df['text_tokenized'][801:811])
#df_test['text_tokenized'].to_csv('評価データ_形態素解析.csv', header=None)


# In[110]:


#TF-IDFで学習用データ、評価用データ作成
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
X       = vectorizer.fit_transform(df['text_tokenized'])    #学習用データ


# In[77]:


X       = pd.DataFrame(X.toarray())


# In[78]:


#次元数確認
X.info()


# In[82]:


#ラベル確認
df['theme'].drop_duplicates()


# In[83]:


#ブ－スティング用正解ラベル作成
df.loc[df['theme'] == '5J064', 'theme1'] = 1
df.loc[df['theme'] == '5J065', 'theme1'] = 2
df.loc[df['theme'] == '5K004', 'theme1'] = 3
df.loc[df['theme'] == '5K022', 'theme1'] = 4
df.loc[df['theme'] == '5K159', 'theme1'] = 5


# In[84]:


#型変換
df['theme1'] = df['theme1'].astype(int)


# In[85]:


#lightGBM
import lightgbm
clf_lgb = lightgbm.LGBMClassifier()
clf_lgb.fit(X, df['theme1'])


# In[31]:


#def make_corpus(text):
#コーパスはテストデータの方がよい？


# In[36]:


import math
from gensim.models import Word2Vec


# In[114]:


n = 100
epoch = 5
word = "符号"
#corpus = X
corpus = df['text_tokenized']


# In[90]:


sentences = []
for text in df['text_tokenized']:
    #text_list = text.split(' ')
    sentences.append(text)


# In[91]:


#Word2vecによる仮モデル
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)


# In[87]:


#lgbmによるモデル
model = clf_lgb


# In[ ]:





# In[115]:


stf = StfidfVectorizer(corpus,model,n,epoch,token_pattern='(?u)\\b\\w+\\b',ngram_range=(1,1))
#words=["feed","a","the","compact","stowage","and","deployment"]
words=["ハードディスク","メモリ","データ","双方","アドレス"]
for word in words:
    stf.update_score(word)


# In[ ]:




