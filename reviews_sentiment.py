import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer
import string
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train =[]
train_y = []


with open("train.csv") as train_set:
    train_set=train_set.readlines()
    for i,line in enumerate(train_set):
        line=str(line).split('\t')
        train.append(line[1])
        train_y.append(line[0])
        if "\n" in line:
            train[i]=train[i].replace('\n',"")


##with open("test.csv") as test:
##    test=test.readlines()
##    for i,line in enumerate(test):
##        line=str(line)
##        if "\n" in line:
##            line=line.replace('\n',"")
##            test[i]=line
##
##
##
            
#randomly select some data to reduce size
random.seed(123)
select=random.sample(range(18505),18000)
select_train = select[0:15000]
select_test = select[15001:] 

test = [train[i] for i in select_test]
train = [train[i] for i in select_train]


test_y = [train_y[i] for i in select_test]
train_y = [train_y[i] for i in select_train]



def preprocessing(text):
    
    # tokenize into words
    tokens = text.split()
    for i,word in enumerate(tokens):
        if (word in ['not','no'] or "n't" in word) and (i != len(tokens) - 1):
            tokens.append("not_" + tokens[i+1])
            del tokens[i]
            del tokens[i+1]

    # remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text= ' '.join(tokens)

    return preprocessed_text 



for i,line in enumerate(train):
    train[i] = preprocessing(line)
    
for i,line in enumerate(test):
    test[i] = preprocessing(line)


def get_all_words(d1,d2):
    allw=[]
    for line in d1:
        for word in line.split():
            allw.append(word)
    for line in d2:
        for word in line.split():
            allw.append(word)

    return set(allw)
    


def predict(num_features):
    vectorizer = CountVectorizer(max_features=num_features)
    #vectorizer.fit(get_all_words(train,test))
    train_X=vectorizer.fit_transform(train)
    test_X=vectorizer.transform(test)


    MNB = MultinomialNB()

    MNB.fit(train_X, train_y)


    pred = MNB.predict(test_X)

    return pred

def predict_tfidf(num_features):
    vectorizer = TfidfVectorizer(max_features=num_features)
    #vectorizer.fit(get_all_words(train,test))
    train_X=vectorizer.fit_transform(train)
    test_X=vectorizer.transform(test)


    MNB = MultinomialNB()

    MNB.fit(train_X, train_y)


    pred = MNB.predict(test_X)

    return pred







possible_n = [2500+ 100*i for i in range(0, 10)]

cnt_accuracies = [accuracy_score(test_y,predict(n)) for n in possible_n]
tfidf_accuracies = [accuracy_score(test_y,predict_tfidf(n)) for n in possible_n]


plt.plot(possible_n, cnt_accuracies, label='Word Count')
plt.plot(possible_n, tfidf_accuracies, label='Tf-idf')
plt.legend()


print(max(cnt_accuracies+tfidf_accuracies))

plt.show()
