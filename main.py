import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
lst=LancasterStemmer()

print("Analyzing...")

# Text to be analysed is in the read.txt file
text=open('read.txt',encoding='utf-8').read()

# Taking words and respective emotions from emotions.txt file and storing in seperate lists
emotions=open('emotions.txt',encoding='utf-8').read()
sample = emotions.replace('\n', '')
sample1 = sample.split(',')
sample2 = []
for line in sample1:
    sample2.append(line.strip())

sample3 = []
for line in sample2:
    sample3.append(line.split(':'))

sample4 = []
for line1 in sample3:
    for line in line1:
        sample4.append(line.strip())

sample5 = []
for word in sample4:
    sample5.append(word.replace("'", ""))

words_list = []
emotions_list = []
i = 0
for word in sample5:
    if i == 0:
        words_list.append(word)
        i = 1
    elif i == 1:
        emotions_list.append(word)
        i = 0

# Stemming the words_list
stemmed_words_list=[]
for words in words_list:
    stemmed_words_list.append(lst.stem(words))

# Short forms and their full forms are in abrevations.txt
abbreviations=open('abbreviations.txt',encoding='utf-8').read()
abr_lower=abbreviations.lower()
abr=abr_lower.split('\n')

# Seperating the short forms into seperate lists
short_list = []
mean_list = []
i = 0
for word in abr:
    if i == 0:
        short_list.append(word)
        i = 1
    elif i == 1:
        mean_list.append(word)
        i = 0

# Replacing all the full forms that show a emotion into thier respective emotion in the mean_list
temp = []
for line in mean_list:
    l = mean_list.index(line)
    temp = line.split(" ")
    for w in temp:
        stem = lst.stem(w)
        if stem in stemmed_words_list:
            idx = stemmed_words_list.index(stem)
            mean_list[l] = emotions_list[idx]
            continue

# Obtaining emotions and their sarcastic meanings in sarcastic_emotions.txt and storing in seperate variables
i=0
sarcastic_emotion=[]
opp_emotion=[]
text1 = open("sarcastic_emotions.txt", encoding="utf-8-sig").read()
text2 =text1.split('\n')
for word in text2:
    if i==0:
        i=1
        sarcastic_emotion.append(word)
    elif i==1:
        i=0
        opp_emotion.append(word)

# Detecting sarcasm by checking if the line has sudden change in emotion from positive to negative

lower_text = text.lower()
sentences = sent_tokenize(lower_text)

half1 = ''
half2 = ''
final_line = ''
for line in sentences:
    temp = word_tokenize(line)
    for i in range(0, int(len(temp) / 2)):
        half1 += temp[i]
        half1 += ' '
    for i in range(int(len(temp) / 2), len(temp)):
        half2 += temp[i]
        half2 += ' '
    senti_half1 = SentimentIntensityAnalyzer().polarity_scores(half1)
    senti_half2 = SentimentIntensityAnalyzer().polarity_scores(half2)
    if senti_half1['compound'] > senti_half2['compound'] and senti_half2['compound'] < 0:
        tokens = word_tokenize(half1)
        for token in tokens:
            if SentimentIntensityAnalyzer().polarity_scores(token)['pos'] == 1:
                if lst.stem(token) in stemmed_words_list:
                    emotion = emotions_list[stemmed_words_list.index(lst.stem(token))]
                    if emotion in sarcastic_emotion:
                        final_line += opp_emotion[sarcastic_emotion.index(emotion)]
                        final_line += ' '
                    else:
                        final_line += emotion
                        final_line += ' '
                else:
                    final_line += token
                    final_line += ' '
            else:
                final_line += token
                final_line += ' '
        final_line += half2

        lower_text = lower_text.replace(line, final_line)
        final_line = ''

    half1 = ''
    half2 = ''


cleaned_text=lower_text.translate(str.maketrans('','',string.punctuation))

# Next step is to tokenize the text and remove stop words from it
tokenized_text=word_tokenize(cleaned_text)


# Replacing the short forms in the text with their full form
for word in tokenized_text:
    if word in short_list:
        indx=short_list.index(word)
        indx1=tokenized_text.index(word)
        tokenized_text[indx1]=mean_list[indx]

clean_text=[]
for word in tokenized_text:
    if word not in string.punctuation and word not in stopwords.words('english'):
        clean_text.append(word)

stemmed_text=[]

for words in clean_text:
    stemmed_text.append(lst.stem(words))

# Getting all the emotions in the text finally after all the cleaning
analyzed_emotions_stem = []
for word in stemmed_text:
    if word in stemmed_words_list:
        index = stemmed_words_list.index(word)
        analyzed_emotions_stem.append(emotions_list[index])
# Counting all the emotions
emotion_count_stem=Counter(analyzed_emotions_stem)

print("Done..!!")

# Ploting a graph on the emotions
fig, ax1 = plt.subplots()
ax1.bar(emotion_count_stem.keys(), emotion_count_stem.values(), align='center')
fig.autofmt_xdate()
plt.xticks(rotation=90)
plt.savefig('graph.png')
plt.show()
