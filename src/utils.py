import nltk
import string

def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)

if __name__ == "__main__":
    string_sample =  "What is the step by step guide to invest in share market in india?"
    generator_obj = tokenize(string_sample)
    for i in generator_obj:
        print(i, end=" ")