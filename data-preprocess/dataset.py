from datasets import load_dataset, Dataset, concatenate_datasets
from unidecode import unidecode
import string
printable = set(string.printable)

def normalize_text(text):
    return unidecode(text).encode('ascii', 'ignore').lower().decode('utf-8')

def load_bookcorpus():
    data = load_dataset('bookcorpus', split='train')
    data = data.filter(lambda x: len(x['text']) > 0, num_proc=8)
    data = data.map(lambda x: {'text': normalize_text(x['text'])}, num_proc=8)

    return data

def load_wikipedia():
    data = load_dataset('wikipedia', '20220301.en', split='train')
    data = data.filter(lambda x: len(x['text']) > 0, num_proc=8)
    data = data.map(lambda x: {'text': normalize_text(x['text'])}, num_proc=8)

    return data

def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-v1', split='train+validation+test') 
    data = data.filter(lambda x: len(x['text']) > 0)
    data = data.map(lambda x: {'text': normalize_text(x['text'])})

    return data

def load():
    '''
    print("Loading datasets")
    bookcorpus = load_bookcorpus()
    print("BookCorpus loaded")
    wikipedia = load_wikipedia()
    print("Wikipedia loaded")
    '''
    data1 = load_wikitext()
    data2 = load_wikitext()

    data = concatenate_datasets([data1, data2])
    print("Datasets merged")

    return data
