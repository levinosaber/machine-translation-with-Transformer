import unicodedata
import re
import pandas as pd
import spacy
import os
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator  # for build a Vocab like class for dataset

from data_processing.Dataset import TranslationDataset

MAX_LEN = 100  # max length of sentence

# transform unicode to ascii
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# normalize string
def normalizeString(s):
    s = s.lower().strip()
    # s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)  # \1表示group(1)即第一个匹配到的 即匹配到'.'或者'!'或者'?'后，一律替换成'空格.'或者'空格!'或者'空格？'
                                       # # \1 represents group(1), which is the first match. Specifically, after matching '.', '!', or '?', replace them with 'space.', 'space!', or 'space?' respectively.
    s = re.sub(r"[^a-zA-Z.!?'À-ÿ]+", r" ", s)  # 非字母以及非.!?的其他任何字符 一律被替换成空格
                                               # Any character that is not a letter and not one of .!? should be replaced with a space.
    s = re.sub(r'[\s]+', " ", s)  # 将出现的多个空格，都使用一个空格代替。例如：w='abc  1   23  1' 处理后：w='abc 1 23 1'
                                  # # Replace multiple spaces with a single space. For example: if w='abc 1 23 1', after processing it would be: w='abc 1 23 1'.
    return s

class DataReader:
    '''
    to read the data from the file, and do basic text data preprocessing.
    '''
    def __init__(self, src_file_path, tgt_file_path, load_data_lines=10000) -> None:
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path if tgt_file_path else src_file_path

        self.src_raw_data = pd.read_table(self.src_file_path, header=None, encoding="utf-8", nrows=load_data_lines)
        self.tgt_raw_data = pd.read_table(self.tgt_file_path, header=None, encoding="utf-8", nrows=load_data_lines)
        self.src_tgt_layout_data = self.src_raw_data.merge(self.tgt_raw_data, left_index=True, right_index=True)
        self.src_tgt_layout_data.columns = ['src', 'tgt']

        self.pairs = [[normalizeString(s) for s in line] for line in self.src_tgt_layout_data.values]
        del self.src_raw_data, self.tgt_raw_data, self.src_tgt_layout_data

    def get_splited_data(self, random_state=None, train_val_test_split=0.2, val_test_split=0.5):
        '''
        return the splited data: train_pairs, val_pairs, test_pairs   of python list type
        '''
        if train_val_test_split in range(0,1):
            if random_state:
                self.train_pairs, val_test_pairs = train_test_split(self.pairs, test_size=train_val_test_split, random_state=random_state)
                self.val_pairs, self.test_pairs = train_test_split(val_test_pairs, test_size=val_test_split, random_state=random_state)
            else:
                self.train_pairs, val_test_pairs = train_test_split(self.pairs, test_size=train_val_test_split)
                self.val_pairs, self.test_pairs = train_test_split(val_test_pairs, test_size=val_test_split)
            del self.pairs # ?? sure ??
            return self.train_pairs, self.val_pairs, self.test_pairs
        else:
            # only for test purpose
            self.train_pairs = self.pairs
            self.val_pairs = self.pairs
            self.test_pairs = self.pairs
            self.test_only = True
    

class TextDataProcessor(DataReader):
    '''
    based on the DataReader, do text data processing.  
    vocab instance can be returned by build_vocab() method.
    a Dataset instance can be returned by get_dataset() method.
    '''
    def __init__(self, src_file_path, tgt_file_path) -> None:
        super(DataReader, self).__init__()
        pass

    @classmethod
    def from_DataReader(cls, data_reader):
        '''
        aims to build a TextDataProcessor instance from the DataReader instance
        '''
        a_text_data_processor = cls.__new__(cls)
        # copy the parents' attributes
        a_text_data_processor.__dict__ = data_reader.__dict__.copy()
        return a_text_data_processor
    

    def get_spacy_core(self, src_lang="fr", tgt_lang="en"):
        '''
        return the spacy core instance
        '''
        if hasattr(self, "spacy_src") and hasattr(self, "spacy_tgt"):
            return self.spacy_src, self.spacy_tgt
        else:
            try:
                self.spacy_src = spacy.load(src_lang+"_core_news_sm")
            except IOError:
                os.system("python -m spacy download "+src_lang+"_core_news_sm")
                self.spacy_src = spacy.load(src_lang+"_core_news_sm")
            
            try:
                self.spacy_tgt = spacy.load(tgt_lang+"_core_web_sm")
            except IOError:
                os.system("python -m spacy download "+tgt_lang+"_core_web_sm")
                self.spacy_tgt = spacy.load(tgt_lang+"_core_web_sm")
        
            return self.spacy_src, self.spacy_tgt
    
    
    def tokenize(self, text, spacy_core):
        '''
        return a list of tokens from a text
        paras:
            text: str
            spacy_core: create from spacy.load()
        '''
        return [tok.text for tok in spacy_core.tokenizer(text)]

    def yield_tokens(self, data_iter, tokenizer, index):
        '''
        a yield functions to meet the requirement of torchtext.vocab.build_vocab_from_iterator, as the input of parameter "iterator" in this function.
        params:
            data_iter: iterator of data(such as iterator of text data pairs)
        '''
        for one_pair in data_iter:
            yield tokenizer(one_pair[index])

    def build_vocab(self, src_lang="fr", tgt_lang="en", min_freq = 5, random_state=None):
        ''' 
        build the vocabulary for the source and target languages
        '''
        print("Building Source language Vocabulary ...")
        train_pairs, val_pairs, test_pairs = self.get_splited_data(random_state=random_state)
        spacy_src, spacy_tgt = self.get_spacy_core(src_lang=src_lang, tgt_lang=tgt_lang)
        self.vocab_src = build_vocab_from_iterator(
            self.yield_tokens(train_pairs + val_pairs + test_pairs, tokenizer=lambda text:self.tokenize(text, spacy_src), index=0),
            min_freq=min_freq,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        print("Building Target language Vocabulary ...")
        self.vocab_tgt = build_vocab_from_iterator(
            self.yield_tokens(train_pairs + val_pairs + test_pairs, tokenizer=lambda text:self.tokenize(text, spacy_tgt), index=1),
            min_freq=min_freq,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )
        self.vocab_src.set_default_index(self.vocab_src["<unk>"])
        self.vocab_tgt.set_default_index(self.vocab_tgt["<unk>"])

        print("Finished.\nVocabulary sizes:")
        print("Source: ", len(self.vocab_src))
        print("Target: ", len(self.vocab_tgt))
        return self.vocab_src, self.vocab_tgt
    

    def get_dataset(self):
        '''
        return the dataset instance
        '''
        if not hasattr(self, "test_only"):
            train_dataset = TranslationDataset([x[0] for x in self.train_pairs], [x[1] for x in self.train_pairs])
            val_dataset = TranslationDataset([x[0] for x in self.val_pairs], [x[1] for x in self.val_pairs])
            test_dataset = TranslationDataset([x[0] for x in self.test_pairs], [x[1] for x in self.test_pairs])
            return train_dataset, val_dataset, test_dataset
        else:
            # only for test purpose
            test_dataset = TranslationDataset([x[0] for x in self.test_pairs], [x[1] for x in self.test_pairs])
            return test_dataset
        
# test this file script  (can't run directly)

# data_reader = DataReader('data/fr-en_fr.txt', 'data/fr-en_en.txt')
# text_data_processor = TextDataProcessor.from_DataReader(data_reader)
# vocab_src, vocab_tgt =  text_data_processor.build_vocab(src_lang="fr", tgt_lang="en", min_freq=5, random_state=42)
# train_dataset, val_dataset, test_dataset = text_data_processor.get_dataset()
# print(f"train_dataset: {len(train_dataset)} val_dataset: {len(val_dataset)} test_dataset: {len(test_dataset)}")
