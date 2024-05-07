import torch 
import os
import pickle
import json
import pandas as pd


from utils_functions import load_config_file_json, load_text_sentences_file, load_trained_model
from data_processing.preprocessing import DataReader, TextDataProcessor
from data_processing.Dataloader import create_test_dataloader
from train_translation_task import Batch, subsequent_mask

def greedy_decode(model, a_batch, max_len, start_symbol):         
    '''
    predict a translation using greedy decoding
        a_batch must be a batch iterated from Batch object
    '''
    src = a_batch.src
    src_mask = a_batch.src_mask
    memory = model.encode(src, src_mask)
    batch_size = src.shape[0]
    ys = torch.zeros(batch_size, 1).fill_(start_symbol).type_as(src.data)   # tensor of prediction, and will be put into the model as tgt

    for i in range(max_len-1):    # max_len is the maximum padding of a sequence, do iteration to generate the whole sequence
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.shape[1]).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat(
            [ys, next_word.type_as(src)], dim=1
        )
    return ys

        
def infer_once(model_para_path, config_file_path, infer_sentences_file_path, ground_truth_file_path = None, src_lang="fr", tgt_lang="en", eos_string="</s>"):
    '''
    infer the translation of the sentences
        model_para_path: str, the path of the model parameters, a file with .pt extension
        config_file_path: str, the path of the configuration file
        infer_sentences_file_path: str, the path of the file with the sentences to be translated
    '''
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = load_config_file_json(config_file_path)
    model = load_trained_model(model_para_path, config).to(device=default_device)
    if ground_truth_file_path:
        test_data_reader = DataReader(infer_sentences_file_path, ground_truth_file_path)
    else:
        test_data_reader = DataReader(infer_sentences_file_path, None)
    test_text_data_processor = TextDataProcessor.from_DataReader(test_data_reader)
    
    check_file1 = os.path.exists("vocab_src.data")
    check_file2 = os.path.exists("vocab_tgt.data")
    check_file3 = os.path.exists("text_data_processor.data")

    if check_file1 and check_file2 and check_file3:
        print("Inference starts, loading data from files...")
        text_data_processor = pickle.load(open("text_data_processor.data", "rb"))
        vocab_src = pickle.load(open("vocab_src.data", "rb"))
        vocab_tgt = pickle.load(open("vocab_tgt.data", "rb"))       
        spacy_src, spacy_tgt = text_data_processor.get_spacy_core(src_lang=src_lang, tgt_lang=tgt_lang)
        print("Loaded vocabularies:\n")
    else:
        FileNotFoundError("the vocab_src.data or vocab_tgt.data or text_data_processor.data does not exist. Please train the model first.")

    test_text_data_processor.get_splited_data(random_state=None, train_val_test_split=1, val_test_split=1)  
    # create self.test_pairs attribute in the test_text_data_processor instance, no returns

    test_dataloader = create_test_dataloader(
        text_data_processor,
        test_text_data_processor, 
        default_device,
        src_lang = src_lang,
        tgt_lang = tgt_lang,
        max_padding=config["max_padding"],
        batch_size=config["batch_size"]
        )

    print(f"Reading test sentences from test_dataloader...")

    pad_idx = vocab_tgt["<blank>"]

    for i, b in enumerate(test_dataloader):
        bb = Batch(b[0], b[1], pad_idx)

        # run model to have prediction
        model_out = greedy_decode(model, bb, config["max_padding"], 0)
        for i, a_sequence in enumerate(model_out):
            src_tokens = [vocab_src.get_itos()[x] for x in bb.src[i] if x != pad_idx]
            print(
                "Input sentence: " + " ".join(src_tokens).replace("\n", "")
            )
            if ground_truth_file_path:
                tgt_tokens = [vocab_tgt.get_itos()[x] for x in bb.tgt[i] if x != pad_idx]
                print(
                    "Ground truth translation: " + " ".join(tgt_tokens).replace("\n", "")
                )
            txt_predicted = " ".join([vocab_tgt.get_itos()[x] for x in a_sequence if x != pad_idx]).split(eos_string, 1)[0]  # form a sentence predicted
            print("Model predictive translation: " + txt_predicted)

if __name__ == "__main__":

    infer_once("best_model.pt", "training_config.json", "test_sentences.txt")
    
