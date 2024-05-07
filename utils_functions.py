import json
import os
import pickle
import torch 

from models.transformer_model_utils import make_model

def load_config_file_json(config_file_path):
    '''
    load the configuration file from the config_file_path
    '''
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"the config_file_path: {config_file_path} does not exist.")
    else:
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        f.close()
    
    # do the key word checking
    default_values = {
        "src_lang":"fr",
        "tgt_lang":"en",
        "file_prefix":"transformer_basd_translation_model",
        "distributed": False,

        "d_model": 512,
        "d_ff":2048,
        "heads":8,
        "dropout":0.1,
        "max_padding": 128,

        "N": 6,
        "num_epochs": 30,
        "num_workers": 4,
        "batch_size": 16,
        "base_lr": 0.0001,
        "warmup": 4000,
        "accum_iter": 1,
        "wait_epochs":5,

        "load_data_lines": 30000
    }
    for key, default_value in default_values.items():
        if key not in config:
            config[key] = default_value

    return config


def load_text_sentences_file(file_path):
    '''
    load the text sentences from the file
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"the file_path: {file_path} does not exist.")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = f.readlines()
        f.close()

    return sentences


def load_trained_model(model_para_path, config):
    '''
    load the trained model from the model_para_path
        model_para_path: str, the path of the model parameters, a file with .pt extension
        config: dict, the configuration of the model
    '''
    if not os.path.exists(model_para_path):
        raise FileNotFoundError(f"the model_para_path: {model_para_path} does not exist.")
    else:
        check_file_vocav_src = os.path.exists("vocab_src.data")
        check_file_vocav_tgt = os.path.exists("vocab_tgt.data")

        if check_file_vocav_src and check_file_vocav_tgt:
            vocab_src = pickle.load(open("vocab_src.data", "rb"))
            vocab_tgt = pickle.load(open("vocab_tgt.data", "rb"))
        else:
            raise FileNotFoundError("the vocab_src.data or vocab_tgt.data does not exist. Please train the model first.")
    model = make_model(len(vocab_src), len(vocab_tgt), N=config["N"], d_model=config["d_model"], d_ff=config["d_ff"], h=config["heads"], dropout=config["dropout"])
    model.load_state_dict(torch.load(model_para_path))

    return model