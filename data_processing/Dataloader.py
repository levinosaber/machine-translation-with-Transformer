import torch
from torch.nn.functional import pad
from torchtext.data.functional import to_map_style_dataset
import torchtext.datasets as datasets
from torch.utils.data.distributed import DistributedSampler  # for distributed training
from torch.utils.data import DataLoader

from data_processing.preprocessing import DataReader, TextDataProcessor  # self-defined classes


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def collate_func(batch, src_vocab, src_tokenizer, tgt_vocab, tgt_tokenizer, device, max_padding=128, pad_idx=1):
    '''
    batch is a list of tuples, with each tuple containing a pair of tensors like
    (a_src_data, a_target_data)
    '''
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device) # </s> token id
    src_list, tgt_list = [], []
    for (a_src_data, a_tgt_data) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(tokenize(a_src_data, tokenizer=src_tokenizer)),
                    dtype=torch.int64,
                    device=device
                ),
                eos_id
            ],
            0
        )

        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tokenize(a_tgt_data, tokenizer=tgt_tokenizer)),
                    dtype=torch.int64,
                    device=device
                ),
                eos_id
            ],
            0
        )

        src_list.append(
            pad(processed_src, (0, max_padding - len(processed_src)), value=pad_idx)
        )
        tgt_list.append(
            pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_idx)
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    
    return (src, tgt)


def create_dataloaders(text_data_processor: TextDataProcessor, device, src_lang = "fr", tgt_lang = "en", max_padding = 128, batch_size=16, is_distributed=False):
    if not hasattr(text_data_processor, "vocab_src") or not hasattr(text_data_processor, "vocab_tgt"):
        src_vocab,tgt_vocab = text_data_processor.build_vocab(src_lang=src_lang, tgt_lang=tgt_lang)
    else:
        src_vocab = text_data_processor.vocab_src
        tgt_vocab = text_data_processor.vocab_tgt

    train_dataset, val_dataset, test_dataset = text_data_processor.get_dataset()

    # for distributed training
    train_sampler = DistributedSampler(to_map_style_dataset(train_dataset)) if is_distributed else None
    valid_sampler = DistributedSampler(to_map_style_dataset(val_dataset)) if is_distributed else None
    
    # collate_fn 
    def collate_fn(batch):
        '''
        batch is a list of tuples, with each tuple containing a pair of tensors like
        (data, target)
        '''
        src_tokenizer, tgt_tokenizer = text_data_processor.get_spacy_core(src_lang=src_lang, tgt_lang=tgt_lang)
        return collate_func(
            batch, 
            src_vocab, 
            src_tokenizer,
            tgt_vocab, 
            tgt_tokenizer,
            device, 
            max_padding, 
            pad_idx=src_vocab.get_stoi()["<blank>"]
            )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def create_test_dataloader(text_data_processor: TextDataProcessor, test_text_data_processor: TextDataProcessor, device, src_lang = "fr", tgt_lang = "en", max_padding = 128, batch_size=16):
    if not hasattr(text_data_processor, "vocab_src") or not hasattr(text_data_processor, "vocab_tgt"):
        src_vocab,tgt_vocab = text_data_processor.build_vocab(src_lang=src_lang, tgt_lang=tgt_lang)
    else:
        src_vocab = text_data_processor.vocab_src
        tgt_vocab = text_data_processor.vocab_tgt

    test_dataset = test_text_data_processor.get_dataset()

    # collate_fn 
    def collate_fn(batch):
        '''
        batch is a list of tuples, with each tuple containing a pair of tensors like
        (data, target)
        '''
        src_tokenizer, tgt_tokenizer = text_data_processor.get_spacy_core(src_lang=src_lang, tgt_lang=tgt_lang)
        return collate_func(
            batch, 
            src_vocab, 
            src_tokenizer,
            tgt_vocab, 
            tgt_tokenizer,
            device, 
            max_padding, 
            pad_idx=src_vocab.get_stoi()["<blank>"]
            )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return test_loader