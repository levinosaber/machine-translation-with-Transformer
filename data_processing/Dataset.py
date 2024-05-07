from torch.utils.data import Dataset 

class TranslationDataset(Dataset):
    def __init__(self, src_lang_data, tgt_lang_data):
        self.src = src_lang_data
        self.tgt = tgt_lang_data
        
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]