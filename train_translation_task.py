import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import GPUtil
import os
import torch.multiprocessing as mp
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle

# own modules
from models.transformer_model_utils import make_model
from regularization.LabelSmoothing import LabelSmoothing
from data_processing.Dataloader import create_dataloaders
from data_processing.preprocessing import DataReader, TextDataProcessor
from learning_rate_schedule.learning_rate_schedulers import get_lr_scheduler
from loss_functions.loss_functions import SimpleLossCompute
from EarlyStopping import EarlyStopping


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0    # return a tensor (1, size, size) with the lower triangle filled(include diagonal) with True

class Batch:
    '''
    Object for holding a batch of data with mask during training. 
    params:
        src: torch.tensor, source data from a batch interated from DataLoader
        tgt: torch.tensor, target data from a batch interated from DataLoader
    '''

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)   # example [16, 128]  -> [16, 1, 128]   in mask tensor the token that != pad will be set as True
        if tgt is not None:
            self.tgt = tgt[:, :-1]   # abandon <eos>
            self.tgt_y = tgt[:, 1:]  # abandon <sos>
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        '''
        Create a mask to hide padding and future words.
        '''
        tgt_mask = (tgt != pad).unsqueeze(-2)   # example tgt_mask.shape = [16, 1, 127]  [B, 1, max_padding-1]   in mask tensor the token that != pad will be set as True
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )      # for tgt_mask, we also need to forbidden the attention score of the future words that has not been predicted yet, so mask tensor's higher triangle will be set as False
        return tgt_mask   # example [16, 127, 127]
    

class TrainState:
    '''
    Track number of steps, examples, and tokens processed
    '''
    def __init__(self, config) -> None:
        self.step: int = 0  # Steps in the current epoch
        self.accum_step: int = 0  # Number of gradient accumulation steps
        self.samples: int = 0  # total # of examples used
        self.tokens: int = 0  # total # of tokens processed

        self.train_writer = SummaryWriter(comment=f'training__load_data_lines_{config["load_data_lines"]}__batch_size {config["batch_size"]}__dmodel_{config["d_model"]}'\
                                           f'num_epochs_{config["num_epochs"]}__max_padding_{config["max_padding"]}'\
                                            f'__base_lr_{config["base_lr"]}')
        self.val_writer = SummaryWriter(comment=f'validation__load_data_lines_{config["load_data_lines"]}__batch_size {config["batch_size"]}__dmodel_{config["d_model"]}'\
                                           f'num_epochs_{config["num_epochs"]}__max_padding_{config["max_padding"]}'\
                                            f'__base_lr_{config["base_lr"]}')

def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, train_state:TrainState, mode="train", accum_iter=1):
    ''' 
    Train a single epoch
    '''

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in tqdm(enumerate(data_iter), total=len(data_iter), desc=f"{mode}ing for one epoch..."):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )  # output of model example [16, 127, 512]
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()  # Update learning rate schedule

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def train_worker(gpu, ngpus_per_node, text_data_processor, vocab_src, vocab_tgt, spacy_src, spacy_tgt, config, is_distributed=False):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"] 
    d_model = config["d_model"]
    model = make_model(len(vocab_src), len(vocab_tgt), N=6, d_model=d_model)    
    model.cuda(gpu)
    module = model
    is_main_process = True
    early_stop = EarlyStopping(wait_epochs=config["wait_epochs"])

    if is_distributed:
        dist.init_process_group("nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node)
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)

    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        text_data_processor, 
        device=default_device,
        src_lang=config["src_lang"],
        tgt_lang=config["tgt_lang"],
        max_padding=config["max_padding"],
        batch_size=config["batch_size"],
        is_distributed=is_distributed
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler_config = {
        "lr_scheduler": "LambdaLR",
        "d_model": d_model,
        "warmup": config["warmup"],
    }
    lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_config)

    train_state = TrainState(config)

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch) # ????
        
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ...", flush=True)
        loss_for_one_epoch, train_state = run_epoch(
            [Batch(b[0], b[1], pad_idx) for b in train_dataloader],  # use Batch class to treat the real tensors to feed the model
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            train_state=train_state,
            mode="train+log",
            accum_iter=config["accum_iter"],
        )
        train_state.train_writer.add_scalar("Training loss", loss_for_one_epoch, epoch)

        GPUtil.showUtilization()
        # if is_main_process:
        #     # file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
        #     # torch.save(model.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"Training loss for epoch {epoch}: {loss_for_one_epoch}")

        print(f"[GPU{gpu}] Epoch {epoch} Validation ...", flush=True)
        model.eval()
        sloss = run_epoch(
            [Batch(b[0], b[1], pad_idx) for b in valid_dataloader],
            model, 
            SimpleLossCompute(module.generator, criterion),
            None,
            None,
            train_state=train_state,
            mode="eval",
        )
        # print(sloss)
        torch.cuda.empty_cache()

        print(f"validation loss for epoch {epoch}: {sloss[0]}")
        train_state.val_writer.add_scalar("Validation loss", sloss[0], epoch)
        if is_main_process:
            if early_stop.stop(sloss[0], model, metric_type="better_decrease"):
                break
        # break # for test

    # save test_dataloader for test 
    if test_dataloader:
        pickle.dump(data_reader, open("test_dataloader.data", "wb"))
        print("Saved test_dataloader to file")


def train_distribution_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    pass


def train_model(vocab_src, vocab_tgt, spacy_src, spacy_tgt, text_data_processor, config):
    '''
    function to actiavte training of the model
    params:
        vocab_src: dict, source vocabulary
        vocab_tgt: dict, target vocabulary
        spacy_src: spacy.lang, source language
        spacy_tgt: spacy.lang, target language
        data_reader: DataReader instance defined in preprocessing.py
        config: dict, configuration parameters 
    '''
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_src, spacy_tgt, config
        )
    else:
        train_worker(
            0, 1, text_data_processor, vocab_src, vocab_tgt, spacy_src, spacy_tgt, config, False
        )



if __name__ == "__main__":

    with open("training_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        # print(config)
        if config["distributed"]:
            print("Training in distributed mode")
        if config["src_file_path"] and config ["tgt_file_path"]:
            src_file_path = config["src_file_path"]
            tgt_file_path = config["tgt_file_path"]
        else:
            print("Please provide the source and target file paths in the configuration file")
            exit()
    f.close()

    check_file1 = os.path.exists("data_reader.data")
    check_file2 = os.path.exists("text_data_processor.data")
    check_file3 = os.path.exists("vocab_src.data")
    check_file4 = os.path.exists("vocab_tgt.data")

    if check_file1 and check_file2 and check_file3 and check_file4:
        print("Loading data from files ...")
        data_reader = pickle.load(open("data_reader.data", "rb"))
        text_data_processor = pickle.load(open("text_data_processor.data", "rb"))
        vocab_src = pickle.load(open("vocab_src.data", "rb"))
        vocab_tgt = pickle.load(open("vocab_tgt.data", "rb"))
        spacy_src, spacy_tgt = text_data_processor.get_spacy_core(src_lang="fr", tgt_lang="en")

        print("Loaded vocabularies:\n")
        print(f"Source: {len(text_data_processor.vocab_src)}")
        print(f"Target: {len(text_data_processor.vocab_tgt)}")
    else:
        data_reader = DataReader(src_file_path, tgt_file_path, config["load_data_lines"])
        text_data_processor = TextDataProcessor.from_DataReader(data_reader)
        vocab_src, vocab_tgt = text_data_processor.build_vocab(src_lang="fr", tgt_lang="en", min_freq=5, random_state=42)
        spacy_src, spacy_tgt = text_data_processor.get_spacy_core(src_lang="fr", tgt_lang="en")

        pickle.dump(data_reader, open("data_reader.data", "wb"))
        pickle.dump(text_data_processor, open("text_data_processor.data", "wb"))
        pickle.dump(vocab_src, open("vocab_src.data", "wb"))
        pickle.dump(vocab_tgt, open("vocab_tgt.data", "wb"))
        print("Saved variables to files \n")
    
    train_model(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_src=spacy_src,
        spacy_tgt=spacy_tgt,
        text_data_processor=text_data_processor,
        config=config
    )
