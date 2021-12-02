import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset, make_dataset_from_prefix
from fairseq.data import iterators, data_utils
import fairseq.utils as fairseq_utils
from model_gec.gec_bert import GecBertModel, LabelSmoothingLoss
from transformers import FlaubertTokenizer
from tag_encoder import TagEncoder

import os
import sys
from utils import EarlyStopping
import shutil
import argparse
import logging


def make_iterator(dataset, args, is_eval=False):
    with data_utils.numpy_seed(args.seed):
        indices = dataset.ordered_indices()

    # filter sentences too long
    indices, _ = dataset.filter_indices_by_size(indices, args.max_positions)

    # implicit batch size
    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        required_batch_size_multiple=args.required_batch_size_multiple,
    )

    dataiter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        num_workers=args.num_workers,
        epoch=1,  # starting epoch count
        buffer_size=1,  # default batch preloading size (not 0)
        skip_remainder_batch=is_eval,
        grouped_shuffling=True,  # shuffle batches and spread it equitably to GPUs
    )
    return dataiter


def load_data(args, tagger, tokenizer):
    logging.debug("LOADING \t" + args.data_path + ".train")
    train_dataset = make_dataset_from_prefix(
        args.data_path + ".train",
        tagger,
        tokenizer,
        ignore_clean=args.ignore_clean
    )
    train_iter = make_iterator(train_dataset, args)

    if args.valid:
        logging.debug("LOADING \t" + args.data_path + ".dev")
        valid_dataset = make_dataset_from_prefix(
            args.data_path + ".dev",
            tagger,
            tokenizer,
            ignore_clean=args.ignore_clean
        )
        valid_iter = make_iterator(valid_dataset, args, is_eval=True)
    else:
        valid_iter = None

    if args.test:
        logging.debug("LOADING \t" + args.data_path + ".test")
        test_dataset = make_dataset_from_prefix(
            args.data_path + ".test",
            tagger,
            tokenizer,
            ignore_clean=args.ignore_clean
        )
        test_iter = make_iterator(test_dataset, args, is_eval=True)
    else:
        test_iter = None

    return train_iter, valid_iter, test_iter


def train(args, device):

    tokenizer = FlaubertTokenizer.from_pretrained(args.tokenizer)
    tagger = TagEncoder(
        path_to_lex=args.path_to_lex,
        path_to_app=args.path_to_app
    )

    model = GecBertModel(
        len(tagger),
        tagger=tagger,
        tokenizer=tokenizer).to(device)
    train_iter, valid_iter, test_iter = load_data(args, tagger, tokenizer)
    criterion = LabelSmoothingLoss(smoothing=0.02)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.save, "tensorboard")
    if args.valid:
        stopper=EarlyStopping(patience=args.early_stopping)
    else:
        stopper=None

    torch.save(model, os.path.join(args.save, "model_best.pt"))

    num_iter=0
    for epoch in range(args.n_epochs):
        logging.debug("EPOCH " + str(epoch))
        train_bs=train_iter.next_epoch_itr(shuffle=True)
        if device == "cuda":
            torch.cuda.empty_cache()
        for batch in tqdm(train_bs):
            if device == "cuda":
                batch=fairseq_utils.move_to_cuda(batch)
                # logging.debug(torch.cuda.memory_allocated(device))
                if num_iter + 1 % 10 == 0:
                    torch.cuda.empty_cache()
            # logging.debug("-" * 72)
            model.train()
            criterion.train()
            #  TRAIN STEP
            optimizer.zero_grad()

            # logging.debug("noise data " +
            #               str(batch["noise_data"]["input_ids"].shape)
            #               )
            # logging.debug(batch["noise_data"]["attention_mask"].sum(-1))
            # logging.debug("tag data " +
            #               str(batch["tag_data"]["input_ids"].shape)
            #               )
            # logging.debug(batch["tag_data"]["attention_mask"].sum(-1))
            try:
                out=model(**batch["noise_data"])
                # logging.debug("tag out " + str(out["tag_out"].shape))
                # logging.debug(out["attention_mask"].sum(-1))

                sizes_out=out["attention_mask"].sum(-1)
                sizes_tgt=batch["tag_data"]["attention_mask"].sum(-1)
                coincide_mask=sizes_out == sizes_tgt

                out=out["tag_out"][coincide_mask][out["attention_mask"]
                                                    [coincide_mask].bool()]

                tgt=batch["tag_data"]["input_ids"][coincide_mask][
                    batch["tag_data"]["attention_mask"][coincide_mask].bool()
                ]
                # logging.debug("TAGs out = " + str(out.data.argmax(-1)[:20]))
                # logging.debug("TAGs tgt = " + str(tgt.data[:20]))
                # logging.debug("out \t" + str(out.shape))
                # logging.debug("tgt \t" + str(tgt.shape))

                # if not coincide_mask.all():
                #     logging.debug(tokenizer.convert_ids_to_tokens(batch["noise_data"][
                #         "input_ids"][~coincide_mask][0][batch["noise_data"]["attention_mask"][~coincide_mask][0].bool()]))

                loss=criterion(out, tgt)
                loss.backward()
                optimizer.step()
                del out, sizes_out, sizes_tgt, coincide_mask, tgt, batch
            except RuntimeError as e:

                if "out of memory" in str(e) and device == "cuda":
                    logging.info("OOM --- trying to recover by clearing cache")
                    torch.cuda.empty_cache()
                    continue
                    out=model(**batch["noise_data"])
                    # logging.debug("tag out " + str(out["tag_out"].shape))
                    # logging.debug(out["attention_mask"].sum(-1))

                    sizes_out=out["attention_mask"].sum(-1)
                    sizes_tgt=batch["tag_data"]["attention_mask"].sum(-1)
                    coincide_mask=sizes_out == sizes_tgt

                    out=out["tag_out"][coincide_mask][out["attention_mask"]
                                                        [coincide_mask].bool()]

                    tgt=batch["tag_data"]["input_ids"][coincide_mask][
                        batch["tag_data"]["attention_mask"][coincide_mask].bool()
                    ]
                    # logging.debug("TAGs out = " + str(out.data.argmax(-1)[:20]))
                    # logging.debug("TAGs tgt = " + str(tgt.data[:20]))
                    # logging.debug("out \t" + str(out.shape))
                    # logging.debug("tgt \t" + str(tgt.shape))

                    # if not coincide_mask.all():
                    #     logging.debug(tokenizer.convert_ids_to_tokens(batch["noise_data"][
                    #         "input_ids"][~coincide_mask][0][batch["noise_data"]["attention_mask"][~coincide_mask][0].bool()]))

                    loss=criterion(out, tgt)
                    loss.backward()
                    optimizer.step()
                    del out, sizes_out, sizes_tgt, coincide_mask, tgt, batch
                else:
                    raise e

            if num_iter == 0:
                torch.cuda.empty_cache()

            if args.tensorboard:
                writer.add_scalar(
                    os.path.join("Loss/train"),
                    loss.item(),
                    num_iter,
                )
                del loss

            # VALID STEP
            if (num_iter + 1) % args.valid_iter == 0:
                if args.valid:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    model.eval()
                    criterion.eval()
                    logging.info("VALIDATION at iter " + str(num_iter))
                    with torch.no_grad():
                        val_losses=list()
                        for valid_batch in tqdm(valid_iter.next_epoch_itr()):
                            if device == "cuda":
                                valid_batch=fairseq_utils.move_to_cuda(
                                    valid_batch)

                            out=model(**valid_batch["noise_data"])

                            sizes_out=out["attention_mask"].sum(-1)
                            sizes_tgt=valid_batch["tag_data"]["attention_mask"].sum(
                                -1)
                            coincide_mask=sizes_out == sizes_tgt

                            out=out["tag_out"][coincide_mask][out["attention_mask"]
                                                                [coincide_mask].bool()]

                            tgt=valid_batch["tag_data"]["input_ids"][coincide_mask][
                                valid_batch["tag_data"]["attention_mask"][coincide_mask].bool(
                                )
                            ]

                            val_loss=criterion(out, tgt).item()
                            val_losses.append(val_loss)
                        del valid_batch
                        val_loss=sum(val_losses) / len(val_losses)
                        writer.add_scalar(
                            os.path.join("Loss/valid"),
                            loss.item(),
                            num_iter,
                        )
                        stopper(val_loss)
                # Regular model save (checkpoint)
                torch.save(
                    model,
                    os.path.join(
                        args.save,
                        "model_{}.pt".format(num_iter))
                )
                # Update best model save if necessary
                if stopper and stopper.counter == 0:
                    logging.debug("NEW BEST MODEL")
                    shutil.copy2(
                        os.path.join(
                            args.save, "model_{}.pt".format(num_iter)), os.path.join(
                            args.save, "model_best.pt"), )
            if stopper and stopper.early_stop:
                break
            num_iter += 1
        if stopper and stopper.early_stop:
            break

    if args.test:
        model.eval()
        if device == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            test_losses=list()
            for test_batch in test_iter.next_epoch_itr():
                if device == "cuda":
                    test_batch=fairseq_utils.move_to_cuda(
                        test_batch)
                out=model(**test_batch["noise_data"])

                sizes_out=out["attention_mask"].sum(-1)
                sizes_tgt=test_batch["tag_data"]["attention_mask"].sum(-1)
                coincide_mask=sizes_out == sizes_tgt

                out=out["tag_out"][coincide_mask][out["attention_mask"]
                                                    [coincide_mask].bool()]

                tgt=test_batch["tag_data"]["input_ids"][coincide_mask][
                    test_batch["tag_data"]["attention_mask"][coincide_mask].bool()
                ]

                test_loss=criterion(out, tgt).item()
                test_losses.append(test_loss)
            test_loss=sum(test_losses) / len(test_losses)
            logging.info("MODEL test loss: " + str(test_loss))

    torch.save(model, os.path.join(args.save, "model_final.pt"))
    logging.info("TRAINING OVER")


def create_logger(logfile, loglevel):
    numeric_level=getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == "stderr":
        logging.basicConfig(
            format="[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            level=numeric_level,
        )
    else:
        logging.basicConfig(
            filename=logfile,
            format="[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            level=numeric_level,
        )
    logging.getLogger().setLevel(numeric_level)


if __name__ == "__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("data_path", help="Input bin data path")
    parser.add_argument("--save", required=True, help="save directory")
    # optional
    parser.add_argument("-v", action="store_true")
    parser.add_argument("--log", default="info", help="logging level")
    parser.add_argument(
        "--seed", type=int, default=0, help="Randomization seed",
    )
    parser.add_argument(
        "--path-to-lex", "--lex", required=True, help="Path to Lexique383.tsv",
    )
    parser.add_argument(
        "--path-to-app",
        "--app",
        required=True,
        help="Path to appendable words file.",
    )
    parser.add_argument(
        "--tokenizer",
        default="flaubert/flaubert_base_cased",
        help="Name of Huggingface tokenizer used.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers used to fetch data.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="If True: write metrics in tensorboard files.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="If True: Uses GPU resources available.",
    )
    parser.add_argument(
        "-lang", "--language", default="fr", help="language of the data"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens per batch.",
    )
    parser.add_argument(
        "--required-batch-size-multiple",
        type=int,
        default=8,
        help="batch size will be a multiplier of this value.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=100,
        help="Maximum number of sentences per batch.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=510,
        help="Maximum size of a tokenized sentence. Sentences too long will be forgotten.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=2,
        help="Number of epochs")
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate value.")
    parser.add_argument(
        "--valid",
        action="store_true",
        help="if True: regularly evaluate model on validation set.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="if True: evaluate model on test set after training.",
    )
    parser.add_argument(
        "--ignore-clean",
        action="store_true",
        help="Ignore clean in triplet (noise, tag, clean)",
    )
    parser.add_argument(
        "--valid-iter",
        type=int,
        default=512,
        help="Updates interval between two validation evaluations.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=5,
        help="Threshold number of consecutive validation scores not improved \
        to consider training over.",
    )

    args=parser.parse_args()
    create_logger(None, args.log)

    device="cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    train(args, device)
