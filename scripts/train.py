import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from data.data import GramercoDataset
from model_gec.gec_bert import GecBertModel, LabelSmoothingLoss
from tag_encoder import TagEncoder
import logging
import os
import sys
from utils import EarlyStopping
import shutil


def load_data(args):

    train_dataset = GramercoDataset(args.data_path + ".train", args.language)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

    if args.valid:
        valid_dataset = GramercoDataset(args.data_path + ".dev", args.language)
        valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=4)
    else:
        valid_dataloader = None

    if args.test:
        test_dataset = GramercoDataset(args.data_path + ".test", args.language)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4)
    else:
        test_dataloader = None

    return train_dataloader, valid_dataloader, test_dataloader


def train(args):
    tagger = TagEncoder()
    train_dataloader, valid_dataloader, test_dataloader = load_data(args)
    criterion = LabelSmoothingLoss(smoothing=0.02)
    model = GecBertModel(len(tagger))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.tensorboard:
        writer = SummaryWriter()
    if args.valid:
        stopper = EarlyStopping(patience=args.early_stopping)

    torch.save(model, os.path.join(args.save, "model_best.pt"))

    num_iter = 0
    for epoch in tqdm(range(args.n_epochs)):
        train = iter(train_dataloader)
        for batch in train:
            #  TRAIN STEP
            optimizer.zero_grad()

            out = model(**batch["noise_data"])

            out = out["tag_out"][out["attention_mask"].bool()]
            tgt = batch["tag_data"]["input_ids"][
                batch["tag_data"]["attention_mask"].bool()
            ]
            logging.debug("TAGs out = " + str(out.data.argmax(-1)[:20]))
            logging.debug("TAGs tgt = " + str(tgt.data[:20]))
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()

            if args.tensorboard:
                writer.add_scalar(
                    os.path.join(args.save, "tensorboard", "loss (train)"),
                    loss.item(),
                    num_iter,
                )

            # VALID STEP
            if (num_iter - 1) % args.valid_iter == 0:
                if args.valid:
                    logging.info("VALIDATION at iter " + str(num_iter))
                    with torch.no_grad():
                        val_losses = list()
                        for valid_batch in valid_dataloader:
                            out = model(**valid_batch["noise_data"])

                            out = out["tag_out"][out["attention_mask"].bool()]
                            tgt = valid_batch["tag_data"]["input_ids"][
                                valid_batch["tag_data"]["attention_mask"].bool()
                            ]
                            val_loss = criterion(out, tgt).item()
                            val_losses.append(val_loss)
                        val_loss = sum(val_losses) / len(val_losses)
                        writer.add_scalar(
                            os.path.join(args.save, "tensorboard", "loss (valid)"),
                            loss.item(),
                            num_iter,
                        )
                        stopper(val_loss)
                # Regular model save (checkpoint)
                torch.save(
                    model, os.path.join(args.save, "model_{}.pt".format(num_iter))
                )
                # Update best model save if necessary
                if stopper.counter == 0:
                    shutil.copy2(
                        os.path.join(args.save, "model_{}.pt".format(num_iter)),
                        os.path.join(args.save, "model_best.pt"),
                    )
            if stopper.early_stop:
                break
            num_iter += 1
        if stopper.early_stop:
            break

    if args.test:
        with torch.no_grad():
            test_losses = list()
            for test_batch in test_dataloader:
                out = model(**test_batch["noise_data"])

                out = out["tag_out"][out["attention_mask"].bool()]
                tgt = valid_batch["tag_data"]["input_ids"][
                    test_batch["tag_data"]["attention_mask"].bool()
                ]
                test_loss = criterion(out, tgt).item()
                test_losses.append(test_loss)
            test_loss = sum(test_losses) / len(test_losses)
            logging.info("MODEL test loss: " + str(test_loss))

    torch.save(model, os.path.join(args.save, "model_final.pt"))
    logging.info("TRAINING OVER")


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", help="Input bin data path")
    parser.add_argument("--save", required=True, help="save directory")
    # optional
    parser.add_argument("-v", action="store_true")
    parser.add_argument("-log", default="info", help="logging level")
    parser.add_argument(
        "-lang", "--language", default="fr", help="language of the data"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size for the training"
    )
    parser.add_argument("--n-epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=0.001, help="Learning rate value."
    )
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
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="If True: write metrics in tensorboard files.",
    )

    args = parser.parse_args()

    create_logger("stderr", args.log)

    train(args)
