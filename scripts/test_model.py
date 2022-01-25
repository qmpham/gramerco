from transformers import FlaubertTokenizer, FlaubertModel
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset
from model_gec.gec_bert import GecBertModel, GecBert2DecisionsModel
from tag_encoder import TagEncoder, error_type_id, id_error_type
from tokenizer import WordTokenizer
import logging
import matplotlib.pyplot as plt
from noiser.Noise import Lexicon
import os
import sys
import re
from infer import apply_tags
from tqdm import tqdm
from train import make_iterator
from data.gramerco_dataset import GramercoDataset, make_dataset_from_prefix
import fairseq.utils as fairseq_utils


def load_bin_test(args, tagger, tokenizer):
    logging.debug("LOADING \t" + args.data_bin + ".test")
    test_dataset = make_dataset_from_prefix(
        args.data_bin + ".test",
        tagger,
        tokenizer,
        ignore_clean=args.ignore_clean,
    )
    test_iter = make_iterator(
        test_dataset,
        args,
        is_eval=True,
        max_sentences=50000,
    )
    return test_iter


def test(args):
    tokenizer = FlaubertTokenizer.from_pretrained(
        args.tokenizer
    )
    word_tokenizer = WordTokenizer(FlaubertTokenizer)
    lexicon = Lexicon(args.path_to_lex)
    tagger = TagEncoder(
        path_to_lex=args.path_to_lex,
        path_to_app=args.path_to_app,
    )

    if os.path.isfile(
        os.path.join(
            args.save_path,
            args.model_id,
            "model_{}.pt".format(args.model_iter),
        )
    ):
        path_to_model = os.path.join(
            args.save_path,
            args.model_id,
            "model_{}.pt".format(args.model_iter),
        )
    else:
        path_to_model = os.path.join(
            args.save_path,
            args.model_id,
            "model_best.pt",
        )
    if args.model_type == "normal":
        model = GecBertModel(
            len(tagger),
            tagger=tagger,
            tokenizer=tokenizer,
            mid=args.model_id,
        )
    elif args.model_type == "decision":
        model = GecBert2DecisionsModel(
            len(tagger),
            tagger=tagger,
            tokenizer=tokenizer,
            mid=args.model_id,
        )
    else:
        raise ValueError("Model type incorrect")
    device = "cuda:" + str(args.gpu_id) \
        if args.gpu and torch.cuda.is_available() else "cpu"
    if os.path.isfile(path_to_model):
        logging.info("loading model from " + path_to_model)
        map_loc = torch.device(device)
        state_dict = torch.load(path_to_model, map_location=map_loc)
        if isinstance(state_dict, GecBertModel):
            model = state_dict
        else:

            model.load_state_dict(
                state_dict["model_state_dict"]
            )

    else:
        logging.info("Model not found at: " + path_to_model)
        return
    model.eval()

    logging.info(torch.cuda.device_count())
    logging.info("device = " + device)
    logging.info(torch.version.cuda)
    model.to(device)

    if args.raw:
        with open(args.file_src, 'r') as f:
            txt_src = f.read(args.sample).split('\n')[:-1]

        with open(args.file_tag, 'r') as f:
            txt_tag = f.read(args.sample).split('\n')[:-1]

        txt_src = txt_src[:len(txt_tag)]
        txt_tag = txt_tag[:len(txt_src)]
        logging.info("contains {} sentences".format(len(txt_src)))
    else:
        test_iter = load_bin_test(args, tagger, tokenizer)

    FP = 0
    TP = 0
    FN = 0
    TN = 0
    num_tags = 0
    num_keeps = 0
    accs = np.zeros(len(id_error_type))
    lens = np.zeros(len(id_error_type))
    pred_tags = list()
    ref_tags = list()
    uu = 0
    if args.raw:
        for i in tqdm(range(len(txt_src) // args.batch_size + 1)):
            # if i > 0:
            #     print()
            batch_txt = txt_src[args.batch_size * i:
                                min(args.batch_size * (i + 1), len(txt_src))]
            batch_tag_ref = txt_tag[args.batch_size *
                                    i: min(args.batch_size * (i + 1), len(txt_tag))]
            # logging.info("noise >>> " + batch_txt[0])
            for j in range(args.num_iter):
                toks = tokenizer(
                    batch_txt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=510,
                ).to(device)
                del toks["token_type_ids"]
                for l in range(len(toks["input_ids"])):
                    if uu == 0:
                        logging.info(str(uu))
                        logging.info("noise >>> " + str(" ".join(map(
                            tokenizer._convert_id_to_token,
                            toks["input_ids"][l][toks["attention_mask"][l].bool()].cpu().numpy()
                        ))))
                        logging.info(str(toks["input_ids"][l][toks["attention_mask"][l].bool()].cpu().numpy()))
                        # logging.info("tag --- " + str(" ".join([
                        #     tagger.tag_to_id(tag) for tag in batch_tag_ref[l].split(" ")
                        # ])))
                        # ref_tag = torch.tensor()
                        # sys.exit(0)
                    uu += 1

                # logging.info(toks["input_ids"].shape)
                dec = list()
                with torch.no_grad():
                    # logging.info(" inputs ### " + str(toks))
                    out = model(**toks)  # tag_out, attention_mask
                    # logging.info(" outputs ### " + str(out))
                    for k, t in enumerate(batch_txt):

                        batch_txt[k] = apply_tags(
                            t,
                            out["decision_out"][k].argmax(-1)[
                                out["attention_mask"][k].bool()
                            ].cpu().bool(),
                            out["tag_out"][k].argmax(-1)[
                                out["attention_mask"][k].bool()
                            ].cpu(),
                            word_tokenizer,
                            tagger,
                            lexicon
                        )
                        if args.model_type == "decision":
                            pred = out["decision_out"][k][
                                out["attention_mask"][k].bool()
                            ].argmax(-1).cpu().bool()
                            dec.append(torch.softmax(
                                out["decision_out"][k][
                                    out["attention_mask"][k].bool()
                                ],
                                -1)[:, 1].max().item())
                            pred_tag = out["tag_out"][k][
                                out["attention_mask"][k].bool()
                            ].argmax(-1)
                        else:
                            yy = out["tag_out"][k][
                                out["attention_mask"][k].bool()
                            ]
                            yy = torch.softmax(yy, -1)
                            jj = yy.topk(3, dim=-1).indices.cpu()
                            ii = torch.arange(
                                jj.size(0)
                            ).unsqueeze(-1).expand(jj.shape)
                            # logging.info(jj)
                            # logging.info(yy[ii, jj])
                            # logging.info(batch_tag_ref[k].split(" "))
                            pred = jj[:, 0].bool()
                            pred_tag = jj[:, 0]

                        ref_tag = torch.tensor([tagger.tag_to_id(tag)
                                                for tag in batch_tag_ref[k].split(" ")])

                        ref = ref_tag.clone().bool()

                        logging.info("dec >>> " + str(pred.cpu().long().numpy()))
                        logging.info("tag >>> " + str(pred_tag.cpu()[pred.bool()].long().numpy()))

                        if len(pred) != len(ref):
                            continue
                        if len(pred_tag) != len(ref_tag):
                            continue
                        ref_tags.append(ref_tag)
                        pred_tags.append(pred_tag)
                        TP += ((pred == ref) & ref).long().sum().item()
                        TN += ((pred == ref) & ~ref).long().sum().item()
                        FN += ((pred != ref) & ref).long().sum().item()
                        FP += ((pred != ref) & ~ref).long().sum().item()
                        num_tags += ref.long().sum().item()
                        num_keeps += (~ref).long().sum().item()
                        pred_types = (
                            pred_tag[ref].clone().to("cpu") +
                            1).apply_(
                            tagger.get_tag_category).long()
                        ref_types = (ref_tag[ref].clone().to("cpu")).apply_(
                            tagger.get_tag_category).long()
                        for err_id in range(len(id_error_type)):
                            pred_types_i = pred_types[ref_types == err_id]
                            accs[err_id] += (pred_types_i == err_id).long(
                            ).sum().item()
                            lens[err_id] += len(pred_types_i)
    else:
        for i, test_batch in enumerate(tqdm(test_iter.next_epoch_itr(shuffle=False))):
            for k in range(len(test_batch["noise_data"]["input_ids"]) * 0 + 1):
                xi = test_batch["noise_data"]["input_ids"][k][
                    test_batch["noise_data"]["attention_mask"][k].bool()
                ]
                xi = " ".join(map(tokenizer._convert_id_to_token, xi.cpu().numpy()))
                logging.info("noise >>> " + xi)
                logging.info(str(test_batch["noise_data"]["input_ids"][k][
                    test_batch["noise_data"]["attention_mask"][k].bool()
                ].cpu().numpy()))
            # sys.exit(0)
            if i > 200:
                break
            # logging.info(toks["input_ids"].shape)
            dec = list()
            with torch.no_grad():
                if args.gpu:
                    test_batch = fairseq_utils.move_to_cuda(test_batch)
                # logging.info(" inputs ### " + str(test_batch["noise_data"]))
                out = model(**test_batch["noise_data"])  # tag_out, attention_mask
                # logging.info(" outputs ### " + str(out))
                sizes_out = out["attention_mask"].sum(-1)
                sizes_tgt = test_batch["tag_data"]["attention_mask"].sum(-1)
                coincide_mask = (sizes_out == sizes_out)
                if args.model_type == "decision":
                    pred = out["decision_out"][coincide_mask][
                        out["attention_mask"][coincide_mask].bool()
                    ].argmax(-1).cpu().bool()
                    logging.info("dec >>> " + str(pred.long().numpy()))
                    pred_tag = out["tag_out"][coincide_mask][
                        out["attention_mask"][coincide_mask].bool()
                    ].argmax(-1).cpu()

                else:
                    yy = out["tag_out"][coincide_mask][
                        out["attention_mask"][coincide_mask].bool()
                    ]
                    pred_tag = yy.argmax(-1).cpu()
                    pred = pred_tag.clone().bool()


                ref_tag = test_batch["tag_data"]["input_ids"][coincide_mask][
                    test_batch["tag_data"]["attention_mask"][coincide_mask].bool()
                ]
                ref = ref_tag.clone().bool().cpu()
                # logging.info("dec ref >>> " + str(ref.cpu().long().numpy()))
                logging.info("tag >>> " + str(" ".join(map(tagger.id_to_tag, pred_tag.cpu()[pred.bool()].long().numpy()))))

                ref_tags.append(ref_tag)
                pred_tags.append(pred_tag)
                # logging.info("pred shape = " + str(pred.shape))
                # logging.info("ref shape = " + str(ref.shape))
                TP += ((pred == ref) & ref).long().sum().item()
                TN += ((pred == ref) & ~ref).long().sum().item()
                FN += ((pred != ref) & ref).long().sum().item()
                FP += ((pred != ref) & ~ref).long().sum().item()
                num_tags += ref.long().sum().item()
                num_keeps += (~ref).long().sum().item()
                pred_types = (pred_tag[ref].clone().cpu() + 1).apply_(
                    tagger.get_tag_category).long()
                ref_types = (ref_tag[ref].clone().cpu()).apply_(
                    tagger.get_tag_category).long()
                for err_id in range(len(id_error_type)):
                    pred_types_i = pred_types[ref_types == err_id]
                    accs[err_id] += (pred_types_i == err_id).long(
                    ).sum().item()
                    lens[err_id] += len(pred_types_i)
    # pts = np.array([len(pt) for pt in pred_tags])
    # rts = np.array([len(pt) for pt in ref_tags])
    # print(pts[pts != rts])
    # print(rts[pts != rts])
    pred_tags = torch.cat(pred_tags).cpu()
    ref_tags = torch.cat(ref_tags).cpu()

    pred_tags_ = torch.cat((
        pred_tags[ref_tags.ne(0)] + 1,
        torch.arange(1, len(tagger) + 1)
    )).numpy()
    ref_tags_ = torch.cat((
        ref_tags[ref_tags.ne(0)],
        len(tagger) * torch.ones(len(tagger), dtype=ref_tags.dtype)
    )).numpy()
    matrix = confusion_matrix(
        ref_tags_,
        pred_tags_
    )[:-1, :-1]
    logging.info(str(matrix.shape))
    logging.info(len(tagger))
    num_true = matrix.diagonal()
    num_tot = matrix.sum(axis=1)
    logging.info("class accs  = " + str(num_true[num_tot != 0] / num_tot[num_tot != 0]))
    # logging.info("mean classes acc  = " + str((num_true[num_tot != 0] / num_tot[num_tot != 0]).mean()))
    logging.info("global acc  = " + str((num_true[num_tot != 0].sum() / num_tot[num_tot != 0].sum())))
    logging.info("class accurate  = " + str(num_true))
    logging.info("class total  = " + str(num_tot))

    pred_error_types = (pred_tags[ref_tags.ne(0)] +
                        1).apply_(tagger.get_tag_category).long()
    ref_error_types = (ref_tags[ref_tags.ne(0)]).apply_(
        tagger.get_tag_category).long()
    ConfusionMatrixDisplay.from_predictions(
        ref_error_types,
        pred_error_types,
        normalize="true",
        cmap="coolwarm",
        display_labels=id_error_type,
    )
    plt.savefig(os.path.join(
        args.save_path,
        args.model_id,
        "confusion_matrix_normalized.png",
    ))
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        ref_error_types,
        pred_error_types,
        cmap="coolwarm",
        display_labels=id_error_type,
    )
    plt.savefig(os.path.join(
        args.save_path,
        args.model_id,
        "confusion_matrix.png",
    ))

    # plt.figure()
    # ConfusionMatrixDisplay.from_predictions(
    #     ref_tags,
    #     pred_tags,
    #     normalize="true",
    #     cmap="coolwarm",
    # )
    # plt.savefig(os.path.join(
    #     args.save_path,
    #     args.model_id,
    #     "confusion_matrix_full.png",
    # ))

    # dec = np.array(dec)
    # logging.info("########## decision")
    # logging.info("dec mean = " + str(dec.mean()))
    # logging.info("dec med = " + str(np.median(dec)))
    # logging.info("dec Q1 = " + str(np.quantile(dec, 0.25)))
    # logging.info("dec std = " + str(np.std(dec)))
    # logging.info("dec Q3 = " + str(np.quantile(dec, 0.75)))
    # logging.info("dec max = " + str(np.max(dec)))
    # logging.info("dec min = " + str(np.min(dec)))
    logging.info("TP = " + str(TP))
    logging.info("TN = " + str(TN))
    logging.info("FN = " + str(FN))
    logging.info("FP = " + str(FP))
    logging.info("non identified errors = " + str(FN / (FN + TP) * 100))
    logging.info("non identified keep = " + str(FP / (FP + TN) * 100))
    logging.info("########## TAG")
    for err_id in range(len(id_error_type)):
        logging.info("acc " +
                     str(id_error_type[err_id]) +
                     " = " +
                     str(accs[err_id] /
                         lens[err_id]))
    logging.info("########## GLOBAL")
    logging.info("# keep = " + str(num_keeps))
    logging.info("# tag = " + str(num_tags))
    logging.info("prop tag = " + str(num_tags / (num_tags + num_keeps) * 100))
    for err_id in range(len(id_error_type)):
        logging.info("# err " +
                     str(id_error_type[err_id]) +
                     " = " +
                     str(int(lens[err_id])))
    # print('\n'.join(batch_txt))
    pli = np.stack((pred_tags[ref_tags.ne(0)] + 1,
                   ref_tags[ref_tags.ne(0)]), -1)
    np.savetxt("pli.csv", pli, delimiter=",", fmt="%d")


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(
            format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d_%H:%M:%S',
            level=numeric_level)
    else:
        logging.basicConfig(
            filename=logfile,
            format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d_%H:%M:%S',
            level=numeric_level)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--file-src', required=True, help="source file")
    parser.add_argument('--file-tag', required=True, help="tag file")

    # optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('--log', default="info", help='logging level')
    parser.add_argument(
        '--model-type',
        default='normal',
        help="Model architecture used.",
    )
    parser.add_argument(
        '--model-iter',
        type=int,
        default=-1,
        help="model iteration id: loading replaces best_model.pt by model_<iter>.pt ;"
        "negative or invalid will load best_model.pt",
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help="Number of samples tested from files (faster testing if files too large)",
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help="GPU usage activation.",
    )
    parser.add_argument(
        '--gpu-id',
        default=0,
        type=int,
        help="GPU id, generally 0 or 1.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='batch size for eval',
    )
    parser.add_argument(
        '--num-iter',
        type=int,
        default=1,
        help='num iteration loops to edit',
    )
    parser.add_argument(
        '--save-path',
        required=True,
        help='model save directory'
    )
    parser.add_argument(
        '--model-id',
        required=True,
        help="Model id (folder name)",
    )
    parser.add_argument(
        '--path-to-lex',
        '--lex',
        required=True,
        help='path to lexicon table.',
    )
    parser.add_argument(
        '--path-to-app',
        '--app',
        required=True,
        help="Path to appendable data.",
    )
    parser.add_argument(
        '--tokenizer',
        default="flaubert/flaubert_base_cased",
        help='model save directory',
    )
    parser.add_argument(
        '--min-positions',
        default=5,
        type=int,
        help='min token per sentence',
    )
    parser.add_argument(
        '--max-positions',
        default=510,
        type=int,
        help='max token per sentence',
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help='randomisation seed',
    )
    parser.add_argument(
        '--max-tokens',
        default=4096,
        type=int,
        help='max number of tokens per batch',
    )
    parser.add_argument(
        '--max-sentences',
        default=128,
        type=int,
        help='max number of sentences per batch',
    )
    parser.add_argument(
        '--required-batch-size-multiple',
        default=8,
        type=int,
        help='batch size multiplier',
    )
    parser.add_argument(
        '--num-workers',
        default=16,
        type=int,
        help='number of workers for data fetching',
    )
    parser.add_argument(
        '--ignore-clean',
        action="store_true",
        help='Ignore clean in triplet (noise, tag, clean)',
    )
    parser.add_argument(
        '--raw',
        action="store_true",
        help='Use raw data instead of bin ones'
    )
    parser.add_argument(
        '--data-bin',
        required=True,
        help='Path to data bin including basname'
    )

    args = parser.parse_args()

    create_logger("stderr", args.log)

    test(args)
