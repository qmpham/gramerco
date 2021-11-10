from torch.utils.data import DataLoader, Dataset
import logging
import torch


def type_from_string(s):
    if s == 'int64':
        return torch.int64
    if s == 'int32':
        return torch.int32
    if s == 'int16':
        return torch.int16
    if s == 'float16':
        return torch.float16
    if s == 'float32':
        return torch.float32
    if s == 'float64':
        return torch.float64

def shape_from_string(s):
    return tuple(map(int, s[1:-1].split(',')))


def data_from_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        logging.debug(data[:50])
        metadata = data[:50].decode('utf-8').rstrip().split('@')
        logging.debug("metadata: " + str(metadata))
        raw = data[50:]
        shape = shape_from_string(metadata[0])
        dtype = type_from_string(metadata[1])
        data = torch.frombuffer(raw, dtype=dtype).reshape(shape)
    return {'input_ids': data[0], 'attention_mask': data[1]}


class GramercoDataset(Dataset):
    def __init__(self, path, lang):

        self.clean_data = data_from_bin_file(path + '.{}.bin'.format(lang))
        self.noise_data = data_from_bin_file(path + '.noise.{}.bin'.format(lang))
        self.tag_data = data_from_bin_file(path + '.tag.{}.bin'.format(lang))

    def __getitem__(self, idx):
        return {'clean_data': {
                                'input_ids': self.clean_data['input_ids'][idx],
                                'attention_mask': self.clean_data['attention_mask'][idx]
                },
                'noise_data': {
                                'input_ids': self.clean_data['input_ids'][idx],
                                'attention_mask': self.clean_data['attention_mask'][idx]
                },
                'tag_data': {
                                'input_ids': self.tag_data['input_ids'][idx],
                                'attention_mask': self.tag_data['attention_mask'][idx]
                }
                }

    def __len__(self):
        return len(self.tag_data['input_ids'])


def load_data(args):
    data = {}

    if args.train_path:
        train_set = GramercoDataset(args.train_path, args.lang)
        train = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
        data["train"] = train

    if args.dev_path:
        dev_set = GramercoDataset(args.dev_path, args.lang)
        dev = DataLoader(dev_set, shuffle=True, batch_size=args.batch_size)
        data["dev"] = dev

    if args.test_path:
        test_set = GramercoDataset(args.test_path, args.lang)
        test = DataLoader(test_set, batch_size=args.batch_size)
        data["test"] = test

    return data

def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)


if __name__ == "__main__":
    create_logger('stderr', 'DEBUG')
    dataset = GramercoDataset('/home/bouthors/workspace/gramerco-repo/gramerco/resources/bin/data', 'fr')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

    for d in iter(dataloader):
        logging.info(d['tag_data']['input_ids'].shape)

    logging.info(dataset[0]['tag_data']['input_ids'].shape)
    logging.info(len(dataset))
