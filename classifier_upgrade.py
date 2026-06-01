#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB
'''

import os
import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  This module performs sentiment classification using GPT2 in a cloze-style (fill-in-the-blank) task.

  In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
  Thus, your forward() should return one logit for each of the 5 classes.
  '''

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()

    # Pretrain mode does not require updating GPT paramters.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    ### YOUR CODE HERE
    if getattr(config, "use_simple_classifier", True):
      self.classifier = torch.nn.Linear(self.gpt.config.hidden_size, self.num_labels)
    else:
      self.classifier = torch.nn.Sequential(
        torch.nn.Dropout(config.hidden_dropout_prob),
        torch.nn.Linear(self.gpt.config.hidden_size, self.gpt.config.hidden_size),
        torch.nn.GELU(),
        torch.nn.Dropout(config.hidden_dropout_prob),
        torch.nn.Linear(self.gpt.config.hidden_size, self.num_labels)
    )
    #raise NotImplementedError


  def forward(self, input_ids, attention_mask):
    '''Takes a batch of sentences and returns logits for sentiment classes'''

    ### TODO: The final GPT contextualized embedding is the hidden state of the last token.
    ###       HINT: You should consider what is an appropriate return value given that
    ###       the training loop currently uses F.cross_entropy as the loss function.
    ### YOUR CODE HERE
    x = self.gpt(input_ids, attention_mask)['last_hidden_state']

    last_token_idx = attention_mask.sum(dim=1) - 1    # 마지막 실제 토큰 위치 찾기
    batch_idx = torch.arange(x.size(0), device=x.device)

    x = x[batch_idx, last_token_idx, :]
    logits = self.classifier(x)

    return logits
    #raise NotImplementedError



class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  if flag == 'test':
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def init_metrics_csv(filepath):
  dirpath = os.path.dirname(filepath)
  if dirpath:
    os.makedirs(dirpath, exist_ok=True)

  fieldnames = [
    'dataset',
    'epoch',
    'seed',
    'fine_tune_mode',
    'lr',
    'batch_size',
    'hidden_dropout_prob',
    'train_loss',
    'train_acc',
    'train_f1',
    'dev_acc',
    'dev_f1',
    'best_dev_acc',
    'is_best',
    'checkpoint_path'
  ]

  with open(filepath, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()


def append_metrics_csv(filepath, row):
  with open(filepath, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writerow(row)
    
def train(args, save_info = None):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)
  
  # Init model.
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': 768,
            'data_dir': '.',
            'fine_tune_mode': args.fine_tune_mode,
            'use_simple_classifier': args.use_simple_classifier
            }

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=getattr(args, "weight_decay", 0))
  
  total_steps = args.epochs * len(train_dataloader)
  warmup_steps = int(total_steps * getattr(args, "warmup_ratio", 0.06))
  
  
  scheduler = None
  if not getattr(args, "unuse_schedule", False):
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
  
  best_dev_acc = -1
  
  metrics_out = getattr(args, "metrics_out", None)
  if metrics_out is not None:
    init_metrics_csv(metrics_out)

  
  train_acc, train_f1 = None, None
  dev_acc, dev_f1 = None, None
  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')

      loss.backward()
      if getattr(args, 'max_grad_norm', None) is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
      optimizer.step()
      if scheduler is not None:
        scheduler.step()
        
      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / (num_batches)

    train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

    is_best = dev_acc > best_dev_acc
    
    if is_best:
      best_dev_acc = dev_acc
    
      if save_info is not None:
        save_info.update(
          best_epoch=epoch,
          best_dev_acc=float(dev_acc),
          best_dev_f1=float(dev_f1)
        )
        
      save_model(model, optimizer, args, config, args.filepath)
      
    if metrics_out is not None:
      append_metrics_csv(metrics_out, {
        'dataset': getattr(args, "dataset", ""),
        'epoch': epoch,
        'seed': getattr(args, "seed", ""),
        'fine_tune_mode': args.fine_tune_mode,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'dev_acc': dev_acc,
        'dev_f1': dev_f1,
        'best_dev_acc': best_dev_acc,
        'is_best': is_best,
        'checkpoint_path': args.filepath
      })
  
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
  
  if save_info is not None:
    save_info.update(
      final_train_acc=float(train_acc),
      final_train_f1=float(train_f1),
      final_dev_acc=float(dev_acc),
      final_dev_f1=float(dev_f1),
    )
    
  if hasattr(args, "final_filepath"):
    save_model(model, optimizer, args, config, args.final_filepath)


def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, map_location=device)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print('DONE DEV')

    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-3)

  parser.add_argument("--sst-filepath", default='sst-classifier.pt')
  parser.add_argument("--cfimdb-filepath", default='cfimdb-classifier.pt')
  parser.add_argument("--predictions-prefix", default='')
  
  args = parser.parse_args()
  return args
  

if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  
  print('Training Sentiment Classifier on SST...')
  config = SimpleNamespace(
#    filepath='sst-classifier.pt',
    filepath=args.sst_filepath,
    lr=args.lr,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=args.batch_size,
    hidden_dropout_prob=args.hidden_dropout_prob,
    train='data/ids-sst-train.csv',
    dev='data/ids-sst-dev.csv',
    test='data/ids-sst-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out='predictions/' + args.predictions_prefix + args.fine_tune_mode + '-sst-dev-out.csv',
    test_out='predictions/' + args.predictions_prefix + args.fine_tune_mode + '-sst-test-out.csv'
  )

  train(config)

  print('Evaluating on SST...')
  test(config)

###############################################################

  print('Training Sentiment Classifier on cfimdb...')
  config = SimpleNamespace(
 #   filepath='cfimdb-classifier.pt',
    filepath=args.cfimdb_filepath,
    lr=args.lr,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=8,
    hidden_dropout_prob=args.hidden_dropout_prob,
    train='data/ids-cfimdb-train.csv',
    dev='data/ids-cfimdb-dev.csv',
    test='data/ids-cfimdb-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out='predictions/' + args.predictions_prefix + args.fine_tune_mode + '-cfimdb-dev-out.csv',
    test_out='predictions/' + args.predictions_prefix + args.fine_tune_mode + '-cfimdb-test-out.csv'
  )
  
  train(config)

  print('Evaluating on cfimdb...')
  test(config)
