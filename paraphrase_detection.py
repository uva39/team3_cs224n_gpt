'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import csv
import math
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

class LoRALinear(nn.Module):
  """Linear layer with a frozen base projection plus a trainable LoRA update."""

  def __init__(self, base_layer, rank, alpha, dropout=0.):
    super().__init__()
    if rank <= 0:
      raise ValueError(f"LoRA rank must be positive, got {rank}")

    self.base_layer = base_layer
    self.rank = rank
    self.alpha = alpha
    self.scaling = alpha / rank
    self.dropout = nn.Dropout(dropout)
    self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
    self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)

    for param in self.base_layer.parameters():
      param.requires_grad = False

    nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_b.weight)

  def forward(self, x):
    return self.base_layer(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def parse_lora_targets(targets):
  return {target.strip() for target in targets.split(',') if target.strip()}


def apply_lora_to_gpt(gpt, rank, alpha, dropout, targets):
  applied = []

  def replace(parent, name, label):
    layer = getattr(parent, name)
    if not isinstance(layer, nn.Linear):
      raise TypeError(f"LoRA target {label} is not nn.Linear")
    setattr(parent, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
    applied.append(label)

  for i, layer in enumerate(gpt.gpt_layers):
    if 'query' in targets:
      replace(layer.self_attention, 'query', f'gpt_layers.{i}.self_attention.query')
    if 'key' in targets:
      replace(layer.self_attention, 'key', f'gpt_layers.{i}.self_attention.key')
    if 'value' in targets:
      replace(layer.self_attention, 'value', f'gpt_layers.{i}.self_attention.value')
    if 'attention_dense' in targets:
      replace(layer, 'attention_dense', f'gpt_layers.{i}.attention_dense')
    if 'interm_dense' in targets:
      replace(layer, 'interm_dense', f'gpt_layers.{i}.interm_dense')
    if 'out_dense' in targets:
      replace(layer, 'out_dense', f'gpt_layers.{i}.out_dense')

  return applied


def parameter_count(model, trainable_only=False):
  return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.dropout = nn.Dropout(args.dropout)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model. LoRA mode freezes the base GPT-2
    # weights and trains only LoRA adapters plus the classification head.
    for param in self.gpt.parameters():
      param.requires_grad = not args.freeze_gpt

    if args.use_lora:
      targets = parse_lora_targets(args.lora_targets)
      args.lora_applied_modules = apply_lora_to_gpt(
        self.gpt,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=targets
      )

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    gpt_out = self.gpt(input_ids, attention_mask)
    last_hidden = gpt_out['last_token']
    logits = self.paraphrase_detection_head(self.dropout(last_hidden))
    return logits



def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def log_epoch_metrics(args, epoch, train_loss, train_acc, train_f1, dev_acc, dev_f1, best_dev_acc, is_best):
  fieldnames = [
    'dataset', 'epoch', 'seed', 'model_size', 'lr', 'batch_size', 'weight_decay', 'dropout',
    'use_lora', 'freeze_gpt', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_targets',
    'resume_from', 'start_epoch', 'trainable_parameters', 'total_parameters',
    'train_loss', 'train_acc', 'train_f1', 'dev_acc', 'dev_f1',
    'best_dev_acc', 'is_best', 'checkpoint_path'
  ]
  row = {
    'dataset': 'quora',
    'epoch': epoch,
    'seed': args.seed,
    'model_size': args.model_size,
    'lr': args.lr,
    'batch_size': args.batch_size,
    'weight_decay': args.weight_decay,
    'dropout': args.dropout,
    'use_lora': args.use_lora,
    'freeze_gpt': args.freeze_gpt,
    'lora_rank': args.lora_rank,
    'lora_alpha': args.lora_alpha,
    'lora_dropout': args.lora_dropout,
    'lora_targets': args.lora_targets,
    'resume_from': args.resume_from,
    'start_epoch': args.start_epoch,
    'trainable_parameters': args.trainable_parameters,
    'total_parameters': args.total_parameters,
    'train_loss': train_loss,
    'train_acc': train_acc,
    'train_f1': train_f1,
    'dev_acc': dev_acc,
    'dev_f1': dev_f1,
    'best_dev_acc': best_dev_acc,
    'is_best': is_best,
    'checkpoint_path': args.filepath,
  }

  metrics_dir = os.path.dirname(args.metrics_out)
  if metrics_dir:
    os.makedirs(metrics_dir, exist_ok=True)

  write_header = not os.path.exists(args.metrics_out)
  with open(args.metrics_out, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()
    writer.writerow(row)


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)
  args.total_parameters = parameter_count(model)
  args.trainable_parameters = parameter_count(model, trainable_only=True)
  print(
    f"trainable parameters :: {args.trainable_parameters} / {args.total_parameters} "
    f"({args.trainable_parameters / args.total_parameters:.2%})"
  )
  if args.use_lora:
    print(
      f"LoRA enabled :: rank={args.lora_rank}, alpha={args.lora_alpha}, "
      f"dropout={args.lora_dropout}, targets={args.lora_targets}"
    )
    print(f"LoRA modules :: {len(args.lora_applied_modules)}")

  lr = args.lr
  optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=args.weight_decay)
  best_dev_acc = 0
  start_epoch = args.start_epoch if args.start_epoch is not None else 0

  if args.resume_from:
    saved = torch.load(args.resume_from, weights_only=False)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optim'])
    if args.start_epoch is None:
      start_epoch = getattr(saved['args'], 'epochs', 0)
    args.start_epoch = start_epoch
    if 'system_rng' in saved:
      random.setstate(saved['system_rng'])
    if 'numpy_rng' in saved:
      np.random.set_state(saved['numpy_rng'])
    if 'torch_rng' in saved:
      torch.random.set_rng_state(saved['torch_rng'])
    print(f"resumed training from {args.resume_from} at epoch {start_epoch}")

  checkpoint_dir = os.path.dirname(args.filepath)
  if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

  # Run for the specified number of epochs.
  for epoch in range(start_epoch, args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    train_y_true = []
    train_y_pred = []
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1
      train_y_true.extend(labels.detach().cpu().tolist())
      train_y_pred.extend(preds.detach().cpu().tolist())

    train_loss = train_loss / num_batches
    train_acc = accuracy_score(train_y_true, train_y_pred)
    train_f1 = f1_score(train_y_true, train_y_pred, average='macro')

    previous_best_dev_acc = best_dev_acc
    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
    is_best = dev_acc > best_dev_acc

    if is_best:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    log_epoch_metrics(args, epoch, train_loss, train_acc, train_f1, dev_acc, dev_f1, previous_best_dev_acc, is_best)

    print(
      f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, "
      f"train f1 :: {train_f1 :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}, "
      f"is best :: {is_best}"
    )


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath, weights_only=False)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
  parser.add_argument("--metrics_out", type=str, default="logs/paraphrase_experiments.csv")
  parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
  parser.add_argument("--experiment_name", type=str, default="base-gpt2")
  parser.add_argument("--filepath", type=str, default=None)
  parser.add_argument("--resume_from", type=str, default=None)
  parser.add_argument("--start_epoch", type=int, default=None)

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--weight_decay", type=float, default=0.)
  parser.add_argument("--dropout", type=float, default=0.)
  parser.add_argument("--use_lora", action='store_true')
  parser.add_argument("--freeze_gpt", action='store_true')
  parser.add_argument("--lora_rank", type=int, default=8)
  parser.add_argument("--lora_alpha", type=float, default=16.)
  parser.add_argument("--lora_dropout", type=float, default=0.)
  parser.add_argument(
    "--lora_targets",
    type=str,
    default="query,value",
    help="Comma-separated GPT-2 linear modules: query,key,value,attention_dense,interm_dense,out_dense"
  )
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  if args.use_lora:
    args.freeze_gpt = True
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  if args.filepath is None:
    experiment_name = args.experiment_name
    if args.use_lora:
      targets = args.lora_targets.replace(',', '-')
      experiment_name = f'{experiment_name}-lora-r{args.lora_rank}-a{args.lora_alpha:g}-{targets}'
    args.filepath = os.path.join(
      args.checkpoint_dir,
      f'{experiment_name}-lr{args.lr:g}-ep{args.epochs}-bs{args.batch_size}-wd{args.weight_decay:g}-drop{args.dropout:g}-quora.pt'
    )
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
