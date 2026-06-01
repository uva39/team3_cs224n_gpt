import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parents[1])

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


import random, numpy as np, argparse, torch
from types import SimpleNamespace

from config import SaveInfo
from classifier import train, test


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def make_save_info_from_config(args, config, dataset: str) -> SaveInfo:
    return SaveInfo(
        dataset=dataset,
        seed=getattr(args, "seed", None),
        fine_tune_mode=config.fine_tune_mode,
        lr=config.lr,
        epochs=config.epochs,
        batch_size=config.batch_size,
        hidden_dropout_prob=config.hidden_dropout_prob,

        checkpoint_path=config.filepath,
        prediction_prefix=getattr(args, "predictions_prefix", ""),

        dev_out=config.dev_out,
        test_out=config.test_out,
        summary_out=config.summary_out,
        metrics_out=config.metrics_out
    )

def make_config(args: argparse.Namespace, is_sst: bool) -> SimpleNamespace:
    dataset_slug = "sst" if is_sst else "cfimdb"

    filepath = args.sst_filepath if is_sst else args.cfimdb_filepath
    batch_size = args.sst_batch_size if is_sst else args.cfimdb_batch_size  # 기존 코드에서는 CFIMDB batch_size가 8로 고정

    prefix = getattr(args, "predictions_prefix", "")

    config = SimpleNamespace(
        dataset=dataset_slug,
        seed=args.seed,

        filepath=filepath,
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        
        train=f"data/ids-{dataset_slug}-train.csv",
        dev=f"data/ids-{dataset_slug}-dev.csv",
        test=f"data/ids-{dataset_slug}-test-student.csv",

        fine_tune_mode=args.fine_tune_mode,
        predictions_prefix=prefix,

        dev_out=f"predictions/{prefix}{args.fine_tune_mode}-{dataset_slug}-dev-out.csv",
        test_out=f"predictions/{prefix}{args.fine_tune_mode}-{dataset_slug}-test-out.csv",

        summary_out=f"runs/{prefix}{args.fine_tune_mode}-{dataset_slug}-summary.json",
        metrics_out=f"runs/{prefix}{args.fine_tune_mode}-{dataset_slug}-metrics.csv",
        
        max_grad_norm=args.max_grad_norm,
    )

    return config


def train_test(config, info):
  print('\n---------------------------------------------------')
  print(f'Training Sentiment Classifier on {info.dataset}...')
  print('---------------------------------------------------')
  train(config, info)

  print('\n---------------------------------------------------')
  print(f'Evaluating on {info.dataset}...')
  print('---------------------------------------------------')
  test(config)
  
  print('\ntraining and test process is end\n')
  
  
def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--sst-batch-size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
  parser.add_argument("--cfimdb-batch-size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--hidden-dropout-prob", type=float, default=0.3)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-3)

  parser.add_argument("--sst-filepath", default='sst-classifier.pt')
  parser.add_argument("--cfimdb-filepath", default='cfimdb-classifier.pt')
  parser.add_argument("--predictions-prefix", default='')
  
  parser.add_argument("--sst-record-name", default='sst_record.json')
  parser.add_argument("--cfimdb-record-name", default='cfimdb_record.json')
  
  parser.add_argument("--max-grad-norm", type=float, default=None)  # 1 사용해보자
  parser.add_argument("--weight-decay", type=float, default=0)
  
  parser.add_argument("--train-flag", type=int , help='0: sst and cfimdb both\n 1: sst only\n 2: cfimdb only', default=0)
  
  args = parser.parse_args()
  return args




def main():
  project_root = Path(__file__).resolve().parents[1]
  os.chdir(project_root)
  
  args = get_args()
  seed_everything(args.seed)
  
  if not args.train_flag in  (0, 1, 2):
    raise "incorect train_flag!!!!"
  
  if args.train_flag == 0 or args.train_flag == 1:
    sst_config = make_config(args, True)
    sst_info = make_save_info_from_config(args, sst_config, 'SST')
    train_test(sst_config, sst_info)
    sst_info.save(project_root / sst_config.summary_out)
  
  if args.train_flag == 0 or args.train_flag == 2:
    cfimdb_config = make_config(args, False)
    cfimdb_info = make_save_info_from_config(args, cfimdb_config, 'cfimdb')
    train_test(cfimdb_config, cfimdb_info)
    cfimdb_info.save(project_root / cfimdb_config.summary_out)




if __name__ == "__main__":
  main()