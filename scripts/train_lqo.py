
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, yaml, numpy as np, torch
from lqo.models.encoders.flat import FlatPlanEncoder
from lqo.models.heads.cost import CostHead
from lqo.models.lqo_model import LQOModel
from lqo.train.loop import PlanDataset, TrainConfig, train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/imdb_joblight.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    # Expect pre-featurized numpy files for simplicity
    X = np.load(cfg['train']['x_path'])
    y = np.load(cfg['train']['y_ms_path'])
    extra = np.load(cfg['train'].get('extra_path')) if cfg['train'].get('extra_path') else None
    Xv = np.load(cfg['valid']['x_path'])
    yv = np.load(cfg['valid']['y_ms_path'])
    extrav = np.load(cfg['valid'].get('extra_path')) if cfg['valid'].get('extra_path') else None

    enc = FlatPlanEncoder(in_dim=X.shape[1], hidden=cfg['model']['enc_hidden'], out_dim=cfg['model']['z_dim'])
    head_in = cfg['model']['z_dim'] + (extra.shape[1] if extra is not None else 0)
    head = CostHead(in_dim=head_in, hidden=cfg['model']['head_hidden'], log_target=True)
    model = LQOModel(enc, head)

    train_set = PlanDataset(X, y, extra)
    valid_set = PlanDataset(Xv, yv, extrav)
    train(model, train_set, valid_set, TrainConfig(**cfg['train_cfg']))

    out_path = cfg['out']['model_path']
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")

if __name__ == '__main__':
    main()
