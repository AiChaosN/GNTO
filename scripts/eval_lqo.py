
#!/usr/bin/env python3
from __future__ import annotations
import argparse, yaml, numpy as np, torch
from lqo.models.encoders.flat import FlatPlanEncoder
from lqo.models.heads.cost import CostHead
from lqo.models.lqo_model import LQOModel
from lqo.train.metrics import q_error, selected_runtime, surpassed_plans, rank_corr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/imdb_joblight.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))

    X = np.load(cfg['test']['x_path'])
    y = np.load(cfg['test']['y_ms_path'])
    extra = np.load(cfg['test'].get('extra_path')) if cfg['test'].get('extra_path') else None

    enc = FlatPlanEncoder(in_dim=X.shape[1], hidden=cfg['model']['enc_hidden'], out_dim=cfg['model']['z_dim'])
    head_in = cfg['model']['z_dim'] + (extra.shape[1] if extra is not None else 0)
    head = CostHead(in_dim=head_in, hidden=cfg['model']['head_hidden'], log_target=True)
    model = LQOModel(enc, head)
    state = torch.load(cfg['out']['model_path'], map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        extra_t = torch.tensor(extra, dtype=torch.float32) if extra is not None else None
        pred_log = model(X_t, extra_t).numpy()
        pred = np.exp(pred_log)  # invert log

    q50, q90, q99 = q_error(pred, y)
    print(f"Q50={q50:.3f} Q90={q90:.3f} Q99={q99:.3f}")

    # If the file encodes multiple alternative plans per query, you can compute selection metrics
    # For demo, we assume contiguous blocks of K plans per query:
    if 'k_per_query' in cfg['test']:
        k = int(cfg['test']['k_per_query'])
        assert len(X) % k == 0
        sr, sp, rc = [], [], []
        for i in range(0, len(X), k):
            sr.append(selected_runtime(pred[i:i+k], y[i:i+k]))
            sp.append(surpassed_plans(pred[i:i+k], y[i:i+k]))
            rc.append(rank_corr(pred[i:i+k], y[i:i+k]))
        print(f"SelectedRuntime(avg)={np.mean(sr):.3f}ms  SurpassedPlans(avg)={np.mean(sp):.2f}%  Spearman(avg)={np.mean(rc):.3f}")

if __name__ == '__main__':
    main()
