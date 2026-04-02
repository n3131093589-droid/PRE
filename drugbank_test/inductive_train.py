from datetime import datetime
import time
import argparse
import os
import random

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np

import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ABLATION_MODE_NAMES = {
    1: 'orig_only',
    2: 'v_residual',
    3: 'v_no_gate',
    4: 'v_only',
}

NODE_GATE_MODE_NAMES = {
    0: 'off',
    1: 'partner_substructure_gate',
}


def with_progress(iterable, desc):
    if tqdm is None:
        return iterable
    total = len(iterable) if hasattr(iterable, '__len__') else None
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


def append_experiment_suffix(path, ablation_mode, node_gate_mode):
    root, ext = os.path.splitext(path)
    suffix = f'-ab{ablation_mode}-ng{node_gate_mode}'
    if root.endswith(suffix):
        return path
    return f'{root}{suffix}{ext}'


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_atom_feats', type=int, default=66, help='num of input features')
    parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
    parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_cuda', action='store_true', help='enable CUDA if available')
    parser.add_argument('--no_cuda', action='store_false', dest='use_cuda', help='disable CUDA')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--device', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 2015])
    parser.add_argument('--ablation_mode', type=int, default=2, choices=[1, 2, 3, 4], help='1=orig_only, 2=v_residual, 3=v_no_gate, 4=v_only')
    parser.add_argument('--node_gate_mode', type=int, default=0, choices=[0, 1], help='0=off, 1=partner_substructure_gate')
    parser.add_argument('--pkl_name', type=str, default=f'./pkl/db-{time.strftime("%m%d_%H%M")}.pkl')
    return parser


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def do_compute(batch, device, model):
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, _ = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)
    return acc, auroc, f1_score, precision, recall, int_ap, ap


def train(model, train_data_loader, s1_data_loader, s2_data_loader, loss_fn, optimizer, n_epochs, device, train_size, s1_size, s2_size, checkpoint_path, scheduler=None):
    print('Starting training at', datetime.today())
    best_mean_metrics, best_epoch = 0, 0
    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        s1_loss = 0
        s2_loss = 0
        train_probas_pred = []
        train_ground_truth = []
        s1_probas_pred = []
        s1_ground_truth = []
        s2_probas_pred = []
        s2_ground_truth = []

        for batch in with_progress(train_data_loader, f'Epoch {i}/{n_epochs} train'):
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, _, _ = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= train_size

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)
            train_acc, train_auc_roc, _, train_precision, train_recall, _, _ = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in with_progress(s1_data_loader, f'Epoch {i}/{n_epochs} s1'):
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s1_probas_pred.append(probas_pred)
                s1_ground_truth.append(ground_truth)
                loss, _, _ = loss_fn(p_score, n_score)
                s1_loss += loss.item() * len(p_score)

            s1_loss /= s1_size
            s1_probas_pred = np.concatenate(s1_probas_pred)
            s1_ground_truth = np.concatenate(s1_ground_truth)
            s1_acc, s1_auc_roc, s1_f1, _, _, _, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)

            for batch in with_progress(s2_data_loader, f'Epoch {i}/{n_epochs} s2'):
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s2_probas_pred.append(probas_pred)
                s2_ground_truth.append(ground_truth)
                loss, _, _ = loss_fn(p_score, n_score)
                s2_loss += loss.item() * len(p_score)

            s2_loss /= s2_size
            s2_probas_pred = np.concatenate(s2_probas_pred)
            s2_ground_truth = np.concatenate(s2_ground_truth)
            s2_acc, s2_auc_roc, s2_f1, _, _, _, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

            s1_metrics = np.average([s1_acc, s1_auc_roc, s1_f1])
            s2_metrics = np.average([s2_acc, s2_auc_roc, s2_f1])
            mean_metrics = np.average([s1_metrics, s2_metrics])
            if mean_metrics > best_mean_metrics:
                best_mean_metrics, best_epoch = mean_metrics, i
                torch.save(model, checkpoint_path)

        if scheduler:
            scheduler.step()

        flag = '*' if best_epoch == i else ' '
        print(f'Epoch: {i}{flag} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f},s2_loss: {s2_loss:.4f}')
        print(f'\t\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f},train_precision: {train_precision:.4f},train_recall:{train_recall:.4f}')
        print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_aupr:{s1_ap:.4f}, s1_f1:{s1_f1:.4f}')
        print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_aupr:{s2_ap:.4f}, s2_f1:{s2_f1:.4f}')

        if i - best_epoch >= 30:
            print(f'Early Stopping at training epoch: {i}, best epoch: {best_epoch}')
            break


def test(s1_data_loader, s2_data_loader, model, device):
    s1_probas_pred = []
    s1_ground_truth = []
    s2_probas_pred = []
    s2_ground_truth = []
    with torch.no_grad():
        for batch in with_progress(s1_data_loader, 'Test s1'):
            model.eval()
            _, _, probas_pred, ground_truth = do_compute(batch, device, model=model)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)

        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)

        for batch in with_progress(s2_data_loader, 'Test s2'):
            model.eval()
            _, _, probas_pred, ground_truth = do_compute(batch, device, model=model)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)

        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

    print('\n')
    print('============================== Best Result ==============================')
    print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f},s1_recall: {s1_recall:.4f},s1_int_ap: {s1_int_ap:.4f},s1_ap: {s1_ap:.4f}')
    print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f},s2_recall: {s2_recall:.4f},s2_int_ap: {s2_int_ap:.4f},s2_ap: {s2_ap:.4f}')


def main():
    seed_everything(42)
    args = build_parser().parse_args()
    n_atom_feats = args.n_atom_feats
    n_atom_hid = args.n_atom_hid
    rel_total = args.rel_total
    lr = args.lr
    n_epochs = args.n_epochs
    kge_dim = args.kge_dim
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    pkl_name = args.pkl_name.replace('.pkl', f'-fold{args.fold}.pkl')
    pkl_name = append_experiment_suffix(pkl_name, args.ablation_mode, args.node_gate_mode)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    if use_cuda:
        torch.cuda.set_device(args.device)
    device = f'cuda:{args.device}' if use_cuda else 'cpu'
    print(args)
    print(f"Ablation mode: {ABLATION_MODE_NAMES[args.ablation_mode]}")
    print(f"Node gate mode: {NODE_GATE_MODE_NAMES[args.node_gate_mode]}")
    print(f"Checkpoint path: {pkl_name}")

    checkpoint_dir = os.path.dirname(pkl_name)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    df_ddi_train = pd.concat([
        pd.read_csv(f'drugbank_test/DrugBank/cold_start/fold{args.fold}/train.csv'),
        pd.read_csv(f'drugbank_test/DrugBank/cold_start/fold{args.fold}/val.csv')
    ], axis=0)
    df_ddi_s1 = pd.read_csv(f'drugbank_test/DrugBank/cold_start/fold{args.fold}/s1.csv')
    df_ddi_s2 = pd.read_csv(f'drugbank_test/DrugBank/cold_start/fold{args.fold}/s2.csv')

    train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
    s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
    s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

    train_data = DrugDataset(train_tup)
    s1_data = DrugDataset(s1_tup, disjoint_split=True)
    s2_data = DrugDataset(s2_tup, disjoint_split=True)

    print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    s1_data_loader = DrugDataLoader(s1_data, batch_size=batch_size * 3, num_workers=2)
    s2_data_loader = DrugDataLoader(s2_data, batch_size=batch_size * 3, num_workers=2)

    model = models.HDN_DDI(
        in_features=n_atom_feats,
        hidd_dim=n_atom_hid,
        kge_dim=kge_dim,
        rel_total=rel_total,
        heads_out_feat_params=[64, 64, 64, 64],
        blocks_params=[2, 2, 2, 2],
        ablation_mode=args.ablation_mode,
        node_gate_mode=args.node_gate_mode,
    )
    loss = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)
    model.to(device=device)

    train(model, train_data_loader, s1_data_loader, s2_data_loader, loss, optimizer, n_epochs, device, len(train_data), len(s1_data), len(s2_data), pkl_name, scheduler)
    test_model = load_checkpoint(pkl_name, device)
    test(s1_data_loader, s2_data_loader, test_model, device)


if __name__ == '__main__':
    main()
