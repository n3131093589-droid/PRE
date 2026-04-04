from datetime import datetime
import time
import argparse
import os

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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
    parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--data_size_ratio', type=float, default=1.0)
    parser.add_argument('--use_cuda', action='store_true', help='enable CUDA if available')
    parser.add_argument('--no_cuda', action='store_false', dest='use_cuda', help='disable CUDA')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--device', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--fold', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--ablation_mode', type=int, default=2, choices=[1, 2, 3, 4], help='1=orig_only, 2=v_residual, 3=v_no_gate, 4=v_only')
    parser.add_argument('--node_gate_mode', type=int, default=1, choices=[0, 1], help='0=off, 1=partner_substructure_gate')
    parser.add_argument('--pkl_name', type=str, default=f'./pkl/db-{time.strftime("%m%d_%H%M")}.pkl')
    return parser


def split_train_valid(data, seed, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_index, val_index = list(cv_split.split(X=data, y=data[:, 2]))[0]
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0], tup[1], int(tup[2])) for tup in train_tup]
    val_tup = [(tup[0], tup[1], int(tup[2])) for tup in val_tup]
    return train_tup, val_tup


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


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device, train_size, val_size, checkpoint_path, scheduler=None):
    best_mean_metrics, best_epoch = 0, 0
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        val_loss = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

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
            train_acc, train_auc_roc, train_f1, _, _, _, train_ap = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in with_progress(val_data_loader, f'Epoch {i}/{n_epochs} val'):
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, _, _ = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)

            val_loss /= val_size
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, _, _, _, val_ap = do_compute_metrics(val_probas_pred, val_ground_truth)
            mean_metrics = np.average([val_acc, val_auc_roc, val_f1])
            if mean_metrics > best_mean_metrics:
                best_mean_metrics, best_epoch = mean_metrics, i
                torch.save(model, checkpoint_path)

        if scheduler:
            scheduler.step()

        flag = '*' if best_epoch == i else ' '
        print(f'Epoch: {i}{flag} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        print(f'\t\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f},train_aupr: {train_ap:.4f},train_f1:{train_f1:.4f}')
        print(f'\t\tval_acc: {val_acc:.4f}, val_roc: {val_auc_roc:.4f}, val_aupr:{val_ap:.4f}, val_f1:{val_f1:.4f}')

        if i - best_epoch >= 40:
            print(f'Early Stopping at training epoch: {i}, best epoch: {best_epoch}')
            break


def test(test_data_loader, model, device):
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in with_progress(test_data_loader, 'Test'):
            model.eval()
            _, _, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)

    test_probas_pred = np.concatenate(test_probas_pred)
    test_ground_truth = np.concatenate(test_ground_truth)
    test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_roc: {test_auc_roc:.4f}, test_f1: {test_f1:.4f}, test_precision: {test_precision:.4f},test_recall: {test_recall:.4f},test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')


def main():
    args = build_parser().parse_args()
    n_atom_feats = args.n_atom_feats
    n_atom_hid = args.n_atom_hid
    rel_total = args.rel_total
    lr = args.lr
    n_epochs = args.n_epochs
    kge_dim = args.kge_dim
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    neg_samples = args.neg_samples
    data_size_ratio = args.data_size_ratio
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

    df_ddi_train = pd.read_csv(f'/cmy_project/code/hdn-main-sub_ex_change-drugbank/PRE-substructure-extraction-change/drugbank_test/DrugBank/warm_start/fold{args.fold}/train.csv')
    df_ddi_test = pd.read_csv(f'/cmy_project/code/hdn-main-sub_ex_change-drugbank/PRE-substructure-extraction-change/drugbank_test/DrugBank/warm_start/fold{args.fold}/test.csv')

    train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
    train_tup, val_tup = split_train_valid(train_tup, 2, val_ratio=0.2)
    test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

    train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
    val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
    test_data = DrugDataset(test_tup, disjoint_split=False)

    print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_data_loader = DrugDataLoader(val_data, batch_size=batch_size * 3, num_workers=2)
    test_data_loader = DrugDataLoader(test_data, batch_size=batch_size * 3, num_workers=2)

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

    train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, len(train_data), len(val_data), pkl_name, scheduler)
    test_model = load_checkpoint(pkl_name, device)
    test(test_data_loader, test_model, device)


if __name__ == '__main__':
    main()