import argparse
import os

import torch
from sklearn import metrics
import pandas as pd
import numpy as np

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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--use_cuda', action='store_true', help='enable CUDA if available')
    parser.add_argument('--no_cuda', action='store_false', dest='use_cuda', help='disable CUDA')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--device', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--ablation_mode', type=int, default=2, choices=[1, 2, 3, 4], help='1=orig_only, 2=v_residual, 3=v_no_gate, 4=v_only')
    parser.add_argument('--node_gate_mode', type=int, default=0, choices=[0, 1], help='0=off, 1=partner_substructure_gate')
    parser.add_argument('--pkl_name', type=str, default='drugbank_test/transductive_drugbank.pkl')
    return parser


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
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')


def main():
    args = build_parser().parse_args()
    pkl_name = append_experiment_suffix(args.pkl_name, args.ablation_mode, args.node_gate_mode)
    device = f'cuda:{args.device}' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(args)
    print(f"Ablation mode: {ABLATION_MODE_NAMES[args.ablation_mode]}")
    print(f"Node gate mode: {NODE_GATE_MODE_NAMES[args.node_gate_mode]}")
    print(f"Checkpoint path: {pkl_name}")

    df_ddi_test = pd.read_csv(f'drugbank_test/DrugBank/warm_start/fold{args.fold}/test.csv')
    test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]
    test_data = DrugDataset(test_tup, disjoint_split=False)

    print(f"Testing with {len(test_data)} samples")
    test_data_loader = DrugDataLoader(test_data, batch_size=args.batch_size * 3, num_workers=2)

    test_model = torch.load(pkl_name, map_location=device)
    test_model.to(device=device)
    test(test_data_loader, test_model, device)


if __name__ == '__main__':
    main()
