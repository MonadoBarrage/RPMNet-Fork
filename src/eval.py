from collections import defaultdict
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import open3d
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch
import argparse

from arguments import rpmnet_eval_arguments
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, to_numpy
from common.math import se3
from common.math_torch import se3
from common.math.so3 import dcm2euler
from data_loader.datasets_ply import get_test_dataset_from_ply
import models.rpmnet


def compute_metrics(data: Dict, pred_transforms) -> Dict:
    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].cpu().numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].cpu().numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        src_transformed = se3.transform(pred_transforms, points_src)
        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_numpy(t_mse),
            't_mae': to_numpy(t_mae),
            'err_r_deg': to_numpy(residual_rotdeg),
            'err_t': to_numpy(residual_transmag),
            'chamfer_dist': to_numpy(chamfer_dist)
        }

    return metrics


def summarize_metrics(metrics):
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k] ** 2))
        else:
            summarized[k] = np.mean(metrics[k])
    return summarized


def print_metrics(logger, summary_metrics: Dict, losses_by_iteration: List = None, title: str = 'Metrics'):
    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.7f}(mean-sq)'.format(summary_metrics['chamfer_dist']))


def inference(data_loader, model: torch.nn.Module):
    _logger.info('Starting inference...')
    model.eval()
    pred_transforms_all = []
    all_betas, all_alphas = [], []
    total_time = 0.0
    endpoints_out = defaultdict(list)
    total_rotation = []

    with torch.no_grad():
        for val_data in tqdm(data_loader):
            dict_all_to_device(val_data, _device)
            time_before = time.time()
            pred_transforms, endpoints = model(val_data, _args.num_reg_iter)
            total_time += time.time() - time_before

            if isinstance(pred_transforms[-1], torch.Tensor):
                pred_transforms_all.append(to_numpy(torch.stack(pred_transforms, dim=1)))
            else:
                pred_transforms_all.append(np.stack(pred_transforms, axis=1))

    _logger.info('Total inference time: {}s'.format(total_time))
    pred_transforms_all = np.concatenate(pred_transforms_all, axis=0)
    return pred_transforms_all, endpoints_out


def evaluate(pred_transforms, data_loader):
    _logger.info('Evaluating transforms...')
    num_processed = 0
    pred_transforms = torch.from_numpy(pred_transforms).to(_device) \
        if pred_transforms.ndim == 4 else torch.from_numpy(pred_transforms[:, None, :, :]).to(_device)

    metrics_for_iter = [defaultdict(list) for _ in range(pred_transforms.shape[1])]
    filenames = []

    for data in tqdm(data_loader, leave=False):
        dict_all_to_device(data, _device)
        batch_size = data['points_src'].shape[0]
        filenames.extend(data['filename'])

        for i_iter in range(pred_transforms.shape[1]):
            cur_pred_transforms = pred_transforms[num_processed:num_processed + batch_size, i_iter, :, :]
            metrics = compute_metrics(data, cur_pred_transforms)
            for k in metrics:
                metrics_for_iter[i_iter][k].append(metrics[k])
        num_processed += batch_size

    for i_iter in range(len(metrics_for_iter)):
        metrics_for_iter[i_iter] = {k: np.concatenate(metrics_for_iter[i_iter][k], axis=0)
                                    for k in metrics_for_iter[i_iter]}
        metrics_for_iter[i_iter]['filename'] = np.array(filenames)
        summary_metrics = summarize_metrics(metrics_for_iter[i_iter])
        print_metrics(_logger, summary_metrics, title=f'Evaluation result (iter {i_iter})')

    return metrics_for_iter, summary_metrics


def save_eval_data(pred_transforms, endpoints, metrics, summary_metrics, save_path):
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'pred_transforms.npy'), pred_transforms)

    for k in endpoints:
        if isinstance(endpoints[k], np.ndarray):
            np.save(os.path.join(save_path, f'{k}.npy'), endpoints[k])
        else:
            with open(os.path.join(save_path, f'{k}.pickle'), 'wb') as fid:
                pickle.dump(endpoints[k], fid)

    writer = pd.ExcelWriter(os.path.join(save_path, 'metrics.xlsx'))
    for i_iter, iter_metrics in enumerate(metrics):
        iter_metrics['r_rmse'] = np.sqrt(iter_metrics['r_mse'])
        iter_metrics['t_rmse'] = np.sqrt(iter_metrics['t_mse'])
        iter_metrics.pop('r_mse')
        iter_metrics.pop('t_mse')
        df = pd.DataFrame(iter_metrics)
        df.to_excel(writer, sheet_name=f'Iter_{i_iter+1}', index=False)
    writer.close()

    summary_metrics_float = {k: float(summary_metrics[k]) for k in summary_metrics}
    with open(os.path.join(save_path, 'summary_metrics.json'), 'w') as f:
        json.dump(summary_metrics_float, f)

    # Save per-file CSV (from last iteration)
    df_csv = pd.DataFrame(metrics[-1])
    df_csv.to_csv
