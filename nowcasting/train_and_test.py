import sys
sys.path.insert(0, '../')
import torch
from nowcasting.hko.dataloader import HKOIterator
from nowcasting.config import cfg
import numpy as np
from nowcasting.hko.evaluation import HKOEvaluation
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os.path as osp
import os
import shutil
import copy



def train_and_test(encoder_forecaster, optimizer, criterion, lr_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name, probToPixel=None):
    # HKO-7 evaluater and dataloader
    IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
    OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
    evaluater = HKOEvaluation(seq_len=OUT_LEN, use_central=False)
    train_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TRAIN,
                                     sample_mode="random",
                                     seq_len=IN_LEN+OUT_LEN)

    valid_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_VALID,
                                     sample_mode="sequent",
                                     seq_len=IN_LEN+OUT_LEN,
                                     stride=cfg.HKO.BENCHMARK.STRIDE)

    train_loss = 0.0
    save_dir = osp.join(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    model_save_dir = osp.join(save_dir, 'models')
    log_dir = osp.join(save_dir, 'logs')
    all_scalars_file_name = osp.join(save_dir, "all_scalars.json")
    pkl_save_dir = osp.join(save_dir, 'pkl')
    if osp.exists(all_scalars_file_name):
        os.remove(all_scalars_file_name)
    if osp.exists(log_dir):
        shutil.rmtree(log_dir)
    if osp.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.mkdir(model_save_dir)

    writer = SummaryWriter(log_dir)

    for itera in tqdm(range(1, max_iterations+1)):
        lr_scheduler.step()
        train_batch, train_mask, sample_datetimes, _ = \
            train_hko_iter.sample(batch_size=batch_size)
        train_batch = torch.from_numpy(train_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_data = train_batch[:IN_LEN, ...]
        train_label = train_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
        mask = torch.from_numpy(train_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)

        encoder_forecaster.train()
        optimizer.zero_grad()
        output = encoder_forecaster(train_data)
        loss = criterion(output, train_label, mask)
        loss.backward()
        torch.nn.utils.clip_grad_value_(encoder_forecaster.parameters(), clip_value=50.0)
        optimizer.step()
        train_loss += loss.item()


        train_label_numpy = train_label.cpu().numpy()
        if probToPixel is None:
            # 未使用分类问题
            output_numpy = np.clip(output.detach().cpu().numpy(), 0.0, 1.0)
        else:
            # if classification, output: S*B*C*H*W
            # 使用分类问题，需要转化为像素值
            # 使用分类 Loss 的阈值
            output_numpy = probToPixel(output.detach().cpu().numpy(), train_label, mask,
                                                            lr_scheduler.get_lr()[0])

        evaluater.update(train_label_numpy, output_numpy, mask.cpu().numpy())

        if itera % test_iteration_interval == 0:
            _, _, train_csi, train_hss, _, train_mse, train_mae, train_balanced_mse, train_balanced_mae, _ = evaluater.calculate_stat()

            train_loss = train_loss/test_iteration_interval

            evaluater.clear_all()

            with torch.no_grad():
                encoder_forecaster.eval()
                valid_hko_iter.reset()
                valid_loss = 0.0
                valid_time = 0
                while not valid_hko_iter.use_up:
                    valid_batch, valid_mask, sample_datetimes, _ = \
                        valid_hko_iter.sample(batch_size=batch_size)
                    if valid_batch.shape[1] == 0:
                        break
                    if not cfg.HKO.EVALUATION.VALID_DATA_USE_UP and valid_time > cfg.HKO.EVALUATION.VALID_TIME:
                        break
                    valid_time += 1
                    valid_batch = torch.from_numpy(valid_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                    valid_data = valid_batch[:IN_LEN, ...]
                    valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
                    mask = torch.from_numpy(valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)
                    output = encoder_forecaster(valid_data)

                    loss = criterion(output, valid_label, mask)
                    valid_loss += loss.item()

                    valid_label_numpy = valid_label.cpu().numpy()
                    if probToPixel is None:
                        output_numpy = np.clip(output.detach().cpu().numpy(), 0.0, 1.0)
                    else:
                        output_numpy = probToPixel(output.detach().cpu().numpy(), valid_label, mask, lr_scheduler.get_lr()[0])

                    evaluater.update(valid_label_numpy, output_numpy, mask.cpu().numpy())
                _, _, valid_csi, valid_hss, _, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae, _ = evaluater.calculate_stat()

                evaluater.clear_all()
                valid_loss = valid_loss/valid_time

            writer.add_scalars("loss", {
                "train": train_loss,
                "valid": valid_loss
            }, itera)

            plot_result(writer, itera, (train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae),
                        (valid_csi, valid_hss, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae))

            writer.export_scalars_to_json(all_scalars_file_name)

            train_loss = 0.0

        if itera % test_and_save_checkpoint_iterations == 0:
            torch.save(encoder_forecaster.state_dict(), osp.join(model_save_dir, 'encoder_forecaster_{}.pth'.format(itera)))

    writer.close()

def plot_result(writer, itera, train_result, valid_result):
    train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae = \
        train_result
    train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae = \
        np.nan_to_num(train_csi), \
        np.nan_to_num(train_hss), \
        np.nan_to_num(train_mse), \
        np.nan_to_num(train_mae), \
        np.nan_to_num(train_balanced_mse), \
        np.nan_to_num(train_balanced_mae)

    valid_csi, valid_hss, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae = \
        valid_result
    valid_csi, valid_hss, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae = \
        np.nan_to_num(valid_csi), \
        np.nan_to_num(valid_hss), \
        np.nan_to_num(valid_mse), \
        np.nan_to_num(valid_mae), \
        np.nan_to_num(valid_balanced_mse), \
        np.nan_to_num(valid_balanced_mae)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):

        writer.add_scalars("csi/{}".format(thresh), {
            "train": train_csi[:, i].mean(),
            "valid": valid_csi[:, i].mean(),
            "train_last_frame": train_csi[-1, i],
            "valid_last_frame": valid_csi[-1, i]
        }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):

        writer.add_scalars("hss/{}".format(thresh), {
            "train": train_hss[:, i].mean(),
            "valid": valid_hss[:, i].mean(),
            "train_last_frame": train_hss[-1, i],
            "valid_last_frame": valid_hss[-1, i]
        }, itera)

    writer.add_scalars("mse", {
        "train": train_mse.mean(),
        "valid": valid_mse.mean(),
        "train_last_frame": train_mse[-1],
        "valid_last_frame": valid_mse[-1],
    }, itera)

    writer.add_scalars("mae", {
        "train": train_mae.mean(),
        "valid": valid_mae.mean(),
        "train_last_frame": train_mae[-1],
        "valid_last_frame": valid_mae[-1],
    }, itera)

    writer.add_scalars("balanced_mse", {
        "train": train_balanced_mse.mean(),
        "valid": valid_balanced_mse.mean(),
        "train_last_frame": train_balanced_mse[-1],
        "valid_last_frame": valid_balanced_mse[-1],
    }, itera)

    writer.add_scalars("balanced_mae", {
        "train": train_balanced_mae.mean(),
        "valid": valid_balanced_mae.mean(),
        "train_last_frame": train_balanced_mae[-1],
        "valid_last_frame": valid_balanced_mae[-1],
    }, itera)


