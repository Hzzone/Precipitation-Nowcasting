import json
import os
import numpy as np
from nowcasting.config import cfg
from nowcasting.helpers.visualization import save_hko_movie
from nowcasting.hko.dataloader import HKOIterator
from nowcasting.hko.evaluation import HKOEvaluation

class HKOBenchmarkEnv(object):
    """The Benchmark environment for the HKO7 Dataset

    There are two settings for the Benchmark, the "fixed" setting and the "online" setting.
    In the "fixed" setting, pre-defined input sequences that have the same length will be
     fed into the model for prediction.
        This setting tests the model's ability to use the instant past to predict the future.
    In the "online" setting, M frames will be given each time and the forecasting model
     is required to predict the next K frames every stride steps.
        If the begin_new_episode flag is turned on, a new episode has begun, which means that the current received images have no relationship with the previous images.
        If the need_upload_prediction flag is turned on, the model is required to predict the
        This setting tests both the model's ability to adapt in an online fashion and
         the ability to capture the long-term dependency.
    The input frame will be missing in some timestamps.

    To run the benchmark in the fixed setting:

    env = HKOBenchmarkEnv(...)
    while not env.done:
        # Get the observation
        in_frame_dat, in_mask_dat, in_datetime_clips, out_datetime_clips, begin_new_episode =
         env.get_observation(batch_size=batch_size)
        # Running your algorithm to get the prediction
        prediction = ...
        # Upload prediction to the environment
        env.upload_prediction(prediction)

    """
    def __init__(self,
                 pd_path,
                 save_dir="hko7_benchmark",
                 mode="fixed"):
        assert mode == "fixed" or mode == "online"
        self._pd_path = pd_path
        self._save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._mode = mode
        self._out_seq_len = cfg.HKO.BENCHMARK.OUT_LEN
        self._stride = cfg.HKO.BENCHMARK.STRIDE
        if mode == "fixed":
            self._in_seq_len = cfg.HKO.BENCHMARK.IN_LEN
        else:
            self._in_seq_len = cfg.HKO.BENCHMARK.STRIDE
        self._hko_iter = HKOIterator(pd_path=pd_path,
                                     sample_mode="sequent",
                                     seq_len=self._in_seq_len + self._out_seq_len,
                                     stride=self._stride)
        self._stat_dict = self._get_benchmark_stat()
        self._begin_new_episode = True
        self._received_pred_seq_num = 0
        self._need_upload_prediction = False
        #TODO Save some predictions

        self._save_seq_inds = set(np.arange(1, cfg.HKO.BENCHMARK.VISUALIZE_SEQ_NUM + 1) * \
                              (self._stat_dict['pred_seq_num'] //
                               cfg.HKO.BENCHMARK.VISUALIZE_SEQ_NUM))
        self._all_eval = HKOEvaluation(seq_len=self._out_seq_len, use_central=False)

        # The seq_ids that will be saved as a gif


        # Holder of the inner data
        self._in_frame_dat = None
        self._in_mask_dat = None
        self._in_datetime_clips = None
        self._out_frame_dat = None
        self._out_mask_dat = None
        self._out_datetime_clips = None

    def reset(self):
        self._hko_iter.reset()
        self._all_eval.clear_all()
        self._begin_new_episode = True
        self._received_pred_seq_num = 0
        self._need_upload_prediction = False

    @property
    def _fingerprint(self):
        pd_file_name = os.path.splitext(os.path.basename(self._pd_path))[0]
        if self._mode == "fixed":
            fingerprint = pd_file_name + "_in" + str(self._in_seq_len)\
                          + "_out" + str(self._out_seq_len) + "_stride" + str(self._stride)\
                          + "_" + self._mode
        else:
            fingerprint = pd_file_name + "_out" + str(self._out_seq_len)\
                          + "_stride" + str(self._stride)\
                          + "_" + self._mode
        return fingerprint

    @property
    def _stat_filepath(self):
        filename = self._fingerprint + ".json"
        return os.path.join(cfg.HKO.BENCHMARK.STAT_PATH, filename)

    def _get_benchmark_stat(self):
        """Get the general statistics of the benchmark

        Returns
        -------
        stat_dict : dict
            'pred_seq_num' --> Total number of predictions the model needs to make
        """
        if os.path.exists(self._stat_filepath):
            stat_dict = json.load(open(self._stat_filepath))
        else:
            seq_num = 0
            episode_num = 0
            episode_start_datetime = []
            while not self._hko_iter.use_up:
                if self._mode == "fixed":
                    datetime_clips, new_start =\
                        self._hko_iter.sample(batch_size=1024, only_return_datetime=True)
                    if len(datetime_clips) == 0:
                        continue
                    seq_num += len(datetime_clips)
                    episode_num += len(datetime_clips)
                elif self._mode == "online":
                    datetime_clips, new_start = \
                        self._hko_iter.sample(batch_size=1, only_return_datetime=True)
                    if len(datetime_clips) == 0:
                        continue
                    episode_num += new_start
                    if new_start:
                        episode_start_datetime.append(datetime_clips[0][0].strftime('%Y%m%d%H%M'))
                        if self._stride != 1:
                            seq_num += 1
                    else:
                        seq_num += 1
                print(self._fingerprint, seq_num, episode_num)
            self._hko_iter.reset()
            stat_dict = {'pred_seq_num': seq_num,
                         'episode_num': episode_num,
                         'episode_start_datetime': episode_start_datetime}
            json.dump(stat_dict, open(self._stat_filepath, 'w'), indent=3)
        return stat_dict

    @property
    def done(self):
        return self._received_pred_seq_num >= self._stat_dict["pred_seq_num"]

    def get_observation(self, batch_size=1):
        """

        Parameters
        ----------
        batch_size : int


        Returns
        -------
        in_frame_dat : np.ndarray
            Will be between 0 and 1
        in_datetime_clips : list
        out_datetime_clips : list
        begin_new_episode : bool
        need_upload_prediction : bool
        """
        if self._mode == "online":
            assert batch_size == 1
        assert not self._need_upload_prediction
        assert not self.done
        assert not self._hko_iter.use_up
        while True:
            frame_dat, mask_dat, datetime_clips, new_start =\
                self._hko_iter.sample(batch_size=batch_size, only_return_datetime=False)
            if len(datetime_clips) == 0:
                continue
            else:
                break
        frame_dat = frame_dat.astype(np.float32) / 255.0
        self._need_upload_prediction = True
        if self._mode == "online":
            self._begin_new_episode = new_start
            if new_start and self._stride == 1:
                self._need_upload_prediction = False
        else:
            self._begin_new_episode = True
        self._in_datetime_clips = [ele[:self._in_seq_len] for ele in datetime_clips]
        self._out_datetime_clips = [ele[self._in_seq_len:(self._in_seq_len +
                                                          self._out_seq_len)]
                                    for ele in datetime_clips]
        self._in_frame_dat = frame_dat[:self._in_seq_len, ...]
        self._out_frame_dat = frame_dat[self._in_seq_len:(self._in_seq_len + self._out_seq_len),
                              ...]
        self._in_mask_dat = mask_dat[:self._in_seq_len, ...]
        self._out_mask_dat = mask_dat[self._in_seq_len:(self._in_seq_len + self._out_seq_len), ...]
        return self._in_frame_dat,\
               self._in_datetime_clips,\
               self._out_datetime_clips,\
               self._begin_new_episode, \
               self._need_upload_prediction

    def upload_prediction(self, prediction, save_result=False):
        """

        Parameters
        ----------
        prediction : np.ndarray

        """
        assert self._need_upload_prediction, "Must call get_observation first!" \
                                             " Also, check the value of need_upload_predction" \
                                             " after calling"
        self._need_upload_prediction = False
        received_seq_inds = range(self._received_pred_seq_num,
                                  self._received_pred_seq_num + prediction.shape[1])
        save_ind_set = set(received_seq_inds).intersection(self._save_seq_inds)
        if len(save_ind_set) > 0 and save_result:
            assert len(save_ind_set) == 1
            ind = save_ind_set.pop()
            ind -= self._received_pred_seq_num
            if not os.path.exists(os.path.join(self._save_dir, self._fingerprint)):
                os.makedirs(os.path.join(self._save_dir, self._fingerprint))
            print("Saving prediction videos to %s" % os.path.join(self._save_dir,
                                                                  self._fingerprint))
            save_hko_movie(im_dat=self._in_frame_dat[:, ind, 0, ...],
                           mask_dat=self._in_mask_dat[:, ind, 0, :, :],
                           datetime_list=self._in_datetime_clips[ind],
                           save_path=os.path.join(self._save_dir, self._fingerprint,
                                                  "%s_in.mp4" %
                                                  self._in_datetime_clips[ind][0]
                                                  .strftime('%Y%m%d%H%M')))
            save_hko_movie(im_dat=self._out_frame_dat[:, ind, 0, ...],
                           mask_dat=self._out_mask_dat[:, ind, 0, :, :],
                           masked=False,
                           datetime_list=self._out_datetime_clips[ind],
                           save_path=os.path.join(self._save_dir, self._fingerprint,
                                                  "%s_out.mp4" %
                                                  self._in_datetime_clips[ind][0]
                                                  .strftime('%Y%m%d%H%M')))
            save_hko_movie(im_dat=prediction[:, ind, 0, ...],
                           mask_dat=self._out_mask_dat[:, ind, 0, :, :],
                           masked=False,
                           datetime_list=self._out_datetime_clips[ind],
                           save_path=os.path.join(self._save_dir, self._fingerprint,
                                                  "%s_pred.mp4" %
                                                  self._in_datetime_clips[ind][0]
                                                  .strftime('%Y%m%d%H%M')))
        self._received_pred_seq_num += prediction.shape[1]
        if self._mode == "online":
            if self._stride == 1:
                assert not self._begin_new_episode
        self._all_eval.update(gt=self._out_frame_dat,
                              pred=prediction,
                              mask=self._out_mask_dat,
                              start_datetimes=[ele[0] for ele in self._out_datetime_clips])

    def print_stat_readable(self):
        self._all_eval.print_stat_readable(prefix="Received:%d " %self._received_pred_seq_num)

    def save_eval(self):
        assert self._received_pred_seq_num == self._stat_dict['pred_seq_num'],\
            "Must upload all the predictions to the testbed!"
        print("Saving evaluation result to %s" %os.path.join(self._save_dir, self._fingerprint))
        self._all_eval.save(prefix=os.path.join(self._save_dir, self._fingerprint, "eval_all"))


if __name__ == '__main__':
    env = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.RAINY_TEST, mode="fixed")
    env._stat_dict['pred_seq_num'] = 10
    # print(env._stat_dict['pred_seq_num'])
    while not env.done:
        # Get the observation
        in_frame_dat, in_mask_dat, in_datetime_clips, out_datetime_clips, begin_new_episode = \
        env.get_observation(batch_size=2)
        # Running your algorithm to get the prediction
        prediction = np.zeros_like(env._out_frame_dat)
        # Upload prediction to the environment
        env.upload_prediction(prediction)
        # print(env._received_pred_seq_num)
    pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl = env._all_eval.calculate_stat()
    # print(env._all_eval.calculate_stat())

    # env_online = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.RAINY_TEST, mode="online")
    # print("Fixed Rainy Test SeqNum:", env_fixed._stat_dict['pred_seq_num'])
    # print("Online Rainy Test SeqNum:", env_online._stat_dict['pred_seq_num'])
    # env_fixed = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.RAINY_VALID, mode="fixed")
    # env_online = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.RAINY_VALID, mode="online")
    # print("Fixed Rainy Valid SeqNum:", env_fixed._stat_dict['pred_seq_num'])
    # print("Online Rainy Valid SeqNum:", env_online._stat_dict['pred_seq_num'])
    # env_fixed = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.ALL_15, mode="fixed")
    # env_online = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.ALL_15, mode="online")
    # print("Fixed Rainy2015 ALL SeqNum:", env_fixed._stat_dict['pred_seq_num'])
    # print("Online Rainy2015 ALL SeqNum:", env_online._stat_dict['pred_seq_num'])
