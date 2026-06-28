"""
SEED数据集，读取并按1s时间窗口分割
"""

import numpy as np

from scipy.io import loadmat
from tqdm import tqdm
import os


def load_data(file, name, split, session_id):
    # trial*channel*sample
    eeg_data = loadmat(file)
    frequency = 200

    segments_list = []
    labels_list = []
    trials_id_list = []

    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    for trial in range(15):
        # 形状(62,47001) (channel, times)
        trial_signal = np.array(eeg_data[name + "_eeg" + str(trial + 1)])

        channels, times = trial_signal.shape

        # 计算每个时间窗口的样本数
        num_sample = times // split

        # 取前num_sample * split时间点
        trial_signal = trial_signal[:, : num_sample * split]
        # 按 时间窗口分割
        segmented_trial = trial_signal.reshape(channels, num_sample, split)
        segmented_trial = segmented_trial.transpose([1, 2, 0])

        # 得到(num_sample, split, channels)
        segments_list.append(segmented_trial)
        labels_list.extend([all_label[trial]] * num_sample)
        trial_id = trial + session_id * 15
        trials_id_list.extend([trial_id] * num_sample)

    # (samples, split, channels), (samples,)
    data = np.concatenate(segments_list, axis=0)
    label = np.array(labels_list)
    trials_id = np.array(trials_id_list)

    return data, label, trials_id


def main():

    file_path = "./SEED/Preprocessed_EEG/"

    people_name = [
        "1_20131027",
        "1_20131030",
        "1_20131107",
        "2_20140404",
        "2_20140413",
        "2_20140419",
        "3_20140603",
        "3_20140611",
        "3_20140629",
        "4_20140621",
        "4_20140702",
        "4_20140705",
        "5_20140411",
        "5_20140418",
        "5_20140506",
        "6_20130712",
        "6_20131016",
        "6_20131113",
        "7_20131027",
        "7_20131030",
        "7_20131106",
        "8_20140511",
        "8_20140514",
        "8_20140521",
        "9_20140620",
        "9_20140627",
        "9_20140704",
        "10_20131130",
        "10_20131204",
        "10_20131211",
        "11_20140618",
        "11_20140625",
        "11_20140630",
        "12_20131127",
        "12_20131201",
        "12_20131207",
        "13_20140527",
        "13_20140603",
        "13_20140610",
        "14_20140601",
        "14_20140615",
        "14_20140627",
        "15_20130709",
        "15_20131016",
        "15_20131105",
    ]

    short_name = [
        "djc",
        "djc",
        "djc",
        "jl",
        "jl",
        "jl",
        "jj",
        "jj",
        "jj",
        "lqj",
        "lqj",
        "lqj",
        "ly",
        "ly",
        "ly",
        "mhw",
        "mhw",
        "mhw",
        "phl",
        "phl",
        "phl",
        "sxy",
        "sxy",
        "sxy",
        "wk",
        "wk",
        "wk",
        "ww",
        "ww",
        "ww",
        "wsf",
        "wsf",
        "wsf",
        "wyw",
        "wyw",
        "wyw",
        "xyl",
        "xyl",
        "xyl",
        "ys",
        "ys",
        "ys",
        "zjy",
        "zjy",
        "zjy",
    ]

    split = 200
    output_dir = "./first_session_data_npy"
    num_session = 3
    os.makedirs(output_dir, exist_ok=True)
    for people_idx in tqdm(
        range(len(people_name) // num_session), desc="Processing", ncols=100
    ):
        sub_name = f"sub-{people_idx + 1:02d}"
        sub_dir = os.path.join(output_dir, sub_name)
        os.makedirs(sub_dir, exist_ok=True)

        data_list = []
        label_list = []
        trials_id_list = []
        print(f"读取 {sub_name} 的数据")
        session_id = 0

        idx = people_idx * num_session + session_id
        file_name = file_path + people_name[idx]
        print(f"读取 {file_name}")
        data, label, trials_id = load_data(
            file_name, short_name[idx], split=split, session_id=session_id
        )
        print(f"全局试次编号：{np.unique(trials_id)}")
        print(f"读取 {file_name} 完成")
        data_list.append(data)
        label_list.append(label)
        trials_id_list.append(trials_id)

        data = np.concatenate(data_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        trials_id = np.concatenate(trials_id_list, axis=0)

        print(f"{sub_name}: {data.shape}, {label.shape}, {trials_id.shape}")
        print(f"保存 {sub_name}的数据")
        np.save(os.path.join(sub_dir, f"{sub_name}_data.npy"), data)
        np.save(os.path.join(sub_dir, f"{sub_name}_label.npy"), label)
        np.save(os.path.join(sub_dir, f"{sub_name}_trials_id.npy"), trials_id)
        print(f"保存 {sub_name}的数据完成")


if __name__ == "__main__":
    main()
