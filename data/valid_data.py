"""
看盾数据集形状是否正确
"""

import numpy as np


def main():
    data = np.load("./data_npy/sub-01/sub-01_data.npy")
    label = np.load("./data_npy/sub-01/sub-01_label.npy")
    trials_id = np.load("./data_npy/sub-01/sub-01_trials_id.npy")
    print(f"数据形状：{data.shape}")
    print(f"标签形状：{label.shape}")
    print(f"试次编号形状：{trials_id.shape}")

    global_trial = np.unique(trials_id)
    print(f"全局试次编号：{global_trial}, len(global_trial)：{len(global_trial)}")
    # 相同试次（trial_id相同）的样本要连续
    assert np.all(np.diff(trials_id) >= 0), "不同试次的样本要连续"
    print(label[:900])


if __name__ == "__main__":
    main()
