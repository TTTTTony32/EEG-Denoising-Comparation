"""
Author: wolider wong
Date: 2024-1-13
Description: 训练, 测试过程
"""
import yaml
import importlib  # import model
import torch
import numpy as np
from datasets import Dataset
import os


def get_config(path: str):
    """
    加载yaml 配置文件
    Load the yaml configuration file
    :param path:
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def init_model(conf: dict) -> torch.nn.Module:
    m = _get_model(conf)
    w_path = conf["model"]["weight_path"]
    if w_path is not None and w_path != '':
        m.load_state_dict(torch.load(w_path))
    torch.cuda.empty_cache()
    return m


def _get_model(conf: dict) -> torch.nn.Module:
    """
    加载model
    Load model
    :return:
    """
    model_path = conf["model"]["path"]
    model_name = conf["model"]["class_name"]
    m = importlib.import_module(model_path)
    clz = getattr(m, model_name)
    return clz(**conf["model"]["config"])  # 实例化对象


def load_dataset(conf: dict, fmt: str = "torch", custom_data=None):
    """
    加载数据集，支持从磁盘加载或直接输入矩阵数据
    
    :param conf: 配置文件 config file
    :param fmt: 数据集中加载的数据的格式  Format of data loaded in the dataset
        在datasets 中支持 [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']
        Support in datasets [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax'].
        此处仅仅支持 ['numpy', 'torch', 'pandas']
        Only ['numpy', 'torch', 'pandas'] are supported here.
    :param custom_data: 自定义数据字典，格式为：
        {
            'train_x': numpy.ndarray,  # 训练集输入 (N_train, 512)
            'train_y': numpy.ndarray,  # 训练集输出 (N_train, 512) 
            'test_x': numpy.ndarray,   # 测试集输入 (N_test, 512)
            'test_y': numpy.ndarray,   # 测试集输出 (N_test, 512)
        }
    :return:
    """
    _dataset_fmt_check(fmt)
    
    if custom_data is not None:
        # 使用自定义矩阵数据
        print("使用自定义矩阵数据...")
        
        # 检查数据格式
        required_keys = ['train_x', 'train_y', 'test_x', 'test_y']
        for key in required_keys:
            if key not in custom_data:
                raise ValueError(f"自定义数据缺少必需的键: {key}")
        
        # 检查数据形状
        train_x, train_y = custom_data['train_x'], custom_data['train_y']
        test_x, test_y = custom_data['test_x'], custom_data['test_y']
        
        if train_x.shape[1] != 512 or train_y.shape[1] != 512:
            raise ValueError(f"训练数据形状错误，期望 (N, 512)，实际: train_x {train_x.shape}, train_y {train_y.shape}")
        
        if test_x.shape[1] != 512 or test_y.shape[1] != 512:
            raise ValueError(f"测试数据形状错误，期望 (N, 512)，实际: test_x {test_x.shape}, test_y {test_y.shape}")
        
        # 计算标准差（用于反归一化）
        train_std = np.ones((train_x.shape[0], 1))
        test_std = np.ones((test_x.shape[0], 1))
        
        # 创建数据集
        train_set = Dataset.from_dict({
            "x": train_x,  # 干净信号 (ground truth)
            "y": train_y,  # 含噪声信号 (输入)
            "std": train_std  # 标准差
        })
        
        test_set = Dataset.from_dict({
            "x": test_x,  # 干净信号 (ground truth)
            "y": test_y,  # 含噪声信号 (输入)
            "std": test_std  # 标准差
        })
        
        print(f"训练集大小: {train_x.shape}")
        print(f"测试集大小: {test_x.shape}")
        
    else:
        # 原有的从磁盘加载方式
        print("从磁盘加载数据集...")
        train_set = Dataset.load_from_disk(**conf["dataset"]["train"])
        test_set = Dataset.load_from_disk(**conf["dataset"]["test"])
    
    # 设置数据格式
    train_set.set_format(fmt)
    test_set.set_format(fmt)
    
    return train_set, test_set


def _dataset_fmt_check(fmt):
    """
    检查fmt的合法性
    Checking the legitimacy of fmt
    :param fmt:
    :return:
    """
    assert fmt in ['numpy', 'torch', 'pandas'], \
        f'''not support data format! need ['numpy', 'torch', 'pandas'], but have {fmt}'''


def load_dataset_with_path(path: str, fmt: str = "torch"):
    """
    从路径加载数据集
    Load dataset from path
    :param path: 数据集的路径  Path to the dataset
    :param fmt: 数据格式  data format
    :return:
    """
    _dataset_fmt_check(fmt)
    return Dataset.load_from_disk(path)


# def load_dataset(train_path: str, test_path: str, fmt: str = "torch"):
#     """
#
#     :param train_path: train data path
#     :param test_path: valid data path
#     :param fmt: 数据集中加载的数据的格式
#         在datasets 中支持 [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']
#         此处仅仅支持 ['numpy', 'torch', 'pandas']
#     :return:
#     """
#     assert fmt in ['numpy', 'torch', 'pandas'], \
#         f'''not support data format! need ['numpy', 'torch', 'pandas'], but have {fmt}'''
#     train_set = Dataset.load_from_disk(train_path)
#     train_set.set_format(fmt)  # set format to pt
#     test_set = Dataset.load_from_disk(test_path)
#     test_set.set_format(fmt)  # set format to pt
#     return train_set, test_set


def check_dir(base_path):
    """
    检查输出文件是否存在, 没有的话则创建
    Check if the output file exists, and create it if it doesn't.
    :return:
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        os.mkdir(os.path.join(base_path, 'img'))
        os.mkdir(os.path.join(base_path, 'logs'))
        os.mkdir(os.path.join(base_path, 'weight'))


def config_backpack(conf_path, save_dir):
    config_save_path = save_dir + "config.yml"
    os.system("cp {} {}".format(conf_path, config_save_path))


def init_optimizer(model, config: dict) -> torch.optim.Optimizer:
    """
    初始化优化器
    Initializing the Optimizer
    :param model:
    :param config:
    :return:
    """
    opti_dict = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SDG": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop
    }
    # 配置优化器
    optim_conf = config["train"].get("optimizer")
    if optim_conf is None:  # 如果说optimizer没有配置, 则加载默认的
        optim = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], betas=(0.5, 0.9), eps=1e-08)
        print(f'''using default optimizer AdamW(lr:{config["train"]["learning_rate"]},betas=(0.5, 0.9), eps=1e-08)''')
    else:
        keys = optim_conf.keys()
        assert len(keys) == 1, f"the optim_conf key must be have one, but found {[key for key in keys]}"
        key = list(keys)[0]
        assert key in opti_dict.keys(), f"un support optimizer! support {[item for item in opti_dict.keys()]}, but get{key}"
        # 如果没有lr参数, 则使用train.learning_rate, 或者是有这个key 但是没有值, 或者是值小于零
        if 'lr' not in optim_conf[key].keys() or optim_conf[key]['lr'] is None or optim_conf[key]['lr'] <= .0:
            optim_conf[key]['lr'] = config["train"]["learning_rate"]
        print(f'''using config optimizer {key}({optim_conf[key]})''')
        optim = opti_dict[key](model.parameters(), **optim_conf[key])
    return optim
