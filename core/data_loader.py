"""
Data Loader - 统一的数据加载接口，集成缓存功能
"""
import torch
import numpy as np
import copy
import dgl
import os
from pathlib import Path
from data.data_loader import data_loader
from core.data_cache import get_cache


def _dataset_path(name):
    default_root = Path(__file__).resolve().parents[1] / "data"
    data_root = Path(os.environ.get("CGL_DATA_ROOT", default_root))
    path = data_root / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. "
            "Set CGL_DATA_ROOT or pass --data-root."
        )
    return str(path)


def load_freebase(feat_type=1, use_cache=True):
    """加载Freebase数据集"""
    cache = get_cache()
    
    if use_cache and cache.has_full_dataset('Freebase', feat_type):
        return cache.load_full_dataset('Freebase', feat_type)
    
    dl = data_loader(_dataset_path('Freebase'))
    link_type_dic = {0: '00', 1: '01', 2: '03', 3: '05', 4: '06',
                     5: '11',
                     6: '20', 7: '21', 8: '22', 9: '23', 10: '25',
                     11: '31', 12: '33', 13: '35',
                     14: '40', 15: '41', 16: '42', 17: '43', 18: '44', 19: '45', 20: '46', 21: '47',
                     22: '51', 23: '55',
                     24: '61', 25: '62', 26: '63', 27: '65', 28: '66', 29: '67',
                     30: '70', 31: '71', 32: '72', 33: '73', 34: '75', 35: '77',
                     36: '-00', 37: '10', 38: '30', 39: '50', 40: '60',
                     41: '-11',
                     42: '02', 43: '12', 44: '-22', 45: '32', 46: '52',
                     47: '13', 48: '-33', 49: '53',
                     50: '04', 51: '14', 52: '24', 53: '34', 54: '-44', 55: '54', 56: '64', 57: '74',
                     58: '15', 59: '-55',
                     60: '16', 61: '26', 62: '36', 63: '56', 64: '-66', 65: '76',
                     66: '07', 67: '17', 68: '27', 69: '37', 70: '57', 71: '-77',
                     }
    book_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        # reverse
        if link_type_dic[link_type + 36][0] != '-':
            data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = dl.links['data'][link_type].T.nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        features = torch.FloatTensor(dl.nodes['attr'][0])
    else:
        indices = np.vstack((np.arange(book_num), np.arange(book_num)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(book_num))
        features = torch.sparse.FloatTensor(indices, values, torch.Size([book_num, book_num]))

    labels = dl.labels_test['data'][:book_num] + dl.labels_train['data'][:book_num]
    labels = [np.argmax(l) for l in labels]
    labels = torch.LongTensor(labels)

    num_classes = 7
    cls_node_ids = {}
    for i in range(num_classes):
        cls_node_ids[i] = torch.where(labels == i)[0]

    train_valid_mask = dl.labels_train['mask'][:book_num]
    test_mask = dl.labels_test['mask'][:book_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['00', '00'], ['01', '10'], ['05', '52', '20'], ['04', '40'], ['04', '43', '30'], ['06', '61', '10'],
                  ['07', '70'], ]
    
    result = (hg, features.to_dense(), labels, num_classes, train_indices, valid_indices, test_indices,
            torch.BoolTensor(train_mask), torch.BoolTensor(valid_mask), torch.BoolTensor(test_mask), cls_node_ids,
            meta_paths)
    
    if use_cache:
        cache.save_full_dataset('Freebase', feat_type, result)
    
    return result


def load_dblp(feat_type=0, use_cache=True):
    """加载DBLP数据集"""
    cache = get_cache()
    
    if use_cache and cache.has_full_dataset('DBLP', feat_type):
        return cache.load_full_dataset('DBLP', feat_type)
    
    prefix = _dataset_path('DBLP')
    dl = data_loader(prefix)
    link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        features = torch.FloatTensor(dl.nodes['attr'][0])
    else:
        features = torch.FloatTensor(np.eye(author_num))

    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    labels = [np.argmax(l) for l in labels]
    labels = torch.LongTensor(labels)

    num_classes = 4
    cls_node_ids = {}
    for i in range(num_classes):
        cls_node_ids[i] = torch.where(labels == i)[0]

    train_valid_mask = dl.labels_train['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    
    result = (hg, features, labels, num_classes, train_indices, valid_indices, test_indices,
           torch.BoolTensor(train_mask), torch.BoolTensor(valid_mask), torch.BoolTensor(test_mask), cls_node_ids,
            meta_paths)
    
    if use_cache:
        cache.save_full_dataset('DBLP', feat_type, result)
    
    return result


def load_imdb(feat_type=0, use_cache=True):
    """加载IMDB数据集"""
    cache = get_cache()
    
    if use_cache and cache.has_full_dataset('IMDB', feat_type):
        return cache.load_full_dataset('IMDB', feat_type)
    
    prefix = _dataset_path('IMDB')
    dl = data_loader(prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        features = torch.FloatTensor(dl.nodes['attr'][0])
    else:
        features = torch.FloatTensor(np.eye(movie_num))

    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = [np.argmax(l) for l in labels]
    labels = torch.LongTensor(labels)
    num_classes = 5

    cls_node_ids = {}
    for i in range(num_classes):
        cls_node_ids[i] = torch.where(labels == i)[0]

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    
    result = (hg, features, labels, num_classes, train_indices, valid_indices, test_indices,
            torch.BoolTensor(train_mask), torch.BoolTensor(valid_mask), torch.BoolTensor(test_mask), cls_node_ids,
            meta_paths)
    
    if use_cache:
        cache.save_full_dataset('IMDB', feat_type, result)
    
    return result


def load_data(dataset, feat_type=0, use_cache=True):
    """
    统一的数据加载接口
    
    Args:
        dataset: 数据集名称 ('DBLP', 'Freebase', 'IMDB')
        feat_type: 特征类型
        use_cache: 是否使用缓存
    
    Returns:
        数据集元组
    """
    load_fun = None
    if dataset == 'Freebase':
        feat_type = 1
        load_fun = load_freebase
    elif dataset == 'DBLP':
        load_fun = load_dblp
    elif dataset == 'IMDB':
        load_fun = load_imdb
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return load_fun(feat_type=feat_type, use_cache=use_cache)


def get_label_ids(dataset):
    """获取数据集的标签顺序"""
    if dataset == 'Freebase':
        return [1, 2, 3, 4, 5, 6, 0]
    elif dataset in ['DBLP', 'IMDB']:
        dataset_classes = {'DBLP': 4, 'IMDB': 5}
        num_cls = dataset_classes.get(dataset, 0)
        return list(np.arange(num_cls))
    else:
        raise ValueError("Name of Dataset is not correct!")
