"""
Task Data Manager - 处理任务级别的数据分割和子图构建
"""
import torch
import numpy as np
import copy
import dgl
from core.data_cache import get_cache
from core.data_loader import get_label_ids


def get_subgraph_dblp(hg, target_nodes, features, labels):
    """构建DBLP数据集的子图"""
    target_nodes = copy.deepcopy(target_nodes)
    related_p_nodes = set()
    related_c_nodes = set()
    related_t_nodes = set()

    for a_node in target_nodes:
        if 'ap' in hg.etypes:
            related_p_nodes.update(hg.successors(a_node, etype='ap').tolist())
        if 'pa' in hg.etypes:
            related_p_nodes.update(hg.predecessors(a_node, etype='pa').tolist())

    for p_node in related_p_nodes:
        if 'pc' in hg.etypes:
            related_c_nodes.update(hg.successors(p_node, etype='pc').tolist())
        if 'pt' in hg.etypes:
            related_t_nodes.update(hg.successors(p_node, etype='pt').tolist())
        if 'cp' in hg.etypes:
            related_c_nodes.update(hg.predecessors(p_node, etype='cp').tolist())
        if 'tp' in hg.etypes:
            related_t_nodes.update(hg.predecessors(p_node, etype='tp').tolist())

    if not isinstance(target_nodes, list):
        target_nodes = target_nodes.tolist()

    node_ids_per_type = {
        '0': target_nodes,
        '1': list(related_p_nodes),
        '2': list(related_c_nodes),
        '3': list(related_t_nodes)
    }

    subgraph = dgl.node_subgraph(hg, node_ids_per_type)
    subgraph.ndata['features'] = {'0': features[target_nodes]}
    subgraph.ndata['labels'] = {'0': labels[target_nodes]}

    return subgraph


def get_subgraph_imdb(hg, target_nodes, features, labels):
    """构建IMDB数据集的子图"""
    target_nodes = copy.deepcopy(target_nodes)
    related_d_nodes = set()
    related_a_nodes = set()
    related_k_nodes = set()
    
    for m_node in target_nodes:
        for etype in ['md', 'dm']:
            if etype in hg.etypes:
                if etype == 'md':
                    related_nodes = hg.successors(m_node, etype=etype)
                else:
                    related_nodes = hg.predecessors(m_node, etype=etype)
                related_d_nodes.update(related_nodes.tolist())

        for etype in ['ma', 'am']:
            if etype in hg.etypes:
                if etype == 'ma':
                    related_nodes = hg.successors(m_node, etype=etype)
                else:
                    related_nodes = hg.predecessors(m_node, etype=etype)
                related_a_nodes.update(related_nodes.tolist())

        for etype in ['mk', 'km']:
            if etype in hg.etypes:
                if etype == 'mk':
                    related_nodes = hg.successors(m_node, etype=etype)
                else:
                    related_nodes = hg.predecessors(m_node, etype=etype)
                related_k_nodes.update(related_nodes.tolist())

    if not isinstance(target_nodes, list):
        target_nodes = target_nodes.tolist()

    node_ids_per_type = {
        '0': target_nodes,
        '1': list(related_d_nodes),
        '2': list(related_a_nodes),
        '3': list(related_k_nodes)
    }

    subgraph = dgl.node_subgraph(hg, node_ids_per_type)
    subgraph.ndata['features'] = {'0': features[target_nodes]}
    subgraph.ndata['labels'] = {'0': labels[target_nodes]}

    return subgraph


def get_subgraph_freebase(hg, target_nodes, features, labels):
    """构建Freebase数据集的子图"""
    target_nodes = copy.deepcopy(target_nodes)
    target_ntype = '0'
    related_nodes = {ntype: set() for ntype in hg.ntypes if ntype != target_ntype}

    for target_node in target_nodes:
        for etype in ['01', '10']:
            if etype in hg.etypes:
                if etype == '01':
                    related_nodes1 = hg.successors(target_node, etype=etype)
                else:
                    related_nodes1 = hg.predecessors(target_node, etype=etype)
                related_nodes['1'].update(related_nodes1.tolist())

        for etype in ['04', '40']:
            if etype in hg.etypes:
                if etype == '04':
                    related_nodes4 = hg.successors(target_node, etype=etype)
                else:
                    related_nodes4 = hg.predecessors(target_node, etype=etype)
                related_nodes['4'].update(related_nodes4.tolist())

        for etype in ['07', '70']:
            if etype in hg.etypes:
                if etype == '07':
                    related_nodes7 = hg.successors(target_node, etype=etype)
                else:
                    related_nodes7 = hg.predecessors(target_node, etype=etype)
                related_nodes['7'].update(related_nodes7.tolist())

        if '05' in hg.etypes:
            related_nodes5 = hg.successors(target_node, etype='05')
            related_nodes['5'].update(related_nodes5.tolist())

        if '06' in hg.etypes:
            related_nodes6 = hg.successors(target_node, etype='06')
            related_nodes['6'].update(related_nodes6.tolist())

        if '20' in hg.etypes:
            related_nodes2 = hg.predecessors(target_node, etype='20')
            related_nodes['2'].update(related_nodes2.tolist())

        if '30' in hg.etypes:
            related_nodes3 = hg.predecessors(target_node, etype='30')
            related_nodes['3'].update(related_nodes3.tolist())

    node_ids_per_type = {ntype: list(nodes) for ntype, nodes in related_nodes.items()}
    if not isinstance(target_nodes, list):
        target_nodes = target_nodes.tolist()
    node_ids_per_type[target_ntype] = target_nodes

    subgraph = dgl.node_subgraph(hg, node_ids_per_type)
    subgraph.ndata['features'] = {'0': features[target_nodes]}
    subgraph.ndata['labels'] = {'0': labels[target_nodes]}

    return subgraph


def get_subgraph_func(dataset):
    """获取对应数据集的子图构建函数"""
    subgraph_funcs = {
        'Freebase': get_subgraph_freebase,
        'IMDB': get_subgraph_imdb,
        'DBLP': get_subgraph_dblp
    }
    
    if dataset not in subgraph_funcs:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return subgraph_funcs[dataset]


def get_task_data(dataset, args, task_id, use_cache=True):
    """
    获取任务数据
    
    Args:
        dataset: 完整数据集元组
        args: 参数字典
        task_id: 任务ID
        use_cache: 是否使用缓存
    
    Returns:
        (subgraph, task_features, task_labels, ids_train, ids_val, ids_test)
    """
    cache = get_cache()
    dataset_name = args['dataset']
    num_tasks = args['num_tasks']
    
    # 检查缓存
    if use_cache and cache.has_task_data(dataset_name, 0, task_id, num_tasks):
        return cache.load_task_data(dataset_name, 0, task_id, num_tasks)
    
    hg, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, cls_node_ids, meta_paths = dataset

    label_ids = get_label_ids(dataset_name)
    n_cls_per_task = num_classes // num_tasks
    task_classes = label_ids[task_id * n_cls_per_task:(task_id + 1) * n_cls_per_task]

    task_train_idx = train_idx[np.isin(labels[train_mask], task_classes)]
    task_val_idx = val_idx[np.isin(labels[val_mask], task_classes)]
    task_test_idx = test_idx[np.isin(labels[test_mask], task_classes)]

    task_train_idx = torch.LongTensor(task_train_idx)
    task_val_idx = torch.LongTensor(task_val_idx)
    task_test_idx = torch.LongTensor(task_test_idx)

    task_nodes = torch.cat([task_train_idx, task_val_idx, task_test_idx]).unique()

    get_subgraph = get_subgraph_func(dataset_name)
    subgraph = get_subgraph(hg, task_nodes, features, labels)

    task_features = features[task_nodes]
    task_labels = labels[task_nodes]

    # 标签重映射
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(task_classes)}
    task_labels = torch.LongTensor([label_mapping[label.item()] for label in task_labels])

    # 映射原始ID到子图ID
    def map_ids(subgraph, original_ids, ntype):
        subgraph_ids = subgraph.nodes[ntype].data['_ID']
        id_map = {orig_id.item(): new_id for new_id, orig_id in enumerate(subgraph_ids)}
        mapped_ids = [id_map[id.item()] for id in original_ids if id.item() in id_map]
        return torch.LongTensor(mapped_ids)

    ids_train = map_ids(subgraph, task_train_idx, '0')
    ids_val = map_ids(subgraph, task_val_idx, '0')
    ids_test = map_ids(subgraph, task_test_idx, '0')

    if dataset_name == 'Freebase':
        task_features = task_features.to_sparse()

    result = (subgraph, task_features, task_labels, ids_train, ids_val, ids_test)
    
    # 保存缓存
    if use_cache:
        cache.save_task_data(dataset_name, 0, task_id, num_tasks, result)
    
    return result
