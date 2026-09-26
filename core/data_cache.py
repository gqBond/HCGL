"""
Data Cache Manager - 用于缓存和加载处理后的数据，避免重复计算
"""
import os
import pickle
import hashlib
import torch
import dgl
from pathlib import Path


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir='./cache'):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, dataset_name, feat_type, task_id=None, prefix=''):
        """
        生成缓存键
        
        Args:
            dataset_name: 数据集名称
            feat_type: 特征类型
            task_id: 任务ID（可选）
            prefix: 前缀（可选）
        """
        key_parts = [prefix, dataset_name, f'feat{feat_type}']
        if task_id is not None:
            key_parts.append(f'task{task_id}')
        key = '_'.join(key_parts)
        return key
    
    def _get_cache_path(self, cache_key):
        """获取缓存文件路径"""
        return self.cache_dir / f'{cache_key}.pkl'
    
    def exists(self, cache_key):
        """检查缓存是否存在"""
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()
    
    def save(self, cache_key, data):
        """
        保存数据到缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        cache_path = self._get_cache_path(cache_key)
        print(f'Saving cache to {cache_path}...')
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f'Cache saved successfully.')
    
    def load(self, cache_key):
        """
        从缓存加载数据
        
        Args:
            cache_key: 缓存键
        
        Returns:
            缓存的数据
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            raise FileNotFoundError(f'Cache file not found: {cache_path}')
        
        print(f'Loading cache from {cache_path}...')
        
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f'Cache loaded successfully.')
        return data
    
    def save_full_dataset(self, dataset_name, feat_type, data):
        """
        保存完整数据集
        
        Args:
            dataset_name: 数据集名称
            feat_type: 特征类型
            data: 数据集元组 (hg, features, labels, ...)
        """
        cache_key = self._get_cache_key(dataset_name, feat_type, prefix='full')
        self.save(cache_key, data)
    
    def load_full_dataset(self, dataset_name, feat_type):
        """
        加载完整数据集
        
        Args:
            dataset_name: 数据集名称
            feat_type: 特征类型
        
        Returns:
            数据集元组
        """
        cache_key = self._get_cache_key(dataset_name, feat_type, prefix='full')
        return self.load(cache_key)
    
    def has_full_dataset(self, dataset_name, feat_type):
        """检查完整数据集缓存是否存在"""
        cache_key = self._get_cache_key(dataset_name, feat_type, prefix='full')
        return self.exists(cache_key)
    
    def save_task_data(self, dataset_name, feat_type, task_id, num_tasks, data):
        """
        保存任务数据
        
        Args:
            dataset_name: 数据集名称
            feat_type: 特征类型
            task_id: 任务ID
            num_tasks: 总任务数
            data: 任务数据元组 (subgraph, features, labels, ...)
        """
        cache_key = self._get_cache_key(
            dataset_name, feat_type, task_id=task_id, 
            prefix=f'task_ntask{num_tasks}'
        )
        self.save(cache_key, data)
    
    def load_task_data(self, dataset_name, feat_type, task_id, num_tasks):
        """
        加载任务数据
        
        Args:
            dataset_name: 数据集名称
            feat_type: 特征类型
            task_id: 任务ID
            num_tasks: 总任务数
        
        Returns:
            任务数据元组
        """
        cache_key = self._get_cache_key(
            dataset_name, feat_type, task_id=task_id,
            prefix=f'task_ntask{num_tasks}'
        )
        return self.load(cache_key)
    
    def has_task_data(self, dataset_name, feat_type, task_id, num_tasks):
        """检查任务数据缓存是否存在"""
        cache_key = self._get_cache_key(
            dataset_name, feat_type, task_id=task_id,
            prefix=f'task_ntask{num_tasks}'
        )
        return self.exists(cache_key)
    
    def clear(self, dataset_name=None):
        """
        清理缓存
        
        Args:
            dataset_name: 数据集名称（可选），如果为None则清理所有缓存
        """
        if dataset_name:
            pattern = f'*{dataset_name}*.pkl'
            files = list(self.cache_dir.glob(pattern))
        else:
            files = list(self.cache_dir.glob('*.pkl'))
        
        for file in files:
            file.unlink()
            print(f'Removed cache file: {file}')
        
        print(f'Cleared {len(files)} cache file(s).')


# 全局缓存管理器实例
_global_cache = None


def get_cache(cache_dir=None):
    """获取全局缓存管理器实例"""
    global _global_cache
    if cache_dir is None:
        cache_dir = os.environ.get('CGL_CACHE_ROOT', './cache')
    if _global_cache is None:
        _global_cache = DataCache(cache_dir)
    return _global_cache
