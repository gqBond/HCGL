"""
Training and Evaluation Utils - 训练和评估的通用函数
"""
import torch
from sklearn.metrics import f1_score


def accuracy(logits, labels, cls_balance=True, ids_per_cls=None):
    """
    计算准确率
    
    Args:
        logits: 模型输出
        labels: 真实标签
        cls_balance: 是否进行类别平衡
        ids_per_cls: 每个类别的样本索引
    
    Returns:
        准确率
    """
    if cls_balance:
        _, indices = torch.max(logits, dim=1)
        acc_per_cls = [torch.sum((indices == labels)[ids]) / len(ids) for ids in ids_per_cls]
        acc = sum(acc_per_cls).item() / len(acc_per_cls)
    else:
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        labels = labels.cpu().numpy()
        acc = (prediction == labels).sum() / len(prediction)
    
    return acc


def get_ids_per_cls(nodes_idx, labels):
    """
    获取每个类别的样本索引
    
    Args:
        nodes_idx: 节点索引
        labels: 标签
    
    Returns:
        每个类别的样本索引列表
    """
    label_set = set(labels[nodes_idx].cpu().numpy())
    ids_per_cls = []
    for l in sorted(label_set):
        cls_ids = [idx for idx in nodes_idx.cpu().numpy() if labels[idx].item() == l]
        ids_per_cls.append(cls_ids)
    return ids_per_cls


def evaluate_model(model, g, features, labels, mask, loss_fcn, args, ids_per_cls=None):
    """
    评估模型
    
    Args:
        model: 模型
        g: 图
        features: 特征
        labels: 标签
        mask: 掩码
        loss_fcn: 损失函数
        args: 参数字典
        ids_per_cls: 每个类别的样本索引
    
    Returns:
        (loss, accuracy)
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(g, features)
        loss = loss_fcn(logits[mask], labels[mask])
        
        if args['metrics'] == 'f1':
            _, indices = torch.max(logits[mask], dim=1)
            prediction = indices.long().cpu().numpy()
            labels_val = labels[mask].cpu().numpy()
            acc = f1_score(labels_val, prediction, average='macro')
        elif args.get('cls_balance', False):
            acc = accuracy(logits, labels, cls_balance=True, ids_per_cls=ids_per_cls)
        else:
            acc = accuracy(logits[mask], labels[mask], cls_balance=False, ids_per_cls=ids_per_cls)
    
    return loss.item(), acc


class TrainingManager:
    """训练管理器 - 统一管理不同方法的训练流程"""
    
    def __init__(self, model, life_model, loss_fcn, args):
        """
        初始化训练管理器
        
        Args:
            model: 模型
            life_model: 终身学习模型
            loss_fcn: 损失函数
            args: 参数字典
        """
        self.model = model
        self.life_model = life_model
        self.loss_fcn = loss_fcn
        self.args = args
        self.method = args['method']
    
    def train_epoch(self, g, features, labels, task_id, train_mask, 
                   ids_per_cls_train=None, dataset=None, hg=None, 
                   features_full=None, labels_full=None):
        """
        训练一个epoch
        
        Args:
            g: 图
            features: 特征
            labels: 标签
            task_id: 任务ID
            train_mask: 训练掩码
            ids_per_cls_train: 每个类别的训练样本索引
            dataset: 完整数据集（某些方法需要）
            hg: 完整图（某些方法需要）
            features_full: 完整特征（某些方法需要）
            labels_full: 完整标签（某些方法需要）
        
        Returns:
            训练损失
        """
        if self.method == 'finetune':
            loss = self.life_model.observe(g, features, labels, task_id, train_mask)
        
        elif self.method in ['dmsg', 'ftf_er', 'metacl_graph']:
            loss = self.life_model.observe(g, hg, features_full, features, 
                                          labels_full, labels, task_id,
                                          train_mask, ids_per_cls_train)
        
        elif self.method in ['hero', 'ergnn']:
            loss = self.life_model.observe(g, features, labels, task_id, 
                                          train_mask, ids_per_cls_train)
        
        elif self.method in ['ewc', 'mas', 'twp']:
            loss = self.life_model.observe(g, features, labels, task_id, 
                                          train_mask, dataset)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return loss
    
    def validate(self, g, features, labels, val_mask, ids_per_cls_val):
        """
        验证模型
        
        Args:
            g: 图
            features: 特征
            labels: 标签
            val_mask: 验证掩码
            ids_per_cls_val: 每个类别的验证样本索引
        
        Returns:
            (val_loss, val_acc)
        """
        return evaluate_model(self.model, g, features, labels, val_mask, 
                            self.loss_fcn, self.args, ids_per_cls_val)
    
    def test(self, g, features, labels, test_mask, ids_per_cls_test):
        """
        测试模型
        
        Args:
            g: 图
            features: 特征
            labels: 标签
            test_mask: 测试掩码
            ids_per_cls_test: 每个类别的测试样本索引
        
        Returns:
            (test_loss, test_acc)
        """
        return evaluate_model(self.model, g, features, labels, test_mask,
                            self.loss_fcn, self.args, ids_per_cls_test)
