import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.autograd import Variable

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = CodaPromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.reweight_cfg = args.get("reweighting", {})
        self.reweight_enabled = self.reweight_cfg.get("enabled", False)
        if self.reweight_enabled:
            buffer_size = self.reweight_cfg.get("buffer_size", 1024)
            feature_dim = self.reweight_cfg.get("feature_dim", self._network.feature_dim)
            self._reweight_state = SampleReweightState(buffer_size, feature_dim,
                                                       self.reweight_cfg.get("presave_ratio", 0.9),
                                                       device=self._device)
        
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad) + sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')


    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self.reweight_enabled:
            self._reweight_state.to_(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                warmup_epochs = int(self.reweight_cfg.get("warmup_epochs", 0)) if self.reweight_enabled else 0
                start_task = int(self.reweight_cfg.get("start_task", 0)) if self.reweight_enabled else 0
                
                # 判断是否启用重加权
                use_reweight = (self.reweight_enabled 
                               and (self._cur_task >= start_task) 
                               and (epoch >= warmup_epochs))
                
                if use_reweight:
                    logits, prompt_loss, features = self._network(
                        inputs, train=True, return_features=True
                    )
                    cfeatures = features.detach()
                    weight1, self._reweight_state = coda_weight_learner(
                        cfeatures,
                        self._reweight_state,
                        self.reweight_cfg,
                        global_epoch=epoch,
                        inner_iter=i,
                        cur_task=self._cur_task,
                    )
                else:
                    logits, prompt_loss = self._network(inputs, train=True)
                    weight1 = torch.ones(inputs.size(0), 1, device=self._device)
                
                logits = logits[:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')
                
                # 计算分类损失
                base_loss = F.cross_entropy(logits, targets.long(), reduction='none')
                
                # 根据归一化模式和是否启用重加权选择损失计算方式
                if use_reweight:
                    norm_mode = self.reweight_cfg.get("norm", "softplus").lower()
                    weight_squeeze = weight1.squeeze(1)
                    
                    if norm_mode in ["sigmoid_mean", "softplus"]:
                        # 均值归一化：使用加权平均
                        loss_cls = (base_loss * weight_squeeze).mean()
                    else:
                        # Softmax 归一化：权重和为1，使用加权求和
                        loss_cls = torch.sum(base_loss * weight_squeeze)
                    
                    # 可选：对 prompt_loss 也进行加权（默认不加权）
                    prompt_weight = float(self.reweight_cfg.get("prompt_loss_weight", 1.0))
                    loss = loss_cls + prompt_weight * prompt_loss.sum()
                else:
                    loss = base_loss.mean() + prompt_loss.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                else:
                    logits = outputs
                logits = logits[:, :self._total_classes]
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to (self._device)
            with torch.no_grad():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                else:
                    logits = outputs
                logits = logits[:, :self._total_classes]
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]

class SampleReweightState:
    def __init__(self, buffer_size, feature_dim, presave_ratio=0.9, device="cpu"):
        self.features = torch.zeros(buffer_size, feature_dim, device=device)
        self.weights = torch.ones(buffer_size, 1, device=device)
        self.valid = 0
        self.presave_ratio = presave_ratio

    def to_(self, device):
        self.features = self.features.to(device)
        self.weights = self.weights.to(device)
        return self

    def update_bank(self, new_features, new_weights):
        bsz = new_features.size(0)
        if self.features.size(0) < bsz:
            self.features = torch.zeros_like(new_features)
            self.weights = torch.ones(bsz, 1, device=new_weights.device)
        ratio = self.presave_ratio
        if self.valid == 0:
            self.features[:bsz] = new_features
            self.weights[:bsz] = new_weights
            self.valid = bsz
        else:
            end = min(self.valid, bsz)
            self.features[:end] = ratio * self.features[:end] + (1 - ratio) * new_features[:end]
            self.weights[:end] = ratio * self.weights[:end] + (1 - ratio) * new_weights[:end]
            if bsz > end:
                self.features[end:bsz] = new_features[end:bsz]
                self.weights[end:bsz] = new_weights[end:bsz]
                self.valid = bsz

def coda_weight_learner(cfeatures, state, cfg, global_epoch=0, inner_iter=0, cur_task=0):
    """
    改进的权重学习器，专门适配 CODA-Prompt 的持续学习场景。
    
    核心改进：
    1. 使用 softplus 替代 sigmoid，避免梯度饱和
    2. 添加温度参数控制权重分布的平滑度
    3. 引入任务感知的正则化强度
    4. 改进特征池化策略，考虑特征多样性
    """
    bsz = cfeatures.size(0)
    device = cfeatures.device
    
    # 获取历史特征和权重
    prev_feat = state.features[:state.valid] if state.valid > 0 else torch.empty(0, cfeatures.size(1), device=device)
    prev_w = state.weights[:state.valid] if state.valid > 0 else torch.empty(0, 1, device=device)
    
    # 初始化权重参数（使用零初始化，配合 softplus 输出约为 0.693）
    weight = Variable(torch.zeros(bsz, 1, device=device))
    weight.requires_grad = True
    
    # 特征池化：对当前特征做 L2 归一化，提高协方差计算的稳定性
    cfeatures_norm = F.normalize(cfeatures, p=2, dim=1)
    prev_feat_norm = F.normalize(prev_feat, p=2, dim=1) if prev_feat.numel() > 0 else prev_feat
    feature_pool = torch.cat([cfeatures_norm, prev_feat_norm.detach()], dim=0)
    weight_pool = torch.cat([weight, prev_w.detach()], dim=0) if prev_w.numel() else weight
    
    # 优化器设置
    base_lr = cfg.get("lr", 0.1)
    # 任务越多，学习率越保守
    task_decay = cfg.get("task_lr_decay", 0.9) ** cur_task
    optimizer_bl = torch.optim.SGD([weight], lr=base_lr * task_decay, momentum=cfg.get("momentum", 0.9))
    
    inner_epochs = max(1, cfg.get("inner_epochs", 5))
    norm_mode = cfg.get("norm", "softplus").lower()  # "softmax" | "sigmoid_mean" | "softplus"
    clamp_ratio = float(cfg.get("clamp_ratio", 2.0))
    temperature = float(cfg.get("temperature", 1.0))

    for inner_epoch in range(inner_epochs):
        _set_balance_lr(optimizer_bl, inner_epoch, cfg, inner_epochs)
        optimizer_bl.zero_grad()

        if norm_mode == "softplus":
            # Softplus 归一化：输出非负，梯度不饱和
            cur_weight = F.softplus(weight / temperature)
            cur_weight = cur_weight / (cur_weight.mean().detach() + 1e-8)
            if clamp_ratio > 1.0:
                cur_weight = torch.clamp(cur_weight, 1.0 / clamp_ratio, clamp_ratio)
                cur_weight = cur_weight / (cur_weight.mean().detach() + 1e-8)

            pooled_weight = F.softplus(weight_pool / temperature)
            pooled_weight = pooled_weight / (pooled_weight.mean().detach() + 1e-8)
            if clamp_ratio > 1.0:
                pooled_weight = torch.clamp(pooled_weight, 1.0 / clamp_ratio, clamp_ratio)
                pooled_weight = pooled_weight / (pooled_weight.mean().detach() + 1e-8)
        elif norm_mode == "sigmoid_mean":
            cur_weight = torch.sigmoid(weight / temperature)
            cur_weight = cur_weight / (cur_weight.mean().detach() + 1e-8)
            if clamp_ratio > 1.0:
                cur_weight = torch.clamp(cur_weight, 1.0 / clamp_ratio, clamp_ratio)
                cur_weight = cur_weight / (cur_weight.mean().detach() + 1e-8)

            pooled_weight = torch.sigmoid(weight_pool / temperature)
            pooled_weight = pooled_weight / (pooled_weight.mean().detach() + 1e-8)
            if clamp_ratio > 1.0:
                pooled_weight = torch.clamp(pooled_weight, 1.0 / clamp_ratio, clamp_ratio)
                pooled_weight = pooled_weight / (pooled_weight.mean().detach() + 1e-8)
        else:
            # Softmax 归一化（原版）
            softmax = nn.Softmax(dim=0)
            cur_weight = softmax(weight / temperature)
            pooled_weight = softmax(weight_pool / temperature)

        # 计算去相关损失
        loss_b = lossb_expect(feature_pool, pooled_weight, num_f=cfg.get("num_fourier", 1), use_sum=cfg.get("use_sum", True))
        
        # 正则化损失：根据归一化模式调整
        decay_pow = cfg.get("decay_pow", 2.0)
        if norm_mode in ["sigmoid_mean", "softplus"]:
            # 让权重趋向于1，而非趋向于0
            loss_p = torch.sum((cur_weight - 1.0).pow(decay_pow))
        else:
            loss_p = torch.sum(cur_weight.pow(decay_pow))

        # Lambda 缩放：考虑任务进度
        lambda_coef = cfg.get("lambda_coef", 70.0)
        decay_factor = cfg.get("lambda_decay_rate", 1.0) ** (global_epoch // max(1, cfg.get("lambda_decay_epoch", 5)))
        # 任务越多，去相关约束越弱（保持可塑性）
        task_lambda_scale = cfg.get("task_lambda_scale", 1.0) ** cur_task
        lambda_scaled = max(cfg.get("min_lambda", 0.01), decay_factor) * lambda_coef * task_lambda_scale
        
        # 正则化权重：随任务增加而减弱
        reg_weight = cfg.get("reg_weight", 0.1) * (cfg.get("reg_task_decay", 0.9) ** cur_task)
        
        loss_g = loss_b / lambda_scaled + reg_weight * loss_p
        if global_epoch == 0:
            loss_g = loss_g * cfg.get("first_step_cons", 1.0)
        
        loss_g.backward(retain_graph=True)
        optimizer_bl.step()
        weight_pool = torch.cat([weight, prev_w.detach()], dim=0) if prev_w.numel() else weight

    # 生成最终权重
    with torch.no_grad():
        if norm_mode == "softplus":
            final_weight = F.softplus(weight / temperature)
            final_weight = final_weight / (final_weight.mean() + 1e-8)
            if clamp_ratio > 1.0:
                final_weight = torch.clamp(final_weight, 1.0 / clamp_ratio, clamp_ratio)
                final_weight = final_weight / (final_weight.mean() + 1e-8)
        elif norm_mode == "sigmoid_mean":
            final_weight = torch.sigmoid(weight / temperature)
            final_weight = final_weight / (final_weight.mean() + 1e-8)
            if clamp_ratio > 1.0:
                final_weight = torch.clamp(final_weight, 1.0 / clamp_ratio, clamp_ratio)
                final_weight = final_weight / (final_weight.mean() + 1e-8)
        else:
            final_weight = nn.Softmax(dim=0)(weight / temperature)

    # 更新特征库（使用原始特征，非归一化版本）
    state.update_bank(cfeatures.detach(), final_weight.detach())
    if global_epoch == 0 and inner_iter < cfg.get("bootstrap_iters", 10):
        state.features[:bsz] = (state.features[:bsz] * inner_iter + cfeatures.detach()) / (inner_iter + 1)
        state.weights[:bsz] = (state.weights[:bsz] * inner_iter + final_weight.detach()) / (inner_iter + 1)
        state.valid = max(state.valid, bsz)

    return final_weight, state

def _set_balance_lr(optimizer, epoch, cfg, total_epochs):
    base_lr = cfg.get("lr", 0.1)
    step = max(1, int(total_epochs * 0.5))
    decay = (epoch // step)
    lr = base_lr * (0.1 ** decay)
    for group in optimizer.param_groups:
        group["lr"] = lr

def lossb_expect(cfeaturec, weight, num_f=1, use_sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, use_sum=use_sum)
    loss = cfeaturec.new_tensor(0.0)
    for i in range(cfeaturecs.size(-1)):
        slice_feat = cfeaturecs[:, :, i]
        cov_matrix = cov(slice_feat, weight)
        cov_square = cov_matrix * cov_matrix
        loss = loss + torch.sum(cov_square) - torch.trace(cov_square)
    return loss

def cov(x, w=None):
    if w is None or w.numel() == 0:
        n = x.shape[0]
        cov_mat = torch.matmul(x.t(), x) / max(1, n)
        e = torch.mean(x, dim=0).view(-1, 1)
    else:
        w = w.view(-1, 1)
        cov_mat = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
    return cov_mat - torch.matmul(e, e.t())

def random_fourier_features_gpu(x, num_f=1, use_sum=True, sigma=None):
    if sigma is None or sigma == 0:
        sigma = 1.0
    n, r = x.size()
    w = (1 / sigma) * torch.randn(num_f, 1, device=x.device)
    b = 2 * math.pi * torch.rand(n, r, num_f, device=x.device)
    mid = torch.matmul(x.view(n, r, 1), w.t()) + b
    mid = mid - mid.min(dim=1, keepdim=True).values
    mid = mid / (mid.max(dim=1, keepdim=True).values + 1e-6)
    mid = mid * (math.pi / 2.0)
    scale = math.sqrt(2.0 / num_f)
    if use_sum:
        return scale * (torch.cos(mid) + torch.sin(mid))
    return scale * torch.cat([torch.cos(mid), torch.sin(mid)], dim=-1)