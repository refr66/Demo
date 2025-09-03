import torch
import numpy as np
from typing import List, Dict, Any, Union, Callable

class Calibrator:
    """
    模型校准器，用于收集模型激活值的统计信息
    """
    def __init__(self, method: str = 'minmax', percentile: float = 99.9, 
                 num_bins: int = 2048):
        """
        初始化校准器
        
        Args:
            method: 校准方法，支持'minmax'、'percentile'或'histogram'
            percentile: 当method为'percentile'时使用的百分比值
            num_bins: 当method为'histogram'时使用的直方图桶数
        """
        self.method = method
        self.percentile = percentile
        self.num_bins = num_bins
        self.stats = {}
    
    def collect_activations(self, model: torch.nn.Module, 
                           calibration_data: List[torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """
        收集模型各层的激活值统计信息
        
        Args:
            model: PyTorch模型
            calibration_data: 校准数据列表
        
        Returns:
            各层激活值的统计信息
        """
        hooks = []
        self.stats = {}
        
        def hook_fn(module_name):
            def hook(module, input, output):
                if module_name not in self.stats:
                    self.stats[module_name] = {'activations': []}
                
                # 处理输出，保存激活值
                if isinstance(output, torch.Tensor):
                    output = [output]
                
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        # 保存激活值样本
                        out_data = out.data.cpu().numpy()
                        # 为了节省内存，我们可以采样部分数据
                        if out_data.ndim <= 2:
                            # 对于1D或2D张量，直接保存
                            self.stats[module_name]['activations'].append(out_data)
                        else:
                            # 对于高维张量，沿第一个维度采样
                            indices = np.random.choice(out_data.shape[0], min(10, out_data.shape[0]), 
                                                      replace=False)
                            self.stats[module_name]['activations'].append(out_data[indices])
            return hook
        
        # 为每个模块注册钩子
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, 
                                  torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 运行校准数据
        model.eval()
        with torch.no_grad():
            for data in calibration_data:
                if isinstance(data, (list, tuple)):
                    model(*data)
                else:
                    model(data)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return self.stats
    
    def calculate_range(self, activations: List[np.ndarray]) -> Tuple[float, float]:
        """
        根据收集的激活值计算量化范围
        
        Args:
            activations: 激活值列表
        
        Returns:
            量化范围的最小值和最大值
        """
        # 合并所有激活值
        all_activations = np.concatenate([a.flatten() for a in activations])
        
        if self.method == 'minmax':
            # 使用最小值和最大值
            min_val = np.min(all_activations)
            max_val = np.max(all_activations)
        elif self.method == 'percentile':
            # 使用百分位数
            min_val = np.percentile(all_activations, 100 - self.percentile)
            max_val = np.percentile(all_activations, self.percentile)
        elif self.method == 'histogram':
            # 使用直方图裁剪异常值
            hist, bin_edges = np.histogram(all_activations, bins=self.num_bins)
            # 计算累积分布
            cdf = np.cumsum(hist) / np.sum(hist)
            # 找到包含大部分值的范围
            lower_idx = np.where(cdf >= 0.001)[0][0]
            upper_idx = np.where(cdf >= 0.999)[0][0]
            min_val = bin_edges[lower_idx]
            max_val = bin_edges[upper_idx + 1]
        else:
            raise ValueError(f"不支持的校准方法: {self.method}")
        
        # 处理零范围情况
        if abs(max_val - min_val) < 1e-8:
            max_val = min_val + 1e-8
            
        return min_val, max_val
    
    def calibrate(self, model: torch.nn.Module, 
                 calibration_data: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        校准模型，计算各层的量化范围
        
        Args:
            model: PyTorch模型
            calibration_data: 校准数据列表
        
        Returns:
            各层的量化范围信息
        """
        # 收集激活值
        self.collect_activations(model, calibration_data)
        
        # 计算每层的量化范围
        calibration_ranges = {}
        for name, stats in self.stats.items():
            if 'activations' in stats and stats['activations']:
                min_val, max_val = self.calculate_range(stats['activations'])
                calibration_ranges[name] = {
                    'min': min_val,
                    'max': max_val
                }
        
        return calibration_ranges
    
    def register_calibrated_ranges(self, model: torch.nn.Module, 
                                  ranges: Dict[str, Dict[str, float]]) -> None:
        """
        将校准范围注册到模型中
        
        Args:
            model: PyTorch模型
            ranges: 各层的量化范围
        """
        # 在模型上创建一个属性来存储校准范围
        if not hasattr(model, 'calibration_ranges'):
            model.calibration_ranges = {}
        
        # 更新校准范围
        model.calibration_ranges.update(ranges)


# 预定义的校准器

def create_minmax_calibrator() -> Calibrator:
    """
    创建使用minmax方法的校准器
    """
    return Calibrator(method='minmax')


def create_percentile_calibrator(percentile: float = 99.9) -> Calibrator:
    """
    创建使用percentile方法的校准器
    
    Args:
        percentile: 百分比值
    """
    return Calibrator(method='percentile', percentile=percentile)


def create_histogram_calibrator(num_bins: int = 2048) -> Calibrator:
    """
    创建使用histogram方法的校准器
    
    Args:
        num_bins: 直方图桶数
    """
    return Calibrator(method='histogram', num_bins=num_bins)


# 便捷函数
def calibrate_model(model: torch.nn.Module, calibration_data: List[torch.Tensor], 
                   method: str = 'minmax', **kwargs) -> Dict[str, Dict[str, float]]:
    """
    校准模型的便捷函数
    
    Args:
        model: PyTorch模型
        calibration_data: 校准数据列表
        method: 校准方法
        **kwargs: 传递给校准器的额外参数
    
    Returns:
        各层的量化范围
    """
    # 创建校准器
    if method == 'minmax':
        calibrator = create_minmax_calibrator()
    elif method == 'percentile':
        percentile = kwargs.get('percentile', 99.9)
        calibrator = create_percentile_calibrator(percentile)
    elif method == 'histogram':
        num_bins = kwargs.get('num_bins', 2048)
        calibrator = create_histogram_calibrator(num_bins)
    else:
        raise ValueError(f"不支持的校准方法: {method}")
    
    # 执行校准
    ranges = calibrator.calibrate(model, calibration_data)
    
    # 注册校准范围
    calibrator.register_calibrated_ranges(model, ranges)
    
    return ranges


def get_calibration_stats(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    获取模型的校准统计信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        校准统计信息
    """
    if hasattr(model, 'calibration_ranges'):
        return model.calibration_ranges
    else:
        return {}


def set_calibration_stats(model: torch.nn.Module, 
                          stats: Dict[str, Dict[str, Any]]) -> None:
    """
    设置模型的校准统计信息
    
    Args:
        model: PyTorch模型
        stats: 校准统计信息
    """
    model.calibration_ranges = stats