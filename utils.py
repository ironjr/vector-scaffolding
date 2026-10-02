import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
import math

class ErrorBasedInit:
    def __init__(
        self,
        pred,
        gt,
        blur=False,
        sigma=1.0,
        device=None,
    ):
        print("pred shape and gt shape: ", pred.shape, gt.shape)

        # Use PyTorch tensors, optionally move to a device for speed (CPU/GPU)
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred)
        if not torch.is_tensor(gt):
            gt = torch.tensor(gt)
        if device is not None:
            pred = pred.to(device)
            gt = gt.to(device)

        # Compute error map ("potential energy map") using torch
        if pred.shape[0] == 1 and gt.shape[0] == 1:
            error_map = ((gt[0] - pred[0]) ** 2).sum(0)
        else:
            error_map = ((gt - pred) ** 2).sum(-1)
        
        # Optionally smooth error map using PyTorch (Gaussian blur via conv2d)
        if blur:
            channels = 1
            kernel_size = int(2 * round(3.0 * sigma) + 1)
            # Create a 2d gaussian kernel
            grid = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.
            gaussian_1d = torch.exp(-0.5 * (grid / sigma)**2)
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            kernel2d = torch.outer(gaussian_1d, gaussian_1d)
            kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size).to(error_map.device)

            error_map_ = error_map.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            error_map = torch.nn.functional.conv2d(error_map_, kernel2d, padding=kernel_size//2)[0,0]
        self.error_map = error_map

        # Avoid all-zero "deadloop": ensure at least some small nonzero probability for uniform sampling (torch only)
        eps = 1e-8
        flattened = self.error_map.flatten() + eps
        prob = flattened / flattened.sum()
        self.prob = prob
        self.shape = self.error_map.shape


    def __call__(self):
        # Sample an index using error map as probability (torch)
        # Workaround for RuntimeError: number of categories cannot exceed 2^24
        # If the flattened probability distribution is too large, we subsample it
        max_categories = 2**24 - 1
        
        if len(self.prob) > max_categories:
            # Strategy: Downsample the error map to fit within the limit
            # We'll reshape, pool, and then sample from the reduced space
            H, W = self.shape
            scale_factor = math.ceil(math.sqrt(len(self.prob) / max_categories))
            
            # Reshape error map and perform average pooling
            error_map_2d = self.error_map.view(H, W)
            pooled_h = H // scale_factor
            pooled_w = W // scale_factor
            
            # Use adaptive_avg_pool2d to downsample
            pooled_error = torch.nn.functional.adaptive_avg_pool2d(
                error_map_2d.unsqueeze(0).unsqueeze(0),
                (pooled_h, pooled_w)
            )[0, 0]
            
            # Create probability distribution from pooled error map
            eps = 1e-8
            pooled_flat = pooled_error.flatten() + eps
            pooled_prob = pooled_flat / pooled_flat.sum()
            
            # Sample from the pooled distribution
            pooled_idx = torch.multinomial(pooled_prob, 1).item()
            h_pooled, w_pooled = np.unravel_index(pooled_idx, (pooled_h, pooled_w))
            
            # Map back to original coordinates (center of the pooled region)
            h = int((h_pooled + 0.5) * scale_factor)
            w = int((w_pooled + 0.5) * scale_factor)
            
            # Clamp to valid range
            h = min(h, H - 1)
            w = min(w, W - 1)
        else:
            idx = torch.multinomial(self.prob, 1).item()
            h, w = np.unravel_index(idx, self.shape)
        
        return [h, w]


class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss

# #code modified from https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization/blob/679e1d16c5809367f2d2db3e403a8548c5419258/LIVE/xing_loss.py#L22C15-L22C21

    













    


def xing_loss(x_tensor, scale=1e-3):
    """
    x_tensor: a tensor of shape (N, 10, 2), where each row is a curve with 10 points (not closed).
              Since (10 - 1)=9 segments, and 9 % 3 == 0, we can group every 3 consecutive segments.
    The function computes a loss that encourages each group of 3 consecutive segments to have
    consistent turning direction, and it is computed in parallel for all N curves.
    """
    N, n_points, _ = x_tensor.shape
    num_segments = n_points - 1

    remainder = num_segments % 3
    if remainder != 0:
        num_missing = 3 - remainder  # e.g. remainder=1 -> need 2 points
        # 取前几个点补在末尾（保持曲线首尾连续）
        to_add = x_tensor[:, 1:1 + num_missing, :]  # 或 0:1 if你只想补P0
        x_tensor = torch.cat([x_tensor, to_add], dim=1)
        n_points = x_tensor.shape[1]
        num_segments = n_points - 1

    assert num_segments % 3 == 0
    num_groups = num_segments // 3

    segments = torch.stack([x_tensor[:, :-1, :], x_tensor[:, 1:, :]], dim=2)
    segments = segments.view(N, num_groups, 3, 2, 2)

    # For each group, extract the three segments: cs1, cs2, cs3
    # Each has shape: (N, num_groups, 2, 2)
    cs1 = segments[:, :, 0, :, :]
    cs2 = segments[:, :, 1, :, :]
    cs3 = segments[:, :, 2, :, :]

    # Compute direction vectors for each segment.
    # For cs1: v1 = cs1[1] - cs1[0], shape: (N, num_groups, 2)
    v1 = cs1[:, :, 1, :] - cs1[:, :, 0, :]
    v2 = cs2[:, :, 1, :] - cs2[:, :, 0, :]
    v3 = cs3[:, :, 1, :] - cs3[:, :, 0, :]

    # Compute sine of angles:
    # sine_theta between cs1 and cs2:
    cross_12 = v1[..., 0]*v2[..., 1] - v1[..., 1]*v2[..., 0]
    norm1 = torch.norm(v1, dim=-1)
    norm2 = torch.norm(v2, dim=-1)
    sine_theta_12 = cross_12 / (norm1 * norm2 + 1e-8)

    # sine_theta between cs1 and cs3:
    cross_13 = v1[..., 0]*v3[..., 1] - v1[..., 1]*v3[..., 0]
    norm3 = torch.norm(v3, dim=-1)
    sine_theta_13 = cross_13 / (norm1 * norm3 + 1e-8)

    # Determine turning direction from cs1 to cs2.
    # If sine_theta_12 >= 0, direct = 1 (e.g. counter-clockwise); otherwise 0.
    direct = (sine_theta_12 >= 0).float()
    opst = 1 - direct

    # Penalize inconsistency in turning:
    # If direct==1, expect sine_theta_13 to be non-negative, so penalize negative values.
    # If direct==0, expect sine_theta_13 to be non-positive, so penalize positive values.
    loss_groups = direct * torch.relu(-sine_theta_13) + opst * torch.relu(sine_theta_13)
    
    # Average the loss over groups and then over all curves.
    seg_loss = loss_groups.mean(dim=1)  # shape: (N,)
    loss = seg_loss.mean() * scale
    return loss

def curvature_loss(paths: torch.Tensor, H: int, angle_thresh_deg=60.0) -> torch.Tensor:
    assert paths.dim() == 4 and paths.shape[1] == 2, "Expect input shape [N, 2, M, 2]"
    N, _, M, _ = paths.shape
    path1 = paths[:, 0, :, :]              # [N, M, 2]
    path2 = torch.flip(paths[:, 1, :, :], dims=[1])  # [N, M, 2]
    full_path = torch.cat([path1, path2], dim=1)     # [N, 2M, 2]
    total_len = full_path.shape[1]

    indices = torch.arange(0, total_len, H, device=paths.device)  # [0, H, 2H, ...]
    # roll 相邻点
    prev = torch.roll(full_path, 5, dims=1)[:, indices, :]  # [N, K, 2]
    curr = full_path[:, indices, :]
    nex = torch.roll(full_path, -5, dims=1)[:, indices, :]

    # --- Curvature ---
    second_diff = prev - 2 * curr + nex           # [N, K, 2]
    curvature = second_diff.pow(2).sum(dim=-1)    # [N, K]

    # --- Angle ---
    v1 = F.normalize(prev - curr, dim=-1)   # [N, K, 2]
    v2 = F.normalize(nex - curr, dim=-1)
    cos_theta = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_theta)           # [N, K]

    # --- Mask ---
    angle_thresh_rad = math.radians(angle_thresh_deg)
    mask = (angle < angle_thresh_rad).float()  # [N, K]

    # --- Apply curvature only if direction is aligned ---
    loss = (curvature * mask).mean()
    return loss

def boundary_loss_on_joints(points: torch.Tensor, degree: int, bound: float = 1.0) -> torch.Tensor:
    """
    只对 Bézier 曲线的连接点（每隔 degree 个点）做边界约束。

    参数:
        points: Tensor of shape (N, M, 2)
        degree: Bézier 曲线的 degree（如 3）
        bound: 边界，通常为 1.0 表示 [-1, 1]

    返回:
        标量 loss
    """
    assert points.dim() == 3 and points.shape[-1] == 2, "Expect shape (N, M, 2)"
    N, M, _ = points.shape

    # 连接点索引：0, degree, 2*degree, ...
    joint_indices = torch.arange(0, M, degree, device=points.device)  # shape: [K]

    # 取出所有连接点
    joints = points[:, joint_indices, :]  # shape: (N, K, 2)

    # 超出 [-1, 1] 的部分
    over = torch.relu(joints - bound)
    under = torch.relu(-bound - joints)
    excess = over + under  # shape: (N, K, 2)

    return excess.mean()








 



























    



def test_xing_loss_consistent():
    num_points = 10
    radius = 10.0
    angles = torch.linspace(0, math.pi/2, steps=num_points)
    x = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)  # shape: (10, 2)
    x_tensor = x.unsqueeze(0)  # shape: (1, 10, 2)
    
    loss = xing_loss(x_tensor)
    print("Loss for consistent turning curve (expected near 0):", loss.item())

def test_xing_loss_inconsistent():
    num_points = 10
    radius = 10.0
    angles = torch.linspace(0, math.pi/2, steps=num_points)
    x = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    x[5, :] = x[5, :] + torch.tensor([5.0, -5.0])
    x_tensor = x.unsqueeze(0)  # shape: (1, 10, 2)
    
    loss = xing_loss(x_tensor)
    print("Loss for inconsistent turning curve (expected > 0):", loss.item())

#

if __name__ == '__main__':
    print("Test Consistent Turning:")
    test_xing_loss_consistent()
    print("\nTest Inconsistent Turning:")
    test_xing_loss_inconsistent()
