from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize import  rasterize_gaussians
from utils import *
import torch
import torch.nn as nn
import math
from optimizer import Adan
import torch.nn.functional as F
import torch.distributions as dist





class FeatureAreaModulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_mask):
        ctx.save_for_backward(weight_mask)
        return input  # just pass through forward

    @staticmethod
    def backward(ctx, grad_output):
        weight_mask, = ctx.saved_tensors
        return grad_output * weight_mask, None  # apply weight to grad only

# 


class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.ori_H, self.ori_W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.iter = 0
        self.device = kwargs["device"]
        self.mode = kwargs['mode']
        self.num_curves_init = 128
        self.num_curves = kwargs["num_curves"]
        if self.mode == 'closed':
            self.num_beziers = 2 * 1
        else:
            self.num_beziers = 3
        self.opacity_mode = 1 # 0 is single opacity, 1 is multi opacity, 1 only works for line-mode
        self.bezier_degree = kwargs["bezier_degree"]
        self.xing_weight = kwargs.get("xing_weight", 2e-2)
    
        self.curves_resolution = 40
        self.max_sh_degree = 1
        self.radius = 0.01
        if self.mode == "line":
            self.total_num_sample= self.num_samples
        elif self.mode == "unclosed":
            self.num_samples = 64
            self.total_num_sample= self.num_samples * self.num_beziers
            self.radius = 0.01
        elif self.mode == "closed":
            self.num_samples = kwargs["num_samples"]
            print("default num_samples: ", self.num_samples)
            self.total_num_sample = self.num_samples * self.num_beziers
        else:
            self.num_samples = 32
            self.total_num_sample= self.num_samples * self.num_beziers + self.curves_resolution**2
        
        self.rotation_activation = torch.sigmoid

        if self.mode == "line":
            self._control_points = self._initialize_control_points_line()
        elif self.mode == "closed":
            self._control_points = self._initialize_control_points()
        elif self.mode == "unclosed":
             self._control_points = self._initialize_control_points_line((self.bezier_degree * 3) - 1)
        else:
            self._control_points = self._initialize_control_points()
        self._features_dc = nn.Parameter(torch.rand(self.num_curves, 3))
        self._cholesky = nn.Parameter(torch.rand(self.num_curves, 3))

        self._scaling = nn.Parameter(torch.ones(self.num_curves, 1) * 2)
        self._rotation = nn.Parameter(torch.zeros(self.num_curves, 1))

        self._xyz = nn.Parameter(torch.zeros(self.num_curves, self.num_samples, 2))
        depth = (self.num_curves - torch.arange(self.num_curves, device=self.device).unsqueeze(1) - 1) / self.num_curves
        self.register_buffer('_depth', depth)
        if self.opacity_mode == 1:
            self._opacity = nn.Parameter(torch.ones(self.num_curves, 3))
        else:
            self._opacity = nn.Parameter(torch.ones(self.num_curves, 1))
        self._bernstein_cache = {}

        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)


        # -------- lr scales --------
        if self.mode == "unclosed":
            lr_cp, lr_feat, lr_opacity = 0.1, 0.5, 10.0
        else:
            lr_cp, lr_feat, lr_opacity = 0.02, 1.0, 1.0

        # -------- optimizer params --------
        l = [
            {'params': [self._control_points], 'lr': kwargs["lr"] * lr_cp, "name": "control_points"},
            {'params': [self._features_dc], 'lr': kwargs["lr"] * lr_feat, "name": "features_dc"},
            {'params': [self._cholesky], 'lr': kwargs["lr"], "name": "cholesky"},
            {'params': [self._scaling], 'lr': kwargs["lr"], "name": "scaling"},
            {'params': [self._opacity], 'lr': kwargs["lr"] * lr_opacity, "name": "opacity"},
        ]

        # -------- optimizer --------
        self.optimizer = (
            torch.optim.Adam(l, lr=kwargs["lr"])
            if kwargs["opt_type"] == "adam"
            else Adan(l, lr=kwargs["lr"])
        )


        for deg in (self.bezier_degree + 1, self.bezier_degree):
            for mul in (1, 2, 4, 8, 16):
                self._update_bernstein_cache(deg, self.num_samples * mul, self.device)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)
    
    def _update_bernstein_cache(self, n: int, num_samples: int, device: torch.device):
        key = (n, num_samples, str(device))
        if key in self._bernstein_cache:
            return

        t = torch.linspace(0.007, 0.993, num_samples, device=device)
        comb = torch.tensor([math.comb(n, i) for i in range(n + 1)],
                            dtype=torch.float32, device=device)
        t_pow = t[:, None] ** torch.arange(n + 1, dtype=torch.float32, device=device)
        one_minus_t_pow = (1 - t[:, None]) ** torch.arange(n, -1, -1, dtype=torch.float32, device=device)
        bernstein = comb * one_minus_t_pow * t_pow  # (num_samples, n + 1)

        self._bernstein_cache[key] = {
            'bernstein': bernstein,  # (num_samples, n + 1)
            # 'bernstein_deriv': bernstein_deriv  # (num_samples, n)
        }
    
    def _initialize_control_points(self):
        """
        Initialize control points for Bézier curves with an initial convex shape,
        distributed across the entire image.

        Returns:
        - control_points: A tensor of shape (num_curves, 12, 2).
        """
        num_segments = self.num_beziers  # Each curve has 3 Bézier segments
        num_points_per_curve = num_segments * (self.bezier_degree + 1)

        # Step 1: Generate random angles and radii for each curve
        angles = torch.linspace(0, 2 * torch.pi, num_points_per_curve, device=self.device)
        angles = angles.unsqueeze(0).expand(self.num_curves, -1)  # Shape: (num_curves, 13)

        # Generate random curve centers within a normalized space [-1, 1]
        x_center = (torch.rand(self.num_curves, 1, 2, device=self.device) - 0.5) * 2  # Shape: (num_curves, 1, 2)
        # Convert polar coordinates to Cartesian relative to center

        with torch.no_grad():
            centers_flat = x_center[:, 0, :]
            pairwise_distances = torch.cdist(centers_flat, centers_flat, p=2)
            pairwise_distances[
                torch.arange(self.num_curves, device=self.device),
                torch.arange(self.num_curves, device=self.device),
            ] = float('inf')
            closest_distances, _ = pairwise_distances.min(dim=1)
            mean_mutual_dist = closest_distances.mean().item()

        self.last_radii = mean_mutual_dist * 0.5
        radii = (torch.rand(self.num_curves, num_points_per_curve, device=self.device) * 0.5 + 0.5) * self.last_radii
        x = x_center[:, :, 0] + radii * torch.cos(angles)  # Shape: (num_curves, num_points_per_curve, 1)
        y = x_center[:, :, 1] + radii * torch.sin(angles)  # Shape: (num_curves, num_points_per_curve, 1)

        # Combine x and y coordinates
        points = torch.stack([x, y], dim=-1)   # Shape: (num_curves, num_points_per_curve, 2)

        # Step 2: Add small perturbations to ensure diversity while maintaining convexity
        perturbation = torch.randn_like(points) * (self.radius * 0.05)
        points = points + perturbation

        # Step 3: Ensure the curve is closed (last point == first point)
        points[:, -1] = points[:, 0]

        # Step 4: Organize into control points (12 points per curve, 4 Bézier segments)
        control_points = points[:, :-1]  # Remove the repeated last point
        control_points = points

        return torch.nn.Parameter(control_points)
    
    def _initialize_control_points_line(self, order_beizer=2):
        """
        Initialize control points for the Bézier curves using vectorized operations.

        Returns:
        - control_points: A tensor of shape (num_curves, 4, 2).
        """
        # Step 1: Generate p0 (random values in range [0, 1])
        p0 = (torch.rand(self.num_curves, 1, 2, device=self.device) - 0.5) * 2

        # Step 2: Generate random offsets for p1, p2, p3 relative to the previous point
        offsets = (torch.rand(self.num_curves, order_beizer + 1, 2, device=self.device) * 0.5 + 0.5) * self.radius


        # Step 3: Accumulate offsets to get relative positions
        relative_points = torch.cumsum(offsets, dim=1)

        # Step 4: Concatenate p0 and relative points to form the control points
        control_points = torch.cat([p0, p0 + relative_points], dim=1)
        print("control_points shape: ", control_points.shape)

        return torch.nn.Parameter(control_points)
    
    def _initialize_control_points_with_center(self, centers, radii=0.02):
        """
        Initialize control points for Bézier curves with an initial convex shape,
        distributed across the entire image.

        Returns:
        - control_points: A tensor of shape (num_curves, 12, 2).
        """
        x_centers = centers.clone()
        x_centers[:,:, 1] = (centers[:,:, 0] / self.H - 0.5) * 2
        x_centers[:,:, 0] = (centers[:,:, 1] / self.W - 0.5) * 2
        num_segments = self.num_beziers  # Each curve has 3 Bézier segments
        if self.mode == 'unclosed':
            num_points_per_curve = num_segments * self.bezier_degree + 1  
            num_offsets = num_points_per_curve - 1      
            p0 = x_centers  # (N, 1, 2)
            num_left = num_offsets // 2
            num_right = num_offsets - num_left
            offsets_left = (torch.rand(x_centers.shape[0], num_left, 2, device=self.device) * 0.5 + 0.5) * 0.005
            offsets_right = (torch.rand(x_centers.shape[0], num_right, 2, device=self.device) * 0.5 + 0.5) * 0.005
            relative_left = -torch.cumsum(offsets_left, dim=1)
            relative_right = torch.cumsum(offsets_right, dim=1)
            points_left = p0 + relative_left.flip(dims=[1])  # (N, num_left, 2)
            points_right = p0 + relative_right               # (N, num_right, 2)
            control_points = torch.cat([points_left, p0, points_right], dim=1)  # (N, num_points, 2)
            return torch.nn.Parameter(control_points)
        else:
            num_points_per_curve = num_segments * (self.bezier_degree + 1)  # Total: 13 points (last point repeats the first)
            num_curves = centers.shape[0]

            # Step 1: Generate random angles and radii for each curve
            angles = torch.linspace(0, 2 * torch.pi, num_points_per_curve, device=self.device)
            angles = angles.unsqueeze(0).expand(num_curves, -1)  # Shape: (num_curves, 13)
            x = x_centers[:, :, 0] + radii * torch.cos(angles)  # Shape: (num_curves, num_points_per_curve, 1)
            y = x_centers[:, :, 1] + radii * torch.sin(angles)  # Shape: (num_curves, num_points_per_curve, 1)
            points = torch.stack([x, y], dim=-1)   # Shape: (num_curves, num_points_per_curve, 2)
            points[:, -1] = points[:, 0]
            control_points = points[:, :-1]  # Remove the repeated last point
            control_points = points
            return torch.nn.Parameter(control_points)
    

    def get_scaling(self, factor=1):
        if self.mode == 'closed':
            return self.get_scaling_closed(factor)
        else:
            return self.get_scaling_open()

    def get_scaling_closed(self, factor):
        xyz = torch.cat([self.xyz, self.xyz_area], dim=1).detach()
        N = xyz.shape[1] 
        diffs = torch.abs(xyz[:, :, 1:, :] - xyz[:, :, :-1, :])
        scale = torch.tensor([self.W * factor, self.H * factor], device=diffs.device).view(1, 1, 1, 2)
        diffs = diffs * scale
           
        sigma = torch.norm(diffs, dim=-1)
        sigma_last  = sigma[:, :, -2:-1].clone()  # shape: [N, 1] 
        sigma_x = torch.cat([sigma, sigma_last], dim=-1) / (3.0 / torch.sqrt(torch.tensor(factor, dtype=torch.float32)))
        scale = torch.tensor([0.4, 0.9, 1.0], device=sigma_x.device).view(1, 1, 3)
        sigma_x[:, :, :3] *= scale
        sigma_x[:, :, -3:] *= scale.flip(dims=[2])
        sigma_x[:, :2, :].clamp_(min=0.3)

        index_order = torch.arange(2, N, device=xyz.device)
        index_order = torch.cat([torch.tensor([0], device=xyz.device), index_order, torch.tensor([1], device=xyz.device)])
        xyz_reordered = xyz[:, index_order, :, :].clone()
        
        diffs_y = torch.abs(xyz_reordered[:, 1:, :, :] - xyz_reordered[:, :-1, :, :])
        diffs_y[:, :, :, 0] *= self.W * factor
        diffs_y[:, :, :, 1] *= self.H * factor
        sigma_ = torch.norm(diffs_y, dim=-1)
        sigma_first = sigma_[:, :1, :].clone() 
        sigma_y = torch.cat([sigma_first, sigma_], dim=1) / (3.0 / torch.sqrt(torch.tensor(factor, dtype=torch.float32))) # 归一化

        sigma_y[:, :2, :].clamp_(max=1.0, min=0.75)

        threshold = 0.1
        ratio = 3.0

        sx = sigma_x.clone()
        sy = sigma_y.clone()
        # mask where either value is extremely small
        mask = (sy < threshold)
        # only apply ratio clamp on positions starting from index 2
        mx = mask[:, 2:, :]
        my = mask[:, 2:, :]
        # sigma_x clamp only where mask = True
        sigma_x[:, 2:, :] = torch.where(
            mx, 
            torch.min(sx[:, 2:, :], sy[:, 2:, :] * ratio),
            sx[:, 2:, :]
        )
        # sigma_y clamp only where mask = True
        sigma_y[:, 2:, :] = torch.where(
            my,
            torch.min(sy[:, 2:, :], sx[:, 2:, :] * ratio),
            sy[:, 2:, :]
        )
        scaling = torch.cat([sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1)], dim=-1).contiguous()
        return scaling.view(-1, 2).detach()
    
    def get_scaling_open(self):
        xyz = self.xyz.view(self._control_points.shape[0], self.total_num_sample, 2).detach()
        diffs = torch.abs(xyz[:, 1:, :] - xyz[:, :-1, :])
        diffs[:, :, 0] *= self.W
        diffs[:, :, 1] *= self.H
        
        sigma_ratio = 2
        sigma = torch.norm(diffs, dim=2)
        sigma_last  = sigma[:, -1:].clone()  # shape: [N, 1] 
        sigma_x = (torch.cat([sigma, sigma_last], dim=1)) / sigma_ratio + 0.5
        sigma_y = torch.abs(self._scaling.repeat_interleave(self.total_num_sample, dim=1) + 0.5)
        scaling = torch.cat([sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1)], dim=-1)
        return scaling.view(-1, 2)
    

    
    def get_xyz_and_depth(self, factor=1, denser_sample=False):
        if self.mode == "line":
            xyz, normals, tangents = self.sample_bezier_curves(self._control_points, self.num_samples * factor)
        elif self.mode == "unclosed":
            xyz= self.sample_bezier_curves_unclose(self._control_points, self.total_num_sample * factor)
            return xyz.reshape(-1, 2), torch.zeros(1)
        else:
            if denser_sample:
                sampled_points, area_points = self.sample_bezier_area(self._control_points, resolution=self.curves_resolution * factor, factor=factor)
            else:
                sampled_points, area_points = self.sample_bezier_area(self._control_points, resolution=self.curves_resolution)

            return sampled_points, area_points
        return xyz
    
    
    
    @property
    def get_features(self):
        if self.mode == 'closed':
            return self.get_features_closed()
        else:
            return self.get_features_open()
    
    def get_area_weight(self, shape, device, alpha=4.0):
        """
        shape: (B, B_area, H, D)
        return: (B, B_area, H, D)
        """
        _, B_area, H, D = shape

        yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)      # [1, 1, H, 1]
        xx = torch.linspace(-1, 1, B_area, device=device).view(1, B_area, 1, 1)  # [1, B_area, 1, 1]

        dist = torch.abs(yy) + torch.abs(xx)  
        weight = 1.0 - torch.exp(-alpha * dist) 

        return weight.repeat(shape[0], 1, 1, D)  # [B, B_area, H, D]
    

    def get_features_closed(self):
        features_dc = self._features_dc.unsqueeze(1).unsqueeze(1).repeat(1, self.xyz.shape[1], self.xyz.shape[2], 1)
        features_dc_area = self._features_dc.unsqueeze(1).unsqueeze(1).repeat(1, self.xyz_area.shape[1], self.xyz.shape[2], 1)
        area_weight = self.get_area_weight(features_dc_area.shape, features_dc_area.device)
        features_dc_area = FeatureAreaModulator.apply(features_dc_area, area_weight)
        features = torch.cat([features_dc, features_dc_area], dim=1)
        _features_dc_expanded = torch.sigmoid(features)
        return _features_dc_expanded
    
    def get_features_open(self):
        _features_dc_expanded = torch.clamp(self._features_dc.unsqueeze(1).expand(-1, self.total_num_sample, -1),min=0.0, max=1.0)
        return _features_dc_expanded.view(-1, 3)
    
    @property
    def get_depth(self):
        if self.mode == "closed":
            return self._depth.unsqueeze(2).repeat(1, self.xyz.shape[1] + self.xyz_area.shape[1], self.xyz.shape[2])
        else:
            return self._depth.repeat_interleave(self.total_num_sample, dim=0)

    def compute_rotations(self, points):
        """
        Compute the rotation angle at each sampled point along each curve.

        Args:
            points: Tensor of shape [N, H, 2]
                - N: number of curves
                - H: number of sampled points per curve
                - Each point is a 2D coordinate.

        Returns:
            rotations: Tensor of shape [N, H]
                - Rotation angle (in radians) at each sampled point.
                - For the last sampled point, the angle is copied from the 
                second-to-last point.
        """
        if self.mode == 'closed':
            xyz = torch.cat([self.xyz, self.xyz_area], dim=1).detach()
        else:
            xyz = points.detach().view(self._control_points.shape[0], self.num_beziers, -1, 2)
        diffs = xyz[:, :, 2:, :] - xyz[:, :, :-2, :]
        diffs[:, :, :, 0] *= self.W
        diffs[:, :, :, 1] *= self.H
        
        theta = torch.atan2(diffs[..., 1], diffs[..., 0])  # shape: [N, H-1]
        theta_first = theta[..., :1].clone()  # shape: [N, 1] 
        theta_last = theta[..., -1:].clone()
        rotations = torch.cat([theta_first, theta, theta_last], dim=-1)
        return -rotations

    # 
    # 

    @property
    def get_opacity(self):
        if self.mode == "closed":
            if self.opacity_mode == 1:
                N = self._opacity.shape[0]  
                L = self.xyz_area.shape[1] 
                M = self.xyz.shape[2] 

                opacities_first = self._opacity[:, :1]  # (N, 1)
                opacities_middle = self._opacity[:, 1:2]  # (N, 1)
                opacities_last = self._opacity[:, 2:]  # (N, 1)

                weights_first = torch.linspace(0, 1, steps=L // 2, device=self._opacity.device).view(1, -1)  # (1, L//2)
                weights_second = torch.linspace(0, 1, steps=L - L // 2, device=self._opacity.device).view(1, -1)  # (1, L-L//2)

                opacities_area_first_half = (1 - weights_first) * opacities_first + weights_first * opacities_middle  # (N, L//2)

                opacities_area_second_half = (1 - weights_second) * opacities_middle + weights_second * opacities_last  # (N, L - L//2)

                opacities_area = torch.cat([opacities_area_first_half, opacities_area_second_half], dim=1)

                opacities_area = opacities_area.unsqueeze(-1).repeat(1, 1, M)
                opacity = torch.cat([
                    opacities_first.unsqueeze(1).repeat(1, 1, M),
                    opacities_last.unsqueeze(1).repeat(1, 1, M),
                    opacities_area
                ], dim=1)  # (N, L+2, M)
                return self.opacity_activation(opacity.contiguous().view(-1, 1))
            else:
                opacities =  self._opacity.unsqueeze(1).repeat(1, self.xyz.shape[1] + self.xyz_area.shape[1], self.xyz.shape[2])
                return  self.opacity_activation(opacities.contiguous().view(-1, 1))
        else:
            if self.opacity_mode == 1:
                N, cols = self._opacity.shape
                base_rep = self.total_num_sample // 3
                remainder = self.num_samples % 3
                parts = []
                for i in range(3):
                    rep = base_rep + (remainder if i == 3 else 0)
                    part = self._opacity[:, i].unsqueeze(1).repeat(1, rep)
                    parts.append(part)
                    out = torch.cat(parts, dim=1)
                return self.opacity_activation(out.contiguous().view(-1, 1))
            else:
                opacities =  self._opacity.repeat(1, self.total_num_sample)
                return  self.opacity_activation(opacities.contiguous().view(-1, 1))



    
    def sample_bezier_curves_uniform(self, bezier_curves: torch.Tensor, num_samples: int):
        """
        Sample Bézier curves and return sampled points, normals, and tangents.

        Args:
            bezier_curves: Tensor (num_curves, num_control_points, 2)
            num_samples: Number of points to sample per curve

        Returns:
            sampled_points: (num_curves, num_samples, 2)
            normals: (num_curves, num_samples, 2)
            tangents: (num_curves, num_samples, 2)
        """
        num_curves, num_control_points, dim = bezier_curves.shape
        if dim != 2:
            raise ValueError("Control points must be 2D coordinates.")

        device = bezier_curves.device
        n = num_control_points - 1  # Bézier degree

        key = (n, num_samples, str(device))
        cache = self._bernstein_cache[key]

        bernstein = cache['bernstein'][None, :, :]  # (1, num_samples, n+1)
        # Sampled points: (num_curves, num_samples, 2)
        sampled_points = torch.sum(bernstein[..., None] * bezier_curves[:, None, :, :], dim=2)
        return sampled_points

    def compute_aabb(self, points_tensor):
        """
        Args:
            points_tensor (torch.Tensor): (N, H, 2)。

        Returns:
            torch.Tensor: AABB,  (N, 4) (x1, y1, x2, y2)。
        """
        if points_tensor.ndim != 3 or points_tensor.size(-1) != 2:
            raise ValueError("points_tensor must have shape (N, H, 2).")

        min_coords, _ = points_tensor.min(dim=1)  # Shape: (N, 2)
        max_coords, _ = points_tensor.max(dim=1)  # Shape: (N, 2)

        aabb = torch.cat([min_coords, max_coords], dim=1)  # Shape: (N, 4)
        return aabb



    # 
    # 

    
    


        
        


        





  

    def sample_bezier_curves_unclose(self, control_points, num_samples, fine_samples=1000):
        """
        Sample boundary points and normals from a set of closed Bézier curves.

        Parameters:
        - control_points: Tensor of shape (num_curves, 12, 2)
        - num_samples: Total number of samples per curve
        - fine_samples: Unused for now

        Returns:
        - sampled_points: (num_curves, 4 * num_samples, 2)
        """
        num_curves, total_control_points, _ = control_points.shape
        assert total_control_points == self.num_beziers * self.bezier_degree + 1, (
            f"Expected {self.num_beziers * self.bezier_degree + 1} control points, got {total_control_points}"
        )
        device = control_points.device
        samples_per_segment = int(num_samples / self.num_beziers)

        # Vectorized generation of control point indices for each Bézier segment

        base = torch.arange(self.num_beziers, device=device).unsqueeze(1) * self.bezier_degree  # (num_beziers, 1)
        offsets = torch.arange(self.bezier_degree + 1, device=device).unsqueeze(0) 
        indices = (base + offsets) % total_control_points

        # Expand for all curves
        indices = indices.unsqueeze(0).expand(num_curves, -1, -1)  # (num_curves, num_beziers, 4)

        # Gather control points for each segment
        control_points_exp = control_points.unsqueeze(1).expand(-1, self.num_beziers, -1, -1)  # (num_curves, num_beziers, total_cp, 2)
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, 2)  # (num_curves, num_beziers, 4, 2)

        segment_control_points = torch.gather(control_points_exp, 2, indices_exp)

        # Merge all segments into a flat batch
        merged_control_points = segment_control_points.reshape(-1, self.bezier_degree + 1, 2)
        sampled_points = self.sample_bezier_curves_uniform(merged_control_points,samples_per_segment)
        # Sample each Bézier segment

        # Reshape back to (num_curves, num_samples, 2)
        return sampled_points


    def sample_bezier_curves(self, bezier_curves, num_samples, fine_samples=1000):
        """
        Generate uniformly spaced sample points and normals along Bézier curves
        using arc-length parameterization (revised version).

        Args:
            bezier_curves (torch.Tensor):
                Control points of the Bézier curves, shaped
                (num_curves, num_control_points, 2).

            num_samples (int):
                Number of target sample points to generate along each curve.

            fine_samples (int):
                Number of fine-grained samples used to approximate arc length.

        Returns:
            sampled_points (torch.Tensor):
                Uniformly distributed sample points along each curve,
                shaped (num_curves, num_samples, 2).

            normals (torch.Tensor):
                Corresponding normal vectors at each sampled point,
                shaped (num_curves, num_samples, 2).
        """
        num_curves, num_control_points, dim = bezier_curves.shape
        if dim != 2:
            raise ValueError("控制点必须是 2D 坐标。")

        device = bezier_curves.device
        n = num_control_points - 1

        t_values_fine = torch.linspace(0, 1, fine_samples, device=device)
        comb = torch.tensor([math.comb(n, i) for i in range(n + 1)], dtype=torch.float32, device=device)
        t_powers = t_values_fine[:, None] ** torch.arange(n + 1, dtype=torch.float32, device=device)  # t^i
        one_minus_t_powers = (1 - t_values_fine[:, None]) ** torch.arange(n, -1, -1, dtype=torch.float32, device=device)  # (1-t)^(n-i)
        bernstein = comb * one_minus_t_powers * t_powers  # Bernstein (fine_samples, n+1)
        bernstein = bernstein.unsqueeze(0)  # (1, fine_samples, n+1)
        fine_points = torch.sum(bernstein[..., None] * bezier_curves[:, None, :, :], dim=2)  # (num_curves, fine_samples, 2)

        # Step 2
        deltas = torch.norm(fine_points[:, 1:, :] - fine_points[:, :-1, :], dim=-1)
        arc_lengths = torch.cat([torch.zeros(num_curves, 1, device=device), deltas.cumsum(dim=-1)], dim=-1)
        total_lengths = arc_lengths[:, -1:]
        normalized_lengths = arc_lengths / total_lengths

        # Step 3
        target_lengths = torch.linspace(0, 1, num_samples, device=device)
        target_lengths = target_lengths.unsqueeze(0).expand(normalized_lengths.size(0), -1)
        indices = torch.searchsorted(normalized_lengths, target_lengths) 

        indices = torch.clamp(indices, 1, fine_samples - 1)
        low_indices = indices - 1
        high_indices = indices
        low_lengths = torch.gather(normalized_lengths, 1, low_indices)  # (num_curves, num_samples)
        high_lengths = torch.gather(normalized_lengths, 1, high_indices)  # (num_curves, num_samples)

        low_t = t_values_fine[low_indices]  # (num_curves, num_samples)
        high_t = t_values_fine[high_indices]  # (num_curves, num_samples)

        high_low_diff = high_lengths - low_lengths + 1e-8

        t_values_uniform = low_t + (target_lengths - low_lengths) / high_low_diff * (high_t - low_t)

        t_powers_uniform = t_values_uniform[:, :, None] ** torch.arange(n + 1, dtype=torch.float32, device=device)  # t^i
        one_minus_t_powers_uniform = (1 - t_values_uniform[:, :, None]) ** torch.arange(n, -1, -1, dtype=torch.float32, device=device)  # (1-t)^(n-i)
        bernstein_uniform = comb * one_minus_t_powers_uniform * t_powers_uniform  # (num_curves, num_samples, n+1)
        sampled_points = torch.sum(bernstein_uniform[..., None] * bezier_curves[:, None, :, :], dim=2)  # (num_curves, num_samples, 2)

        bezier_derivative = n * (bezier_curves[:, 1:, :] - bezier_curves[:, :-1, :])  # Derivative control points
        comb_derivative = torch.tensor([math.comb(n - 1, i) for i in range(n)], dtype=torch.float32, device=device)
        bernstein_derivative = comb_derivative * one_minus_t_powers_uniform[:, :, :-1] * t_powers_uniform[:, :, 1:]  # (num_curves, num_samples, n)
        tangents = torch.sum(bernstein_derivative[..., None] * bezier_derivative[:, None, :, :], dim=2)  # Tangents
        normals = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        return sampled_points, normals, tangents

    # 


    # 

    def split_bezier_segments(self, ctrl_pts):
        degree = self.bezier_degree + 1
        N, total_pts, _ = ctrl_pts.shape
        assert (total_pts - 1) % degree == 0, "Control points must follow M*D + 1 pattern"
        M = (total_pts - 1) // degree
        segments = [ctrl_pts[:, i*degree:i*degree + degree + 1, :].unsqueeze(1) for i in range(M)]  # list of (N, 1, D+1, 2)
        segments = torch.cat(segments, dim=1)  # shape: (N, M, D+1, 2)
        segments = segments.contiguous().view(-1, degree + 1, 2)  # shape: (N * M, D+1, 2)
        return segments

    def sample_bezier_area(self, control_points, resolution=20, factor=1):
        """
        Samples points along valid line segments formed by Bézier curve intersections with horizontal scanlines.

        Args:
            control_points (torch.Tensor): Shape (num_curves, num_control_points, 2), Bézier control points.
            sample_points (torch.Tensor): Shape (num_curves, num_points, 2), Bézier curve sample points.
            resolution (int): Number of horizontal scanlines.
            total_samples_per_row (int): Total number of points to sample per scanline.

        Returns:
            sampled_positions (torch.Tensor): (num_curves, resolution * total_samples_per_row, 2)
                - Sampled points along the valid segments.
        """
        N, total_pts, _ = control_points.shape
        num_samples = self.num_samples * factor
        assert (total_pts - 2) % 2 == 0, "Control point count must be 2M+2 for degree M Bézier pairs"
        M = (total_pts - 2) // 2  # degree of Bézier
        bezier1 = control_points[:, :M+2, :]
        bezier2 = torch.cat([control_points[:, M+1:, :], control_points[:, 0:1, :]], dim=1).flip(dims=[1])

        bezier1_segments = self.split_bezier_segments(bezier1)
        bezier2_segments = self.split_bezier_segments(bezier2)
        boundary_beziers = torch.cat([bezier1_segments, bezier2_segments], dim=0)  # (2*N*M, degree+2, 2)
        boundary = self.sample_bezier_curves_uniform(boundary_beziers, num_samples)
        
        bezier1_samples, bezier2_samples = boundary.chunk(2, dim=0)
        sampled_boundary = torch.stack([bezier1_samples.reshape(N, int(self.num_beziers * num_samples / 2), 2), \
            bezier2_samples.reshape(N, int(self.num_beziers * num_samples / 2), 2)], dim=1)


        bezier1 = bezier1.unsqueeze(1)
        bezier2 = bezier2.unsqueeze(1) 
        t_vals = torch.linspace(-2, 2, resolution, device=control_points.device)
        t_vals = dist.Normal(0, 0.85).cdf(t_vals).view(1, resolution, 1, 1)  # (1, R, 1, 1)
        interp_cp = (1 - t_vals) * bezier1 + t_vals * bezier2  # (N, R, M+2, 2)
        interp_cp_flat = interp_cp.view(-1, M + 2, 2)  # (N * R, M+2, 2)

        interp_segments = self.split_bezier_segments(interp_cp_flat)
        interp_samples = self.sample_bezier_curves_uniform(interp_segments, num_samples)  # (N * R, num_samples, 2)
        return sampled_boundary, interp_samples.view(self._control_points.shape[0], resolution, -1, 2)

    # 
    # 

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                stored_state["exp_avg_diff"] = stored_state["exp_avg_diff"][mask]
                stored_state["neg_pre_grad"] = stored_state["neg_pre_grad"][mask]

                del self.optimizer.state[group['params'][0]]
                if group["name"] == "xyz":
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                else:
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
                opacity_mask = mask.detach().cpu()
                group["params"][0] = nn.Parameter(group["params"][0][opacity_mask].requires_grad_(False))
                optimizable_tensors[group["name"]] = group["params"][0].to(mask.device)
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_diff"] = torch.cat((stored_state["exp_avg_diff"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["neg_pre_grad"] = torch.cat((stored_state["neg_pre_grad"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor.to(group["params"][0].device)), dim=0).requires_grad_(False))
                optimizable_tensors[group["name"]] = group["params"][0].to(extension_tensor.device)
        return optimizable_tensors

    # 

    def prune_beizer_curves(self, mask):
        valid_beizer_mask = mask
        optimizable_tensors = self._prune_optimizer(valid_beizer_mask)
        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        self._scaling = optimizable_tensors["scaling"]
        self._opacity = optimizable_tensors["opacity"]
        self._depth = self._depth[mask]
        self.xyz=self.xyz.view(-1,self.total_num_sample,2)[mask].view((-1,self.total_num_sample,2))
        self.num_curves = self._control_points.shape[0]

    def densification_postfix(self, new_control_points, new_features, new_cholesky, new_depth, new_opacities, new_scaling):
        d = {"control_points": new_control_points,
        "features_dc": new_features,
        "cholesky": new_cholesky,
        "opacity": new_opacities,
        "scaling": new_scaling,
        "depth" : new_depth}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._depth = torch.cat((self._depth, new_depth.to(self._depth.device)), dim=0)
        self.num_curves = self._control_points.shape[0]



    def densify(self, num, pos_init_method, gt_image, radii=0.02):
        centers = torch.tensor([pos_init_method() for _ in range(num)], dtype=torch.float32).to(self._control_points.device)
        centers_rounded = centers.round().long()
        centers_rounded[:, 0] = torch.clamp(centers_rounded[:, 0], 0, self.H - 1)
        centers_rounded[:, 1] = torch.clamp(centers_rounded[:, 1], 0, self.W - 1)
        gt_values = gt_image[:, :, centers_rounded[:, 0], centers_rounded[:, 1]].squeeze(0).T
        new_control_points = self._initialize_control_points_with_center(centers.unsqueeze(1), radii).to(self._control_points.device)
        new_cholesky = nn.Parameter(torch.rand(centers.shape[0], 3)).to(self._control_points.device)
        logits = torch.logit(gt_values.clamp(1e-6, 1 - 1e-6))  # 防止除以0或log(0)
        new_features = nn.Parameter(logits.to(self._control_points.device))
        new_depth = nn.Parameter(torch.zeros(centers.shape[0], 1)).to(self._control_points.device)
        new_scaling = nn.Parameter(torch.rand(centers.shape[0], 1)).to(self._control_points.device)
        new_opacity = nn.Parameter(torch.ones(centers.shape[0], self._opacity.shape[1])).to(self._control_points.device)
        self.densification_postfix(new_control_points, new_features, new_cholesky, new_depth, new_opacity, new_scaling)





    def forward(self, factor=1, denser_sample=False):
        final_h = self.H * factor
        final_w = self.W * factor
        self.xyz, self.xyz_area = self.get_xyz_and_depth(factor, denser_sample)
        if self.mode == 'closed':
            xyz_input = torch.cat([self.xyz, self.xyz_area], dim=1).contiguous().view(-1, 2)
        else:
            xyz_input = self.xyz
        
        with torch.no_grad():
            rotation_input = self.compute_rotations(
                    self.xyz.view(self._control_points.shape[0], -1, 2)).view(-1, 1).detach()    

        if self.mode == 'closed':
            with torch.no_grad():
                scaling = self.get_scaling(factor)
        else:
            scaling = self.get_scaling(factor)

        opacity = self.get_opacity
        features_dc = self.get_features.contiguous()
        self.tile_bounds = (
            (final_w + self.BLOCK_W - 1) // self.BLOCK_W,
            (final_h + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.xys, depths_, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
                xyz_input, scaling, rotation_input, final_h, final_w, self.tile_bounds)

        depth = self.get_depth.detach()
        if self.mode == "closed":
            depth = self.get_depth.view(-1, 1)
        else:
            if self.iter < 10000 and self.iter % 20 == 0:
                with torch.no_grad():
                    xys = self.xys.clone().detach().view(self._control_points.shape[0], -1, 2)
                    diffs = torch.norm(xys[:, 1:, :] - xys[:, :-1, :], dim=-1).sum(-1, keepdim=True) * torch.abs(self._scaling.detach())
                    self._depth.copy_(diffs).contiguous()
            depth = self.get_depth.detach()

        if factor > 1:
            print("See the range: ", self.xys.max(), self.xys.min(), self.xyz.max(), self.xyz.min())
            
        out_img = rasterize_gaussians(
                self.xys,
                depth,
                self.radii,
                conics,
                num_tiles_hit,
                features_dc.view(-1, 3),
                opacity.view(-1, 1),
                final_h, final_w,
                self.BLOCK_H, self.BLOCK_W,
                background=self.background,
                return_alpha=False)

        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, final_h, final_w, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    



    



    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss_reg = xing_loss(self._control_points, scale=self.xing_weight)
        loss += loss_reg
        loss += 1e-2 * torch.abs(torch.sigmoid(self._opacity) - 1.0).mean()

        # # print("boundary: ", self.xyz.shape)
        loss += boundary_loss_on_joints(self._control_points, self.bezier_degree + 1)
        loss += curvature_loss(self.xyz, self.num_samples)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.iter += 1
        return loss, psnr, image
    
    def train_iter_opencurves(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.iter += 1
        return loss, psnr, image
