# code copied from https://ridiqulous.com/pytorch-non-local-means/

import torch
import torch.nn as nn

EPSILON = 1E-6


class NonLocalMeansFast(nn.Module):
    def __init__(self, search_window_size=11, patch_size=5, trainable=False):
        super().__init__()
        self.gen_window_stack = ShiftStack(window_size=search_window_size)
        self.box_sum = BoxFilter(window_size=patch_size, reduction='sum')
        
    def forward(self, rgb, sigma):
        h = sigma * 2
        y = rgb_to_luminance(rgb)  # (N, 1, H, W)

        rgb_window_stack = self.gen_window_stack(rgb)  # (N, 3, H, W, w*y)
        y_window_stack = self.gen_window_stack(y)  # (N, 1, H, W, w*y)

        distances = torch.sqrt(self.box_sum((y.unsqueeze(-1) - y_window_stack) ** 2))  # (N, 1, H, W, w*y)
        weights = torch.exp(-distances / (torch.relu(h.unsqueeze(-1)) + EPSILON))  # (N, 1, H, W, w*y)

        denoised_rgb = (weights * rgb_window_stack).sum(dim=-1) / weights.sum(dim=-1)  # (N, 3, H, W)

        return torch.clamp(denoised_rgb, 0, 1)  # (N, 3, H, W)


class NonLocalMeans(nn.Module):
    def __init__(self, sigma, search_window_size=11, patch_size=5, trainable=False):
        super().__init__()
        self.h = nn.Parameter(torch.tensor([float(sigma*2)]), requires_grad=trainable)

        self.box_sum = BoxFilter(window_size=patch_size, reduction='sum')
        self.r = search_window_size // 2

    def forward(self, rgb):
        batch_size, _, height, width = rgb.shape
        weights = torch.zeros((batch_size, 1, height, width)).float().to(rgb.device)  # (N, 1, H, W)
        denoised_rgb = torch.zeros_like(rgb)  # (N, 3, H, W)

        y = rgb_to_luminance(rgb)  # (N, 1, H, W)

        for x_shift in range(-self.r, self.r + 1):
            for y_shift in range(-self.r, self.r + 1):
                shifted_rgb = torch.roll(rgb, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 3, H, W)
                shifted_y = torch.roll(y, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 1, H, W)

                distance = torch.sqrt(self.box_sum((y - shifted_y) ** 2))  # (N, 1, H, W)
                weight = torch.exp(-distance / (torch.relu(self.h) + EPSILON))  # (N, 1, H, W)

                denoised_rgb += shifted_rgb * weight  # (N, 3, H, W)
                weights += weight  # (N, 1, H, W)

        return torch.clamp(denoised_rgb / weights, 0, 1)  # (N, 3, H, W)


class BoxFilter(nn.Module):
    def __init__(self, window_size, reduction='mean'):
        # :param window_size: Int or Tuple(Int, Int) in (win_width, win_height) order
        # :param reduction: 'mean' | 'sum'
        super().__init__()
        wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
        self.rx, self.ry = wx // 2, wy // 2
        self.area = wx * wy
        self.reduction = reduction

    def forward(self, tensor):
        # :param tensor: torch.Tensor(N, C, H, W, ...)
        # :return: torch.Tensor(N, C, H, W, ...)
        local_sum = torch.zeros_like(tensor)
        for x_shift in range(-self.rx, self.rx + 1):
            for y_shift in range(-self.ry, self.ry + 1):
                local_sum += torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))

        return local_sum if self.reduction == 'sum' else local_sum / self.area


class ShiftStack(nn.Module):
    """
    Shift n-dim tensor in a local window and generate a stacked
    (n+1)-dim tensor with shape (*orig_shapes, w*y), where wx
    and wy are width and height of the window
    """

    def __init__(self, window_size):
        # :param window_size: Int or Tuple(Int, Int) in (win_width, win_height) order
        super().__init__()
        wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
        self.rx, self.ry = wx // 2, wy // 2

    def forward(self, tensor):
        # :param tensor: torch.Tensor(N, C, H, W, ...)
        # :return: torch.Tensor(N, C, H, W, ..., w*y)
        shifted_tensors = []
        for x_shift in range(-self.rx, self.rx + 1):
            for y_shift in range(-self.ry, self.ry + 1):
                shifted_tensors.append(
                    torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))
                )

        return torch.stack(shifted_tensors, dim=-1)


def rgb_to_luminance(rgb_tensor):
    # :param rgb_tensor: torch.Tensor(N, 3, H, W, ...) in [0, 1] range
    # :return: torch.Tensor(N, 1, H, W, ...) in [0, 1] range
    # assert rgb_tensor.min() >= 0.0 and rgb_tensor.max() <= 1.0
    return 0.299 * rgb_tensor[:, :1, ...] + 0.587 * rgb_tensor[:, 1:2, ...] + 0.114 * rgb_tensor[:, 2:, ...]
