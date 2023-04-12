import numpy as np
import torch
import torch.nn as nn


class Augment(nn.Module):
    def __init__(self, base_denoiser):
        super().__init__()
        self.base_denoiser = base_denoiser
        self.iter = 0

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor):
        iter = self.iter

        x = self.augment(x, iter % 8)

        x = self.base_denoiser.denoise(x, sigma)

        if iter % 8 == 3 or iter % 8 == 5:
            x = self.augment(x, 8 - iter % 8)
        else:
            x = self.augment(x, iter % 8)

        self.iter += 1
        return x

    def reset(self):
        self.iter = 0

    @staticmethod
    def augment(img, mode=0):
        if mode == 0:
            return img
        elif mode == 1:
            return img.rot90(1, [2, 3]).flip([2])
        elif mode == 2:
            return img.flip([2])
        elif mode == 3:
            return img.rot90(3, [2, 3])
        elif mode == 4:
            return img.rot90(2, [2, 3]).flip([2])
        elif mode == 5:
            return img.rot90(1, [2, 3])
        elif mode == 6:
            return img.rot90(2, [2, 3])
        elif mode == 7:
            return img.rot90(3, [2, 3]).flip([2])


class DeepTVDenoiser:
    def __init__(self, deep_denoise, tv_denoising,
                 deep_hypara_list=[40., 20., 10., 5.], tv_hypara_list=[10, 0.01]):
        self.deep_hypara_list = deep_hypara_list
        self.tv_hypara_list = tv_hypara_list
        self.tv_denoising = tv_denoising
        self.deep_denoise = deep_denoise

    def denoise(self, x):
        import cvxpy as cp
        # x: 1,31,512,512
        deep_num = len(self.deep_hypara_list)
        tv_num = len(self.tv_hypara_list)
        deep_list = [self.deep_denoise(x, torch.tensor(level/255.).to(x.device)) for level in self.deep_hypara_list]
        deep_list = [tmp.squeeze().permute(1, 2, 0) for tmp in deep_list]

        tv_list = [self.tv_denoising(x.squeeze().permute(1, 2, 0), level, 5).clamp(0, 1) for level in self.tv_hypara_list]

        ffdnet_mat = np.stack(
            [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in deep_list],
            axis=0)
        tv_mat = np.stack(
            [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in tv_list],
            axis=0)
        w = cp.Variable(deep_num + tv_num)
        P = np.zeros((deep_num + tv_num, deep_num + tv_num))
        P[:deep_num, :deep_num] = ffdnet_mat @ ffdnet_mat.T
        P[:deep_num, deep_num:] = -ffdnet_mat @ tv_mat.T
        P[deep_num:, :deep_num] = -tv_mat @ ffdnet_mat.T
        P[deep_num:, deep_num:] = tv_mat @ tv_mat.T
        one_vector_ffdnet = np.ones((1, deep_num))
        one_vector_tv = np.ones((1, tv_num))
        objective = cp.quad_form(w, P)
        problem = cp.Problem(
            cp.Minimize(objective),
            [one_vector_ffdnet @ w[:deep_num] == 1,
                one_vector_tv @ w[deep_num:] == 1,
                w >= 0])
        problem.solve()
        w_value = w.value
        x_ffdnet, x_tv = 0, 0

        for idx in range(deep_num):
            x_ffdnet += w_value[idx] * deep_list[idx]
        for idx in range(tv_num):
            x_tv += w_value[idx + deep_num] * tv_list[idx]
        v = 0.5 * (x_ffdnet + x_tv)
        v = v.permute(2, 0, 1).unsqueeze(0)
        return v
