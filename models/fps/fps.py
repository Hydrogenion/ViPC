import torch
from torch.autograd import Function

import point_gpu


class FarthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, cnt: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param cnt: int, number of features in the sampled set
        :return:
             output: (B, npoint) index of sampling points
        """
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()

        B, N, _ = xyz.size()
        output: torch.Tensor = torch.cuda.IntTensor(B, cnt)
        # 用来临时记录当前已选择点集到所有点的距离
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        point_gpu.farthest_point_sampling_wrapper(B, N, cnt, xyz, temp, output)
        return output.long()

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = FarthestPointSampling.apply
