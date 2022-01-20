from models.modality_transfer import ModalityTransfer
from models.part_refinement import PartRefinement
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fps import farthest_point_sample
from models.cd_distance.chamfer_distance import ChamferDistance
import kaolin as kal 

class ViPC(nn.Module):
    def __init__(self,category, init_weights=False):
        super(ViPC, self).__init__()
        self.modality_transfer = ModalityTransfer()
        self.cd_distance = ChamferDistance()
        self.part_refinement = PartRefinement(2,2)
        self.category = category
        self.pointnet_encoder = kal.models.PointNet.PointNetFeatureExtractor(
        in_channels=3, 
        feat_size=1024,
        layer_dims=[64, 64, 64, 128],
        transposed_input = True)
        self.pointnet_encoder_2 = kal.models.PointNet.PointNetFeatureExtractor(
        in_channels=3, 
        feat_size=1024,
        layer_dims=[64, 64, 64, 128],
        transposed_input = True)

        if init_weights:
            self.init_psgnet()

    def init_psgnet(self):
        print('init parameter for Modality Transfer')
        ckpt_path = f'./checkpoints/{self.category}.pth'
        self.modality_transfer.load_state_dict({k.replace('ip2net.', ''):v for k, v in torch.load(ckpt_path).items()})
        print(f'load ckpt from {ckpt_path}')

    def forward(self,view,partial_pc):
        batch_size = view.size(0)
        reconstructed_pc,img_feature = self.modality_transfer(view)
        for parameter in self.modality_transfer.parameters():
            parameter.requires_grad = False 
        concat_pc = torch.cat([reconstructed_pc,partial_pc],dim = 1)

        coarse_pc_indices = farthest_point_sample(concat_pc, 1024).unsqueeze(-1).expand(batch_size,1024,3)
        coarse_pc = torch.gather(concat_pc,1,coarse_pc_indices)

        # Part Filter
        theta = self.cd_distance(coarse_pc[:,0:512,:],coarse_pc[:,512:,:])[0].mean().float()
        
        indices_lost = self.cd_distance(coarse_pc,partial_pc)[0]>theta
        indices_part = self.cd_distance(coarse_pc,partial_pc)[0]<=theta

        # protective mask
        indices_mask = torch.cat([indices_part, torch.zeros(1,1024).bool().to('cuda')],dim = 1)

        partial_point_feat = self.pointnet_encoder(partial_pc.permute(0,2,1))
        global_point_feat = self.pointnet_encoder_2(reconstructed_pc.permute(0,2,1))

        offset = self.part_refinement([coarse_pc.permute(0,2,1), partial_point_feat, global_point_feat, img_feature], 2)

        point_out = coarse_pc.repeat(1,2,1)
        offset[indices_mask] = torch.clamp(offset[indices_mask], -0.02, 0.02)

        fine_point_cloud = point_out + offset

        return fine_point_cloud, reconstructed_pc, coarse_pc

if __name__ == "__main__":
    vipc = ViPC('plane',False).to('cuda')
    output = vipc(torch.rand(1,3,224,224).to('cuda'),torch.rand(1,2048,3).to('cuda'))
    print(output[0].shape)
