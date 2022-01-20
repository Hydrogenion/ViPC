import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math 
import os
import cv2 
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

def gen_grid_up(up_rtatio):
    sqrted = int(math.sqrt(up_rtatio))+1
    for i in range(1,sqrted+1).__reversed__():
        if (up_rtatio%i) == 0:
            num_x = i
            num_y = up_rtatio//i
            break
    grid_x = torch.linspace(-0.2,0.2, num_x)
    grid_y = torch.linspace(-0.2,0.2, num_y)

    x, y = torch.meshgrid(grid_x,grid_y)
    grid = torch.reshape(torch.stack([x,y], axis=-1), [-1,2])
    return grid.to('cuda')


class ProjectionLayer(nn.Module):
    def __init__(self):
        super(ProjectionLayer, self).__init__()

    def forward(self, img_features, input):

        self.img_feats = img_features 

        h = 100 * (-input[:, 1]) + 111.5
        w = 100 * (input[:, 0]) + 111.5

        h = torch.clamp(h, min = 0, max = 223)
        w = torch.clamp(w, min = 0, max = 223)

        img_sizes = [56, 28, 14, 7]
        out_dims = [64, 128, 256, 512]
        feats = []

        for i in range(4):
            out = self.project(i, h, w, img_sizes[i], out_dims[i])
            feats.append(out)
            
        output = torch.cat(feats, 1)
        
        return output

    def project(self, index, h, w, img_size, out_dim):

        img_feat = self.img_feats[index]
        x = h / (224. / img_size)
        y = w / (224. / img_size)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        x2 = torch.clamp(x2, max = img_size - 1)
        y2 = torch.clamp(y2, max = img_size - 1)

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        x, y = x.long(), y.long()

        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0 ,1))

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22

        return output


class PartRefinement(nn.Module):
    def __init__(self,step_ratio = 2, up_ratio = 2):
        super(PartRefinement,self).__init__()
        self.step_ratio = step_ratio
        self.up_ratio = up_ratio
        self.projection = ProjectionLayer()


        self.mlp1 = nn.Linear(1024,128)
        self.conv1d_1 = nn.Conv1d(3013,1024,1)
        self.conv1d_2 = nn.Conv1d(1024,128,1)
        self.conv1d_3 = nn.Conv1d(128,64,1)

        self.conv2d_1 = nn.Conv2d(64,64,[1,self.up_ratio])
        self.conv2d_2 = nn.Conv2d(64,128,[1,1])
        self.conv2d_3 = nn.Conv2d(64,32,[1,1])

        self.conv1d_4 = nn.Conv1d(64,512,1)
        self.conv1d_5 = nn.Conv1d(512,512,1)
        self.conv1d_6 = nn.Conv1d(512,6,1)

        self.fc = nn.Linear(1*1024,1*1024)
        self.feat = None


    def forward(self,x, rate):
        # x = [concat, partial point feat]
        # concat and downsample point clouds should be 1024
        level0 = x[0]
        code = x[1]
        global_code = x[2]
        img_fea = x[3]
        batch_size = level0.shape[0]
        input_point_nums = level0.shape[2]
        for i,key in enumerate(img_fea):
            img_fea[i] = torch.squeeze(key)
        level0_squeeze = torch.squeeze(level0)
        img_proj_feat = self.projection(img_fea,level0_squeeze.permute(1,0))

        num_fine = rate*input_point_nums
        grid = gen_grid_up(rate**(0+1))
        grid = grid.unsqueeze(0).permute(0,2,1)
        grid_feat = grid.repeat(level0.shape[0],1,int(input_point_nums/2))

        point_out = level0.unsqueeze(2).repeat(1,1,rate,1)
        point_out =  torch.reshape(point_out,[-1,3,num_fine])

        point_feat = level0.unsqueeze(2).repeat(1,1,1,1)
        point_feat = torch.reshape(point_feat,[-1,3,int(num_fine/2)])

        global_feat = code.unsqueeze(2).repeat(1,1,int(num_fine/2))
        generate_feat = global_code.unsqueeze(2).repeat(1,1,int(num_fine/2))

        img_proj_feat = img_proj_feat.permute(1,0).unsqueeze(0)
        img_proj_feat = self.fc(img_proj_feat)
        
        feat = torch.cat([grid_feat,point_feat,global_feat,generate_feat,img_proj_feat],axis=1)
        self.feat = feat

        # Dynamic Offset Predictor
        feat1 = self.conv1d_1(feat)
        feat1 = self.conv1d_2(feat1)
        feat1 = self.conv1d_3(feat1)
        feat1 = F.relu(feat1)

        feat2 = feat1.unsqueeze(-1).repeat(1,1,1,2)
        feat2 = self.conv2d_1(feat2)
        feat2 = self.conv2d_2(feat2)

        feat2 = feat2.view(feat2.shape[0], self.up_ratio, 64, -1).permute(0, 2, 1, 3)
        feat2 = self.conv2d_3(feat2)
        feat2 = feat2.view(feat2.shape[0], 64, -1)

        feat = feat1 + feat2

        feat = self.conv1d_4(feat)
        feat = self.conv1d_5(feat)
        feat = self.conv1d_6(feat)
        offset = feat.view(-1,3,2048)

        fine = offset + point_out

        return offset.permute(0,2,1)


if __name__ == "__main__":
    net = PartRefinement(step_ratio=2).to('cuda')
    img_feat = [torch.rand(1,64,56,56).to('cuda'),torch.rand(1,128,28,28).to('cuda'),torch.rand(1,256,14,14).to('cuda'),torch.rand(1,512,7,7).to('cuda')]
    output = net([torch.rand((1,3,1024)).to('cuda'),torch.rand((1,1024)).to('cuda'),torch.rand((1,1024)).to('cuda'),img_feat],2)
    print(output.shape)
