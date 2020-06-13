import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d(in_channel, out_channel, kernel, use_activation=False):

    if use_activation:
        module = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
            )
    else:
        module = nn.Conv1d(in_channel, out_channel, kernel)

    return module

def conv2d(in_channel, out_channel, kernel, use_activation=False):

    if use_activation:
        module = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
            )
    else:
        module = nn.Conv2d(in_channel, out_channel, kernel)

    return module


class DGCNet(nn.Module):

    # default hyper params are fitted to Citysacpes
    def __init__(self, use_spatial_gcn=True, use_feature_gcn=True, d=8):

        super(DGCNet, self).__init__()

        # Coordinate(Spatial) Space GCN
        self.use_spatial_gcn = use_spatial_gcn

        if use_spatial_gcn:
            self.d = d
            self.spatial_projection = nn.AvgPool2d(kernel_size=d, stride=d) # Parameter Free
            self.delta = conv2d(2048, 1024, 1)
            self.psi = conv2d(2048, 1024, 1)
            self.v = conv2d(2048, 1024, 1)
            self.Ws = conv2d(1024, 1024, 1)
            self.xi = conv2d(1024, 2048, 1)

        # Feature Space GCN
        self.use_feature_gcn = use_feature_gcn

        if use_feature_gcn:
            self.pi = conv2d(2048, 512, 1, True)
            self.theta = conv2d(2048, 1024, 1, True)
            self.adj_conv = conv1d(512, 512, 1, True)
            self.Wf = conv1d(1024, 1024, 1, True)
            self.phi = conv2d(1024, 2048, 1, True) # D1 -> D


    def forward(self, x):

        N, C, H, W = x.shape
        if self.use_spatial_gcn:
            Vs = self.spatial_projection(x) # (N, C, H/d, W/d)
            m_delta = torch.flatten(self.delta(Vs), start_dim=2) # (N, C/2, HW/d^2)
            m_psi = torch.flatten(self.psi(Vs), start_dim=2) # (N, C/2. HW/d^2)
            m_v = torch.flatten(self.v(Vs), start_dim=2) # (N, C/2, HW/d^2)

            m_0 = torch.bmm(m_psi, torch.transpose(m_v, 1, 2)) # (N, C/2, C/2)
            m_1 = torch.bmm(torch.transpose(m_delta, 1, 2), m_0) # (N, HW/d^2, C/2)
            m_1 = torch.reshape(torch.transpose(m_1, 1, 2), (N,1024,int(H/self.d),int(W/self.d)))
            # (N, HW/d^2, C/2) => (N, C/2, H/d, W/d)
            Ms = self.Ws(m_1) # (N, C/2, H/d, W/d)
            x = x + self.xi(nn.Upsample(size=(H, W))(Ms)) # (N, C/2, H/d, W/d) -> (N, C, H, W)

        if self.use_feature_gcn:
            Hf = torch.flatten(self.pi(x), start_dim=2) # (N, C/4, H, W) -> (N, C/4, HW)
            Vf = torch.bmm(Hf, torch.transpose(torch.flatten(self.theta(x), start_dim=2), 1, 2))
            # (N, C/4, HW) bmm (N, HW, C/2) -> (N, C/4, C/2)
            Af = Vf - self.adj_conv(Vf) # (N, C/4, C/2)
            Mf = self.Wf(torch.transpose(Af, 1, 2)) # (N, C/2, C/4)
            Mf = torch.transpose(Mf, 1, 2) # (N, C/4, C/2)
            Xf = torch.transpose(torch.bmm(torch.transpose(Hf, 1, 2), Mf), 1, 2) # (N, C/2, HW)
            x = x + self.phi(torch.reshape(Xf, (N, 1024, H, W))) # (N, C, H, W)

        return x



if __name__ == "__main__":

    dgcn = DGCNet()
    print(dgcn)









