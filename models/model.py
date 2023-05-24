from torch.functional import align_tensors
from .tensorBase import *
from .quaternion_utils import *
from utils import N_to_reso

import numpy as np
import math


class TensorDecomposition(torch.nn.Module):
    def __init__(self, grid_size, num_features, scale, device, reduce_sum=False):
        super(TensorDecomposition, self).__init__()
        self.grid_size = torch.tensor(grid_size)
        self.num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
        self.reduce_sum = reduce_sum

        X, Y, Z = grid_size
        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, num_features, Y, X), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, num_features, Z, Y), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, num_features, Z, X), device=device))

        self.line_z = torch.nn.Parameter(scale * torch.randn((1, num_features, Z, 1), device=device))
        self.line_x = torch.nn.Parameter(scale * torch.randn((1, num_features, X, 1), device=device))
        self.line_y = torch.nn.Parameter(scale * torch.randn((1, num_features, Y, 1), device=device))

    def forward(self, coords_plane, coords_line):
        feature_xy = F.grid_sample(self.plane_xy, coords_plane[0], mode='bilinear', align_corners=True)
        feature_yz = F.grid_sample(self.plane_yz, coords_plane[1], mode='bilinear', align_corners=True)
        feature_xz = F.grid_sample(self.plane_xz, coords_plane[2], mode='bilinear', align_corners=True)

        feature_x = F.grid_sample(self.line_x, coords_line[0], mode='bilinear', align_corners=True)
        feature_y = F.grid_sample(self.line_y, coords_line[1], mode='bilinear', align_corners=True)
        feature_z = F.grid_sample(self.line_z, coords_line[2], mode='bilinear', align_corners=True)

        out_x = feature_yz * feature_x
        out_y = feature_xz * feature_y
        out_z = feature_xy * feature_z

        _, C, N, _ = out_x.size()
        if self.reduce_sum:
            output = out_x.sum(dim=(0, 1, 3)) + out_y.sum(dim=(0, 1, 3)) + out_z.sum(dim=(0, 1, 3))
        else:
            output = [out_x.view(-1, N).T, out_y.view(-1, N).T, out_z.view(-1, N).T]

        return output

    def L1loss(self):
        loss = torch.abs(self.plane_xy).mean() + torch.abs(self.plane_yz).mean() + torch.abs(self.plane_xz).mean()
        loss += torch.abs(self.line_x).mean() + torch.abs(self.line_y).mean() + torch.abs(self.line_z).mean()
        loss = loss / 6

        return loss

    def TV_loss(self):
        loss = self.TV_loss_com(self.plane_xy)
        loss += self.TV_loss_com(self.plane_yz)
        loss += self.TV_loss_com(self.plane_xz)
        loss = loss / 6

        return loss

    def TV_loss_com(self, x):
        loss = (x[:, :, 1:] - x[:, :, :-1]).pow(2).mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
        return loss


    def shrink(self, bound):
        # bound [3, 2]
        x, y, z = bound[0], bound[1], bound[2]
        self.plane_xy = torch.nn.Parameter(self.plane_xy.data[:, :, y[0]:y[1], x[0]:x[1]])
        self.plane_yz = torch.nn.Parameter(self.plane_yz.data[:, :, z[0]:z[1], y[0]:y[1]])
        self.plane_xz = torch.nn.Parameter(self.plane_xz.data[:, :, z[0]:z[1], x[0]:x[1]])

        self.line_x = torch.nn.Parameter(self.line_x.data[:, :, x[0]:x[1]])
        self.line_y = torch.nn.Parameter(self.line_y.data[:, :, y[0]:y[1]])
        self.line_z = torch.nn.Parameter(self.line_z.data[:, :, z[0]:z[1]])

        self.grid_size = bound[:, 1] - bound[:, 0]


    def upsample(self, aabb):
        target_res = N_to_reso(self.num_voxels, aabb)


        self.grid_size = torch.tensor(target_res)

        self.plane_xy = torch.nn.Parameter(F.interpolate(self.plane_xy.data,
            size=(target_res[1], target_res[0]), mode='bilinear', align_corners=True))
        self.plane_yz = torch.nn.Parameter(F.interpolate(self.plane_yz.data,
            size=(target_res[2], target_res[1]), mode='bilinear', align_corners=True))
        self.plane_xz = torch.nn.Parameter(F.interpolate(self.plane_xz.data,
            size=(target_res[2], target_res[0]), mode='bilinear', align_corners=True))


        self.line_x = torch.nn.Parameter(F.interpolate(self.line_x.data,
            size=(target_res[0], 1), mode='bilinear', align_corners=True))
        self.line_y = torch.nn.Parameter(F.interpolate(self.line_y.data,
            size=(target_res[1], 1), mode='bilinear', align_corners=True))
        self.line_z = torch.nn.Parameter(F.interpolate(self.line_z.data,
            size=(target_res[2], 1), mode='bilinear', align_corners=True))

class MultiscaleTensorDecom(torch.nn.Module):
    def __init__(self, num_levels, num_features, base_resolution, max_resolution, device, reduce_sum=False, scale=0.1):
        super(MultiscaleTensorDecom, self).__init__()
        self.reduce_sum = reduce_sum

        tensors = []
        if num_levels == 1:
            factor = 1
        else:
            factor = math.exp( (math.log(max_resolution) - math.log(base_resolution)) / (num_levels-1) )

        for i in range(num_levels):
            level_resolution = int(base_resolution * factor**i)
            level_grid = (level_resolution, level_resolution, level_resolution)
            tensors.append(TensorDecomposition(level_grid, num_features, scale, device, reduce_sum=reduce_sum))

        self.tensors = torch.nn.ModuleList(tensors)

    def coords_split(self, pts, dim=2, z_vals=None):
        N, D = pts.size()
        pts = pts.view(1, N, 1, D)

        out_plane = []
        if dim == 2:
            out_plane.append(pts[..., [0, 1]])
            out_plane.append(pts[..., [1, 2]])
            out_plane.append(pts[..., [0, 2]])
        elif dim == 3:
            out_plane.append(pts[..., [0, 1, 2]][:, :, None])
            out_plane.append(pts[..., [1, 2, 0]][:, :, None])
            out_plane.append(pts[..., [0, 2, 1]][:, :, None])

        if z_vals is None:
            coord_x = pts.new_zeros(1, N, 1, 1)
        else:
            coord_x = z_vals.view(1, N, 1, 1)
        out_line = []
        out_line.append(torch.cat((coord_x, pts[..., [0]]), dim=-1))
        out_line.append(torch.cat((coord_x, pts[..., [1]]), dim=-1))
        out_line.append(torch.cat((coord_x, pts[..., [2]]), dim=-1))

        return out_plane, out_line

    def L1loss(self):
        loss = 0.
        for tensor in self.tensors:
            loss += tensor.L1loss()

        return loss / len(self.tensors)

    def shrink(self, aabb, new_aabb):
        aabb_size = aabb[1] - aabb[0]
        xyz_min, xyz_max = new_aabb

        for tensor in self.tensors:
            grid_size = tensor.grid_size
            units = aabb_size / (grid_size - 1)
            t_l, b_r = (xyz_min - aabb[0]) / units, (xyz_max - aabb[0]) / units

            t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long()
            b_r = torch.stack([b_r, grid_size]).amin(0)

            bound = torch.stack((t_l, b_r), dim=-1)
            tensor.shrink(bound)

    def upsample(self, aabb):
        for tensor in self.tensors:
            tensor.upsample(aabb)

    def forward(self, pts):
        coords_plane, coords_line = self.coords_split(pts)

        if self.reduce_sum:
            output = pts.new_zeros(pts.size(0))
        else:
            output = []

        for level_tensor in self.tensors:
            output += level_tensor(coords_plane, coords_line)

        return output

class RenderingEquationEncoding(torch.nn.Module):
    def __init__(self, num_theta, num_phi, device):
        super(RenderingEquationEncoding, self).__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi

        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)

    def forward(self, omega_o, a, la, mu):
        Smooth = F.relu((omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True)) # N, num_theta, num_phi, 1

        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (self.omega_la * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2) -mu * (self.omega_mu * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2)
        out = a * Smooth * torch.exp(exp_input)

        return out

class RenderingNet(torch.nn.Module):
    def __init__(self, num_theta = 8, num_phi=16, data_dim_color=192, featureC=256, device='cpu'):
        super(RenderingNet, self).__init__()

        self.ch_cd = 3
        self.ch_s = 3
        self.ch_normal = 3
        self.ch_bottleneck = 128

        self.num_theta = 8
        self.num_phi = 16
        self.num_asg = self.num_theta * self.num_phi

        self.ch_asg_feature = 128
        self.ch_per_theta = self.ch_asg_feature // self.num_theta

        self.ch_a = 2
        self.ch_la = 1
        self.ch_mu = 1
        self.ch_per_asg = self.ch_a + self.ch_la + self.ch_mu

        self.ch_normal_dot_viewdir = 1


        self.ree_function = RenderingEquationEncoding(num_theta, num_phi, device)

        self.spatial_mlp = torch.nn.Sequential(
                torch.nn.Linear(data_dim_color, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, self.ch_cd + self.ch_s + self.ch_bottleneck + self.ch_normal + self.ch_asg_feature)).to(device)

        self.asg_mlp = torch.nn.Sequential(torch.nn.Linear(self.ch_per_theta, self.num_phi * self.ch_per_asg)).to(device)

        self.directional_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.ch_bottleneck + self.num_asg * self.ch_a + self.ch_normal_dot_viewdir, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, 3)).to(device)


    def spatial_mlp_forward(self, x):
        out = self.spatial_mlp(x)
        sections = [self.ch_cd, self.ch_s, self.ch_normal, self.ch_bottleneck, self.ch_asg_feature]
        diffuse_color, tint, normals, bottleneck, asg_features = torch.split(out, sections, dim=-1)
        normals = -F.normalize(normals, dim=1)
        return diffuse_color, tint, normals, bottleneck, asg_features

    def asg_mlp_forward(self, asg_feature):
        N = asg_feature.size(0)
        asg_feature = asg_feature.view(N, self.num_theta, -1)
        asg_params = self.asg_mlp(asg_feature)
        asg_params = asg_params.view(N, self.num_theta, self.num_phi, -1)

        a, la, mu = torch.split(asg_params, [self.ch_a, self.ch_la, self.ch_mu], dim=-1)
        return a, la, mu

    def directional_mlp_forward(self, x):
        out = self.directional_mlp(x)
        return out

    def reflect(self, viewdir, normal):
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out

    def forward(self, viewdir, feature):
        diffuse_color, tint, normal, bottleneck, asg_feature = self.spatial_mlp_forward(feature)
        refdir = self.reflect(-viewdir, normal)

        a, la, mu = self.asg_mlp_forward(asg_feature)
        ree = self.ree_function(refdir, a, la, mu) # N, num_theta, num_phi, ch_per_asg
        ree = ree.view(ree.size(0), -1)

        normal_dot_viewdir = ((-viewdir) * normal).sum(dim=-1, keepdim=True)
        dir_mlp_input = torch.cat([bottleneck, ree, normal_dot_viewdir], dim=-1)
        specular_color = self.directional_mlp_forward(dir_mlp_input)

        raw_rgb = diffuse_color + tint * specular_color
        rgb = torch.sigmoid(raw_rgb)

        return rgb, normal


########################################################################################

class NRFF(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(NRFF, self).__init__(aabb, gridSize, device, **kargs)

        self.rendering_net = RenderingNet(8, 16, device=device)
        self.init_feature_field(device)

    def init_feature_field(self, device):
        self.density_field = MultiscaleTensorDecom(num_levels=16, num_features=2, base_resolution=16, max_resolution=512, device=device, reduce_sum=True)
        self.appearance_field = MultiscaleTensorDecom(num_levels=16, num_features=4, base_resolution=16, max_resolution=512, device=device)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = []

        grad_vars += [{'params': self.density_field.parameters(), 'lr': lr_init_spatialxyz}]
        grad_vars += [{'params': self.appearance_field.parameters(), 'lr': lr_init_spatialxyz}]
        grad_vars += [{'params': self.rendering_net.parameters(), 'lr':lr_init_network}]

        return grad_vars


    def density_L1(self):
        return self.density_field.L1loss()

    def compute_densityfeature(self, pts):
        output = self.density_field(pts)
        return output

    def compute_appfeature(self, pts):
        app_feature = self.appearance_field(pts)
        app_feature = torch.cat(app_feature, dim=-1)
        return app_feature

    @torch.no_grad()
    def shrink(self, new_aabb):
        self.train_aabb = new_aabb

        self.density_field.shrink(self.aabb.cpu(), new_aabb.cpu())
        self.appearance_field.shrink(self.aabb.cpu(), new_aabb.cpu())

        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units


        t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long()
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb

        self.density_field.upsample(new_aabb.cpu())
        self.appearance_field.upsample(new_aabb.cpu())

        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_stepSize(res_target)
