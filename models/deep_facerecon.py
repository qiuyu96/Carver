"""Contains the implementation of 3D face reconstruction model.

This file is mostly borrowed from:

https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/master/models

We employ an off-the-shelf 3D face reconstruction model to obtain the
'ground-truth' face geometries and poses,
which are subsequently utilized to evaluate the metric of 3D-aware GANs.

The face reconstruction model is introduced in paper:

https://arxiv.org/abs/1903.08527
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.facerecon_networks import define_net_recon
from array import array
from scipy.io import loadmat
from scipy.io import savemat
from typing import List

try:
    import nvdiffrast.torch as dr
except:
    pass


def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3)
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()


class SH:

    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [
            1 / np.sqrt(4 * np.pi),
            np.sqrt(3.) / np.sqrt(4 * np.pi),
            3 * np.sqrt(5.) / np.sqrt(12 * np.pi)
        ]


# load expression basis
def LoadExpBasis(bfm_folder='BFM'):
    n_vertex = 53215
    Expbin = open(os.path.join(bfm_folder, 'Exp_Pca.bin'), 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(Expbin, 1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(Expbin, 3*n_vertex)
    expPC.fromfile(Expbin, 3*exp_dim[0]*n_vertex)
    Expbin.close()

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(os.path.join(bfm_folder, 'std_exp.txt'))

    return expPC, expEV


# transfer original BFM09 to our face model
def transferBFM09(bfm_folder='BFM'):
    print('Transfer BFM09 to BFM_model_front......')
    original_BFM = loadmat(os.path.join(bfm_folder, '01_MorphableModel.mat'))
    shapePC = original_BFM['shapePC']  # shape basis
    shapeEV = original_BFM['shapeEV']  # corresponding eigen value
    shapeMU = original_BFM['shapeMU']  # mean face
    texPC = original_BFM['texPC']  # texture basis
    texEV = original_BFM['texEV']  # eigen value
    texMU = original_BFM['texMU']  # mean texture

    expPC, expEV = LoadExpBasis()

    # transfer BFM09 to our face model

    idBase = shapePC*np.reshape(shapeEV, [-1, 199])
    idBase = idBase/1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC*np.reshape(expEV, [-1, 79])
    exBase = exBase/1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC*np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # Our face model is cropped along face landmarks,
    # and contains only 35709 vertex.
    # Original BFM09 contains 53490 vertex,
    # and expression basis provided by Guo et al. contains 53215 vertex.
    # Thus we select corresponding vertex to get our face model.

    index_exp = loadmat(os.path.join(bfm_folder, 'BFM_front_idx.mat'))
    # starts from 0 (to 53215)
    index_exp = index_exp['idx'].astype(np.int32) - 1

    index_shape = loadmat(os.path.join(bfm_folder, 'BFM_exp_idx.mat'))
    index_shape = index_shape['trimIndex'].astype(
        np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3])/1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    other_info = loadmat(os.path.join(bfm_folder, 'facemodel_info.mat'))
    frontmask2_idx = other_info['frontmask2_idx']
    skinmask = other_info['skinmask']
    keypoints = other_info['keypoints']
    point_buf = other_info['point_buf']
    tri = other_info['tri']
    tri_mask2 = other_info['tri_mask2']

    # save our face model
    savemat(
        os.path.join(bfm_folder, 'BFM_model_front.mat'), {
            'meanshape': meanshape,
            'meantex': meantex,
            'idBase': idBase,
            'exBase': exBase,
            'texBase': texBase,
            'tri': tri,
            'point_buf': point_buf,
            'tri_mask2': tri_mask2,
            'keypoints': keypoints,
            'frontmask2_idx': frontmask2_idx,
            'skinmask': skinmask
        })


class ParametricFaceModel:
    def __init__(self,
                bfm_folder='./BFM',
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,
                center=112.,
                is_train=True,
                default_name='BFM_model_front.mat'):

        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            transferBFM09(bfm_folder)
        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3*N,1]
        self.mean_shape = model['meanshape'].astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)
        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)
        # texture basis. [3*N,80]
        self.tex_base = model['texBase'].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if is_train:
            # vertex indices for small face region to compute photometric error.
            # starts from 0.
            self.front_mask = np.squeeze(model['frontmask2_idx']).astype(
                np.int64) - 1
            # vertex indices for each face from small face region.
            # starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # vertex indices for pre-defined skin region to compute
            # reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        self.persc_proj = perspective_projection(focal, center)
        self.device = 'cpu'
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)


    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))


    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])


    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture
                -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff
                -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base,
                                    tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, -1, 3])


    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat(
            [face_norm,
             torch.zeros(face_norm.shape[0], 1, 3).to(self.device)],
            dim=1)

        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm


    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color
                -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture
                -- torch.tensor, size (B, N, 3), from texture model,
                   range (0, 1.)
            face_norm
                -- torch.tensor, size (B, N, 3), rotated face normal
            gamma
                -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
            a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2], a[1] * c[1] *
            face_norm[..., 2:], -a[1] * c[1] * face_norm[..., :1],
            a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:]**2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:], 0.5 *
            a[2] * c[2] * (face_norm[..., :1]**2 - face_norm[..., 1:2]**2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    def compute_rotation(self, angles):
        """
        Return:
            rot
                -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles
                -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj
                -- torch.tensor, size (B, N, 2),
                   y direction is opposite to v direction

        Parameters:
            face_shape
                -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape
                -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape
                -- torch.tensor, size (B, N, 3)
            rot
                -- torch.tensor, size (B, 3, 3)
            trans
                -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def get_landmarks(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)

        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }
    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex
                -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color
                -- torch.tensor, size (B, N, 3), in RGB order
            landmark
                -- torch.tensor, size (B, 68, 2),
                   y direction is opposite to v direction
        Parameters:
            coeffs
                -- torch.tensor, size (B, 257)
        """
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_range = torch.cat((face_shape.min(1)[0],face_shape.max(1)[0]))
        face_range = torch.cat((face_range, face_shape[0,::500]),0)
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted,
                                        coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark, face_range


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0], [0, n / -x, 0, 0],
                     [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10,
                rasterize_size=224):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(
            x=x, n=znear,
            f=zfar)).matmul(torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.glctx = None

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask
                -- torch.tensor, size (B, 1, H, W)
            depth
                -- torch.tensor, size (B, 1, H, W)
            features(optional)
                -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex
                -- torch.tensor, size (B, N, 3)
            tri
                -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)
                -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices,
        # the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat(
                [vertex, torch.ones([*vertex.shape[:2], 1]).to(device)],
                dim=-1)
            vertex[..., 1] = -vertex[..., 1]


        vertex_ndc = vertex @ ndc_proj.t()
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=device)
            print("create glctx on device cuda:%d"%device.index)

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0]
                                 for f in tri]).unsqueeze(1).to(device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum],
                               axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3],
        # for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx,
                                   vertex_ndc.contiguous(),
                                   tri,
                                   resolution=[rsize, rsize],
                                   ranges=ranges)

        depth, _ = dr.interpolate(
            vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(),
            rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image

class DeepFaceRecon(torch.nn.Module):
    def __init__(self, bfm_folder='./BFM',
                 weight_path='checkpoints/pretrained/epoch_20.pth'):
        super().__init__()
        self.net_recon = define_net_recon(
            net_recon='resnet50', use_last_fc=False, init_path=None)
        self.model_names = ['net_recon']

        focal = 1015.0
        center = 112.0
        self.facemodel = ParametricFaceModel(bfm_folder=bfm_folder,
                                             camera_distance=10.0,
                                             focal=focal,
                                             center=center,
                                             is_train=False)

        fov = 2 * np.arctan(center / focal) * 180 / np.pi
        self.renderer = MeshRenderer(rasterize_fov=fov,
                                     znear=5.0,
                                     zfar=15.0,
                                     rasterize_size=int(2 * center))

        state_dict = torch.load(weight_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])

    def forward(self, input_img):
        output_coeff = self.net_recon(input_img)
        self.facemodel.to(input_img.device)
        (self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm,
         self.pred_range) = self.facemodel.compute_for_render(output_coeff)
        self.pred_mask, self.pred_depth, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)

        output_depth_numpy = self.pred_depth  # [N, 1, H, W]

        return {'depth': output_depth_numpy,
                'angle': self.pred_coeffs_dict['angle'],
                'trans': self.pred_coeffs_dict['trans']}


if __name__ == '__main__':
    face_model = DeepFaceRecon().cuda()
    face_model.eval()
    im = cv2.imread('test.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224))
    im = torch.tensor(np.array(im)/255. * 2.0 - 1.0,
                      dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    im = im / 2.0 + 0.5
    output = face_model(im)
    print(output)
