import open3d as o3d
import numpy as np
import torch

def write_numpy_torch_verts_faces_to_mesh(verts, faces=None, path="temp.obj", display=False):
    verts_o3d = faces_o3d = None
    if verts and isinstance(verts, np.array):
        verts_o3d = o3d.utility.Vector3iVector(verts)
    if verts and isinstance(faces, torch.Tensor):
        faces_o3d = o3d.utility.Vector3iVector(torch.numpy(faces))
    mesh = o3d.geometry.TriangleMesh()
    if verts_o3d:
        mesh.vertices = verts_o3d
    if faces_o3d:
        mesh.triangles = faces_o3d
    o3d.io.write_triangle_mesh(path, mesh)
    if display:
        o3d.visualization.draw([mesh])
