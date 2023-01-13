import os
import open3d as o3d
import numpy as np
import torch

def write_numpy_torch_verts_faces_to_mesh(verts, faces=None, path="temp.obj", display=False):
    verts_o3d = faces_o3d = None
    if isinstance(verts, np.ndarray):
        verts_o3d = o3d.utility.Vector3dVector(verts)
    if isinstance(faces, np.ndarray):
        faces_o3d = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(verts_o3d, faces_o3d)
    # if verts_o3d:
    #     mesh.vertices = verts_o3d
    # if faces_o3d:
    #     mesh.triangles = faces_o3d
    o3d.io.write_triangle_mesh(path, mesh)
    if display:
        o3d.visualization.draw([mesh])

# all TriangleMesh with 4890 points and 9776 triangles. (6890,) 4890 False
def print_smpl_obj_data_info():
    folder = "./dataset_train/Meshes/SMPL/topology"
    v_mask_list = []
    for name in os.listdir(folder):
        obj_path = os.path.join(folder, name + "/T-pose.obj")
        mask_path = os.path.join(folder, name + "/v_mask.npy")
        mesh = o3d.io.read_triangle_mesh(obj_path)
        v_mask = np.load(mask_path)
        v_mask.sum()
        v_mask_list.append(v_mask) # check every v_mask is the same
        print(mesh, v_mask.shape, v_mask.sum(), (v_mask == v_mask_list[0]).all())


# all offset.npy(24, 3) pose.npy(72,) shape.npy(10,) t-pose.npy(27554, 3) 
def print_MultiGarment_obj_data_info():
    folder = "./dataset_train/Meshes/MultiGarment"
    for name in os.listdir(folder):
        shape_str = ""
        for npy_name in ["offset.npy", "pose.npy", "shape.npy", "t-pose.npy"]:
            npy_path = os.path.join(folder, name +"/"+ npy_name)
            data = np.load(npy_path)
            shape_str += npy_name
            shape_str += str(data.shape)
            shape_str += " "
        print(shape_str)

def load_MultiGarment_npy_to_obj():
    folder = "./dataset_train/Meshes/MultiGarment"
    folder_temp = "./temp_MultiGarment"
    for name in os.listdir(folder):
        shape_str = ""
        if not os.path.isdir(os.path.join(folder, name)):
            continue
        for npy_name in ["t-pose.npy"]:
            npy_path = os.path.join(folder, name +"/"+ npy_name)
            data = np.load(npy_path)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(data)
            folder_dst = os.path.join(folder_temp, name)
            os.makedirs(folder_dst, exist_ok=True)
            path_dst = os.path.join(folder_dst, npy_name+".obj")
            o3d.io.write_triangle_mesh(path_dst, mesh)

            shape_str += npy_name
            shape_str += str(data.shape)
            shape_str += " "
        print(shape_str)

if __name__ == "__main__":
    print_smpl_obj_data_info()