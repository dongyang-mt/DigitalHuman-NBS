import sys
# import bpy
import numpy as np
import random
import argparse
import logging

logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)

def generate_color_table(num_colors):
    """
    Generate num_colors of colors from matplotlib's tableau colors (saved as npy), repeat if necessary
    @param num_colors: int, number of colors needed
    @return: rgb, (num_colors, 3)
    """
    base_colors = np.load('./blender_scripts/tableau_color.npy')
    print("base_colors.shape=", base_colors.shape)
    idx = list(range(base_colors.shape[0]))
    logging.warning("----")
    random.seed(5)
    random.shuffle(idx)
    pt = 0
    res = []
    for i in range(num_colors):
        res.append(base_colors[idx[pt]])
        pt += 1
        pt %= base_colors.shape[0]
    logging.debug(len(res))
    print("len(res)=", len(res))
    print("len(res[0])=", len(res[0]))

    logging.debug(res)
    res_np = np.array(res)
    print("res_np.shape=", res_np.shape)
    return np.array(res)


def weight2color(weight):
    n_color = weight.shape[1]
    colors = generate_color_table(n_color)
    res = np.matmul(weight, colors)
    logging.warning("weight.shape="+str(weight.shape))
    logging.warning("colors.shape="+str(colors.shape))
    logging.warning("res.shape="+str(res.shape)+str(res.dtype))
    print(res)
    return res


def add_material_for_mesh(objs):
    mat = bpy.data.materials.new(name='VertexColor')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    vert_color = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")

    bsdf.inputs[5].default_value = 0

    links = mat.node_tree.links
    links.new(vert_color.outputs[0], bsdf.inputs[0])

    for obj in objs:
        if len(obj.data.materials):
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def load_vert_col(me, colors):
    vcols = me.data.vertex_colors
    polys = me.data.polygons

    vcol = vcols.new(name="Visualization")
    logging.warning("load_vert_col")
    print(vcols)
    print(polys)

    idx = 0
    for poly in polys:
        verts = poly.vertices
        for i, _ in enumerate(poly.loop_indices):
            c = colors[verts[i]]
            vcol.data[idx].color = (c[0], c[1], c[2], 1.0)
            idx += 1


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='./demo/T-pose.obj')
    parser.add_argument('--weight_path', type=str, default='./demo/weight.npy')
    return parser


def change_shading_mode(shading_mode):
    """
    https://blender.stackexchange.com/questions/124347/blender-2-8-python-code-to-switch-shading-mode-between-wireframe-and-solid-mo/124427
    """
    for area in bpy.context.workspace.screens[0].areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = shading_mode


if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)

    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete()

    weight = np.load(args.weight_path)
    color = weight2color(weight)
    print("np.max(color)=", np.max(color))
    print("color[0]=", color[0])
    print("color.dtype", color.dtype)

    # color[0]=
    print("weight.shape=", weight.shape)
    print("color.shape=", color.shape)
    import open3d as o3d
    mesh_np = o3d.io.read_triangle_mesh(args.obj_path)
    # mesh_np.vertex_colors = o3d.utility.Vector3dVector(color)
    color2 = np.random.uniform(0, 1, size=(color.shape[0], 3))
    print("color2.dtype", color2.dtype)
    # mesh_np.vertex_colors = o3d.utility.Vector3dVector(
    #     np.random.uniform(0, 1, size=(color.shape[0], 3)))
    N=6890
    # color3 = np.repeat([[1,0,0]], repeats=[N],axis=1)
    color3 = np.zeros_like(color)
    color3[:,2]=1
    # color3[0:2000,2]=1
    # color3[2000:,1]=1
    print("color3.shape", color3.shape)
    print("color3.dtype", color3.dtype)
    for i in [0,10,100,1000,3000]:
        print("color3[]", i, color3[i])
    mesh_np.vertex_colors = o3d.utility.Vector3dVector(color3)
    # mesh_np.vertex_colors = o3d.utility.Vector3dVector(
    #     np.random.uniform(0, 1, size=(N, 3)))
    # mesh_np.vertex_colors = color
    # mesh_np.paint_uniform_color([0,1,0])
    # mesh_np.paint_uniform_color(color3[0])
    # mesh_np.compute_vertex_normals()
    
    print(np.asarray(mesh_np.triangle_normals))
    print(np.asarray(mesh_np.vertex_colors))
    print(np.asarray(mesh_np.vertex_colors).dtype)
    mesh_colors = np.asarray(mesh_np.vertex_colors)
    print(color3 == mesh_colors)
    print((color3 == mesh_colors).all())
    print("Displaying mesh made using numpy ...")
    # o3d.visualization.draw_geometries([mesh_np], mesh_show_wireframe=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_np.vertices)
    # Add color and estimate normals for better visualization.
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # pcd.colors = o3d.utility.Vector3dVector(color3)
    pcd.paint_uniform_color([0.5, 0, 0])
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying Open3D pointcloud made using numpy array ...")
    o3d.visualization.draw([pcd])

    # bpy.ops.import_scene.obj(filepath=args.obj_path, split_mode='OFF')
    # me = bpy.context.selected_objects[0]
    # bpy.ops.object.shade_smooth()

    # load_vert_col(me, color)
    # add_material_for_mesh([me])

    # change_shading_mode('MATERIAL')
