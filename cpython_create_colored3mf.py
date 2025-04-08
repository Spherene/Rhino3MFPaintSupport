#! python 3
# r: numpy
# r: scipy

import numpy as np

import rhinoscriptsyntax as rs
import xml.etree.ElementTree as ET
import sys
import io 
from scipy.spatial import KDTree
import shutil

def generate_xml(template_path, points, faces, supported_faces):
    ins = open(template_path).read()
    fmt = '\t<vertex x="{}" y="{}" z="{}"/>\n'
    vertex_stream = io.StringIO()
    for v in points:
        vertex_stream.write(fmt.format(*v))
    ins = ins.replace("%vertices%", vertex_stream.getvalue())
    fmt =  '\t<triangle v1="{}" v2="{}" v3="{}"/>\n'
    face_stream = io.StringIO()
    for f in faces:
        face_stream.write(fmt.format(*f))
    fmt =  '\t<triangle v1="{}" v2="{}" v3="{}" paint_supports="4"/>\n'
    for f in supported_faces:
        face_stream.write(fmt.format(*f))
    ins = ins.replace("%triangles%", face_stream.getvalue())
    return ins


def get_neighbourhood(ind):
    ifaces = np.isin(faces, ind)
    cond = np.any(ifaces, axis=1)
    return cond

def get_green_downward_faces(mesh):
    normals = mesh.FaceNormals
    normals = np.array([tuple(n) for n in normals])
    colors = mesh.VertexColors
    greens = np.array([c.G for c in colors])
    cond1 = greens > 200
    faces = np.array([[f.A, f.B, f.C] for f in mesh.Faces])
    fcond1 = cond1[faces]
    cond1 = np.any(fcond1, axis=1)
    cond2 = normals[:,2]  < -0.8
    return cond1 & cond2

def get_supported_faces(points, faces, support_points):
    tree = KDTree(points)
    dists, ind = tree.query(support_points)
    cond = get_neighbourhood(ind)
    tot_cond = cond
    for j in range(0):
        print(f"{j} {np.sum(cond)=}")
        ind = np.where(cond)[0]
        cond = get_neighbourhood(ind)
        tot_cond = np.logical_or(cond, tot_cond)
    print(f"num faces supported {np.sum(tot_cond)=}")
    return tot_cond


rids = rs.SelectedObjects()

points = []
meshes = []

for rid in rids:
    if rs.IsPoint(rid):
        points.append(tuple(rs.PointCoordinates(rid)))
    elif rs.IsMesh(rid):
        meshes.append(rid)
points = np.array(points)
print(f"{len(points)=}")
if len(meshes) !=1:
    print("You have to select exactly one mesh and some support points")
mesh = meshes[0]
mesh = rs.coercerhinoobject(mesh).Geometry
cond_green_downward = get_green_downward_faces(mesh)
print(f"green_downward_faces = {np.sum(cond_green_downward)}")
mesh.Vertices.CombineIdentical(True,True)
vertices = np.array([tuple(v) for v in mesh.Vertices])

shift_z = np.array((0, 0, -vertices[:, 2].min()))
vertices += shift_z
points += shift_z
faces = mesh.Faces
faces = np.array([[f.A, f.B, f.C] for f in faces])

cond = get_supported_faces(vertices, faces, points)
cond = cond | cond_green_downward
unsupported = faces[~cond]
supported = faces[cond]


import os
cpath =os.path.dirname(__file__)
template_path =  cpath+"/template"
out_path =  cpath+"/support"

try:
    shutil.rmtree(out_path)
except:
    pass
print(out_path)

shutil.copytree(template_path, out_path)

fname = "/3D/Objects/object_1.model"
path1 = template_path + fname
path2 = out_path + fname
import shutil
s = generate_xml(path1, vertices, unsupported, supported)
open(path2, "w").write(s)

name = "out"
created_archive_path = shutil.make_archive(
            base_name= os.path.join(cpath, name),
            format='zip',
            root_dir=out_path,    # Needs to be a string
            base_dir="."
        )

fname = os.path.join(cpath, name + ".3mf")
if os.path.isfile(fname):
    os.remove(fname)
os.rename(os.path.join(cpath, name)+".zip",fname)


try:
    shutil.rmtree(out_path)
except:
    print("not possible")