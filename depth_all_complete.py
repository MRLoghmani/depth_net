#---------------------------------------------------
# File nodes.py
#---------------------------------------------------
import bpy
import mathutils
from math import *
import sys,random,time
from os.path import join
import os,random

random.seed()
start_time = time.time()
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
filepath=argv[0]
output_root_dir=argv[1]
tag=argv[2]
n_samples=int(argv[3])
#import code
#code.interact(local=locals())
path=os.path.dirname(filepath)
categorydir=os.path.basename(os.path.normpath(path))

file_name=bpy.path.display_name_from_filepath(filepath) #get the .blend name without extension and path

#png16_path = join(output_root_dir, "16bit", categorydir,file_name)
png8_path = join(output_root_dir, "8bit", categorydir,file_name)
category=path.split('/').pop()
#if not os.path.exists(png16_path):
#    os.makedirs(png16_path)
if not os.path.exists(png8_path):
    os.makedirs(png8_path)

######cleaning_py code!
scene = bpy.context.scene
scene.use_nodes = True
nodes = scene.node_tree.nodes
bpy.ops.object.select_all(action = 'DESELECT')
for ob in scene.objects:
    if ob.type == 'MESH':
        ob.select = True
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM') # remove all parents
bpy.ops.object.select_all(action = 'DESELECT')
for ob in scene.objects:
    if ob.type != 'MESH':
        ob.select = True
bpy.ops.object.delete() #remove all non meshes
bpy.ops.object.select_all()
scene.objects.active = bpy.data.objects[0]
bpy.ops.object.join()
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
ob = bpy.context.object
ob.name = "All Meshes"
ob.dimensions = ob.dimensions / max(ob.dimensions) #scaling on max dimension
scene.cursor_location = (0.0,0.0,0.0)
# set the origin on the current object to the 3dcursor location
bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
#bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

#mesh_obj = bpy.context.active_object
#minz = 999999.0
#for vertex in ob.data.vertices:
#    # object vertices are in object space, translate to world space
#    v_world = mesh_obj.matrix_world * mathutils.Vector((vertex.co[0],vertex.co[1],vertex.co[2]))
#    if v_world[2] < minz:
#        minz = v_world[2]
#        print(str(vertex.co) + " " + str(v_world[2]))
#ob.location.z = -(minz)
#######ending of cleaning.py code!


#set scene unit to Metrics (centimeters!)
scene.unit_settings.system='METRIC'

#set scene resolution&engine:
scene.render.tile_x=16
scene.render.tile_y=16
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
ZoomDist=scene.render.resolution_x/256 #value that I need to multiply to Camdist to get the "256px size"
scene.render.engine = 'CYCLES' # We use the Cycles Render
scene.render.threads_mode = 'FIXED'
scene.render.threads = 2
scene.render.use_local_coords=True
scene.render.image_settings.color_depth='8'
scene.render.image_settings.color_mode='BW'
scene.render.image_settings.compression=100
scene.render.use_persistent_data=True
#import code
#code.interact(local=locals())
#bpy.data.scenes['Scene'].render.image_settings.color_mode='BW'
cycles=scene.cycles
#cycles.samples = 128
cycles.max_bounces=0
cycles.glossy_bounces=0
cycles.min_bounces=0
cycles.diffuse_bounces=0
cycles.caustics_reflective=False
cycles.caustics_refractive=False
#cycles.use_cache=True#increase the performance: a lot!


#set a new camera and its target:All meshes!
cam = bpy.data.cameras.new("Camera")
cam.clip_end=15.0 #not working for z depth, still not clipping it.
cam_ob = bpy.data.objects.new("Camera", cam)
scene.objects.link(cam_ob)
scene.camera = cam_ob
cam_ob.location=(2,0,5)
ob_target = bpy.data.objects.get('All Meshes', False)
track=bpy.data.objects["Camera"].constraints.new(type='TRACK_TO')
track.target=ob_target
track.up_axis = 'UP_Y'
track.track_axis='TRACK_NEGATIVE_Z'

#I DO NOT ADD A PLANE, TO MAKE 360° views ;)
#bpy.ops.mesh.primitive_plane_add()
#ob2 = bpy.context.object
#ob2.dimensions = (100,100,1)

#Use the default nodes generated in blender 
render_layers = nodes['Render Layers']
output_viewer = nodes['Composite']

#Add Map Value node for scaling and normalizing the Z values
norm = nodes.new('CompositorNodeMapRange')
inputs = norm.inputs
inputs["From Max"].default_value = 6.5
inputs["To Min"].default_value = 0.0
inputs["To Max"].default_value = 1.0

# bpy.ops.node.add_node(type="CompositorNodeMapRange", use_transform=True)
#Add output DepthImg file node
output_8bit_png = nodes.new('CompositorNodeOutputFile')
output_8bit_png.base_path = png8_path
output_8bit_png.format.color_mode='BW'
output_8bit_png.format.color_depth='8'
output_8bit_png.format.file_format='PNG'

#Add output 16bit file node
#output_16bit_png = nodes.new('CompositorNodeOutputFile')
#output_16bit_png.base_path = png16_path
#output_16bit_png.format.color_mode='BW'
#output_16bit_png.format.color_depth='16'
#output_16bit_png.format.file_format='PNG'

scene.node_tree.links.new(
        render_layers.outputs['Z'],
        norm.inputs['Value']
    )
scene.node_tree.links.new(
        norm.outputs['Value'],
        output_8bit_png.inputs[0]
    )
#scene.node_tree.links.new(
#        norm.outputs['Value'],
#        output_16bit_png.inputs[0]
#    )
CoRX=0.0
CoRY=0.0
CoR=(CoRX,CoRY,0)
CamStartDist=3.8
CamDist=0.0#to be sure it is a float!
counter=0
axis = None
angle = 0
lensStandard=35.0
dimStandardX=ob.scale[0]
dimStandardY=ob.scale[1]
dimStandardZ=ob.scale[2]
size=1
CamDistStandard=ZoomDist*CamStartDist/(size)

sintheta_out = sin(0.12566370614) #-->pi/4 # radians(random.random()*360)
cosphi_out = cos(0.12566370614) #--pi/4  #radians(random.random()*360)
for n in range (1,n_samples+1):
   # size=random.random()*0.6 + 0.9   #1.6 as maximum 
    counter=counter+1   
   
    ob.scale[0] = dimStandardX * (1 + random.random()*0.2 - 0.1)
    ob.scale[1] = dimStandardY * (1 + random.random()*0.2 - 0.1)
    ob.scale[2] = dimStandardZ * (1 + random.random()*0.2 - 0.1)        
    cam.lens =random.randint(16,95)
    CamDist =(0.4 + random.random()/5)* (CamDistStandard/lensStandard)  * cam.lens  
    cam_ob.location = (CoRX + CamDist*cos(phi_out)*sin(theta_out),CoRY + CamDist*sin(phi_out)*sin(theta_out),CamDist*cos(theta_out))  
    ob.rotation_euler = (random.random()*360.0,random.random()*360.0,random.random()*360.0)

    str1=category+'_'+file_name+'_%d_%s' % (counter, tag)
   # output_16bit_png.file_slots[0].path = str1  
    output_8bit_png.file_slots[0].path = str1            
    bpy.ops.render.render(write_still=True, use_viewport=True)
       
print("done in: %d ",time.time() - start_time)
os._exit(0)
