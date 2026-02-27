"""
Mixamo FBX -> VRMA converter using a target VRM rig.

Usage:
    blender --background --python tools/mixamo_to_vrma.py -- input.fbx output.vrma target.vrm
"""

import bpy
import math
import os
import sys

BONE_MAP = {
    "mixamorig:Hips": "Hips",
    "mixamorig:Spine": "Spine",
    "mixamorig:Spine1": "Chest",
    "mixamorig:Spine2": "UpperChest",
    "mixamorig:Neck": "Neck",
    "mixamorig:Head": "Head",
    "mixamorig:LeftShoulder": "Shoulder_L",
    "mixamorig:LeftArm": "UpperArm_L",
    "mixamorig:LeftForeArm": "LowerArm_L",
    "mixamorig:LeftHand": "Hand_L",
    "mixamorig:RightShoulder": "Shoulder_R",
    "mixamorig:RightArm": "UpperArm_R",
    "mixamorig:RightForeArm": "LowerArm_R",
    "mixamorig:RightHand": "Hand_R",
    "mixamorig:LeftUpLeg": "UpperLeg_L",
    "mixamorig:LeftLeg": "LowerLeg_L",
    "mixamorig:LeftFoot": "Foot_L",
    "mixamorig:LeftToeBase": "Toes_L",
    "mixamorig:RightUpLeg": "UpperLeg_R",
    "mixamorig:RightLeg": "LowerLeg_R",
    "mixamorig:RightFoot": "Foot_R",
    "mixamorig:RightToeBase": "Toes_R",
}


def parse_args():
    if "--" not in sys.argv:
        print("Usage: blender --background --python mixamo_to_vrma.py -- input.fbx output.vrma target.vrm")
        sys.exit(1)
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 3:
        print("Error: need input.fbx output.vrma target.vrm")
        sys.exit(1)
    return args[0], args[1], args[2]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes) + list(bpy.data.armatures) + list(bpy.data.actions):
        bpy.data.batch_remove([block])


def import_fbx(path):
    sys.path.append("/usr/share/blender/5.0/scripts/addons_core")
    from io_scene_fbx import import_fbx as _fbx_loader

    class _FakeOp:
        use_custom_normals = True
        use_subsurf = False
        use_image_search = True
        use_alpha_decals = False
        decal_offset = 0.0
        use_anim = True
        anim_offset = 1.0
        use_custom_props = True
        use_custom_props_enum_as_string = True
        ignore_leaf_bones = False
        force_connect_children = False
        automatic_bone_orientation = False
        primary_bone_axis = "Y"
        secondary_bone_axis = "X"
        use_prepost_rot = True
        axis_forward = "-Z"
        axis_up = "Y"
        global_scale = 1.0
        bake_space_transform = False

        def report(self, level, msg):
            print(f"[{level}] {msg}")

    _fbx_loader.load(_FakeOp(), bpy.context, filepath=path)


def remap_bones():
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        for bone in obj.data.bones:
            new_name = BONE_MAP.get(bone.name)
            if new_name:
                bone.name = new_name

    for action in bpy.data.actions:
        fcurves = []
        if action.is_action_layered:
            for layer in action.layers:
                for strip in layer.strips:
                    if hasattr(strip, "channelbags"):
                        for cb in strip.channelbags:
                            fcurves.extend(cb.fcurves)
        else:
            fcurves = list(action.fcurves)
        for fc in fcurves:
            dp = fc.data_path
            for old, new in BONE_MAP.items():
                if f'"{old}"' in dp:
                    fc.data_path = dp.replace(f'"{old}"', f'"{new}"')
                    break


def get_armature(exclude=None):
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name != exclude:
            return obj
    return None


def import_vrm(path):
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.vrm")
    bpy.ops.import_scene.vrm(filepath=path)


def prepare_vrm1_humanoid(target_arm):
    ext = target_arm.data.vrm_addon_extension
    if hasattr(ext, "SPEC_VERSION_VRM1"):
        ext.spec_version = ext.SPEC_VERSION_VRM1

    # Clear stale VRM0 humanoid assignments if present.
    try:
        vrm0_human_bones = ext.vrm0.humanoid.human_bones
        while len(vrm0_human_bones) > 0:
            vrm0_human_bones.remove(0)
    except Exception:
        pass

    result = bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(
        armature_object_name=target_arm.name
    )
    print(f"assign_vrm1_humanoid_human_bones_automatically: {result}")

    ext = target_arm.data.vrm_addon_extension
    if hasattr(ext, "is_vrm1") and not ext.is_vrm1():
        print("Error: target armature is not VRM 1.0 after preparation")
        sys.exit(1)
    if not ext.vrm1.humanoid.human_bones.all_required_bones_are_assigned():
        print("Error: required VRM 1.0 humanoid bones are not fully assigned")
        sys.exit(1)


def normalize_source_armature(source_arm):
    # Mixamo FBX often carries object-level axis correction.
    # Bake it away so pose-space constraints don't inherit a flipped basis.
    bpy.ops.object.select_all(action="DESELECT")
    source_arm.select_set(True)
    bpy.context.view_layer.objects.active = source_arm

    source_arm.rotation_mode = "XYZ"
    source_arm.rotation_euler = (math.radians(90.0), 0.0, math.radians(180.0))
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


def retarget_with_constraints(source_arm, target_arm):
    shared = {b.name for b in source_arm.data.bones} & {b.name for b in target_arm.data.bones}
    print(f"Shared bones: {len(shared)}")
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.object.mode_set(mode="POSE")
    for bone_name in shared:
        pose_bone = target_arm.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        if bone_name == "Hips":
            # Hips rotation often carries global axis mismatch and flips the rig.
            continue
        c = pose_bone.constraints.new("COPY_ROTATION")
        c.target = source_arm
        c.subtarget = bone_name
        # Copy pose-space rotation to avoid local-axis mismatch between rigs.
        c.target_space = "POSE"
        c.owner_space = "POSE"
        c.use_x = True
        c.use_y = True
        c.use_z = True
    bpy.ops.object.mode_set(mode="OBJECT")


def bake_to_target(source_arm, target_arm):
    frame_start, frame_end = 1, 100
    if source_arm.animation_data and source_arm.animation_data.action:
        r = source_arm.animation_data.action.frame_range
        frame_start = int(r[0])
        frame_end = int(r[1])

    bpy.context.view_layer.objects.active = target_arm
    target_arm.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=True,
        bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    if target_arm.animation_data and target_arm.animation_data.action:
        target_arm.animation_data.action.name = "Idle"


def export_vrma(path, target_arm):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    bpy.context.view_layer.objects.active = target_arm
    prereq = bpy.ops.wm.vrma_export_prerequisite(
        armature_object_name=target_arm.name,
    )
    print(f"VRMA prerequisite result: {prereq}")
    if "FINISHED" not in prereq:
        print("Warning: VRMA export prerequisite was not FINISHED; trying export anyway")
    result = bpy.ops.export_scene.vrma(
        filepath=path,
        armature_object_name=target_arm.name,
    )
    print(f"VRMA export result: {result}")
    if not os.path.exists(path):
        print(f"Error: VRMA export did not create file: {path}")
        sys.exit(1)


def main():
    input_path, output_path, vrm_path = parse_args()
    if not os.path.exists(input_path):
        print(f"Error: input not found: {input_path}")
        sys.exit(1)
    if not os.path.exists(vrm_path):
        print(f"Error: target VRM not found: {vrm_path}")
        sys.exit(1)

    clear_scene()
    import_fbx(input_path)
    remap_bones()
    source_arm = get_armature()
    if source_arm is None:
        print("Error: no armature found after FBX import")
        sys.exit(1)
    source_arm.name = "SourceArmature"

    import_vrm(vrm_path)
    target_arm = get_armature(exclude="SourceArmature")
    if target_arm is None:
        print("Error: VRM armature not found after import")
        sys.exit(1)

    prepare_vrm1_humanoid(target_arm)
    normalize_source_armature(source_arm)
    retarget_with_constraints(source_arm, target_arm)
    bake_to_target(source_arm, target_arm)
    export_vrma(output_path, target_arm)
    print(f"Done: {output_path}")


main()
