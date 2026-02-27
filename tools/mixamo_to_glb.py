"""
Mixamo FBX → GLB converter with VRM-based retargeting.

Usage:
    blender --background --python tools/mixamo_to_glb.py -- input.fbx output.glb [target.vrm]

Steps:
  1. Import FBX (Mixamo source armature)
  2. Rename Mixamo bones → VRM 0.x / MANUKA naming
  3. If target.vrm is given:
       - Import VRM as target armature
       - Add Copy Rotation constraints (target ← source)
       - Bake animation onto target with visual keying
       - Delete source armature + constraints
       - Export target armature only
  4. If no target VRM, export source directly (legacy mode)
"""

import bpy
import sys
import os

# ── Mixamo → VRM 0.x bone name map ──────────────────────────────────────────
BONE_MAP = {
    "mixamorig:Hips":           "Hips",
    "mixamorig:Spine":          "Spine",
    "mixamorig:Spine1":         "Chest",
    "mixamorig:Spine2":         "UpperChest",
    "mixamorig:Neck":           "Neck",
    "mixamorig:Head":           "Head",

    "mixamorig:LeftShoulder":   "Shoulder_L",
    "mixamorig:LeftArm":        "UpperArm_L",
    "mixamorig:LeftForeArm":    "LowerArm_L",
    "mixamorig:LeftHand":       "Hand_L",

    "mixamorig:RightShoulder":  "Shoulder_R",
    "mixamorig:RightArm":       "UpperArm_R",
    "mixamorig:RightForeArm":   "LowerArm_R",
    "mixamorig:RightHand":      "Hand_R",

    "mixamorig:LeftUpLeg":      "UpperLeg_L",
    "mixamorig:LeftLeg":        "LowerLeg_L",
    "mixamorig:LeftFoot":       "Foot_L",
    "mixamorig:LeftToeBase":    "Toes_L",

    "mixamorig:RightUpLeg":     "UpperLeg_R",
    "mixamorig:RightLeg":       "LowerLeg_R",
    "mixamorig:RightFoot":      "Foot_R",
    "mixamorig:RightToeBase":   "Toes_R",

    "mixamorig:LeftHandThumb1":  "Thumb1_L",
    "mixamorig:LeftHandThumb2":  "Thumb2_L",
    "mixamorig:LeftHandThumb3":  "Thumb3_L",
    "mixamorig:LeftHandIndex1":  "Index1_L",
    "mixamorig:LeftHandIndex2":  "Index2_L",
    "mixamorig:LeftHandIndex3":  "Index3_L",
    "mixamorig:LeftHandMiddle1": "Middle1_L",
    "mixamorig:LeftHandMiddle2": "Middle2_L",
    "mixamorig:LeftHandMiddle3": "Middle3_L",
    "mixamorig:LeftHandRing1":   "Ring1_L",
    "mixamorig:LeftHandRing2":   "Ring2_L",
    "mixamorig:LeftHandRing3":   "Ring3_L",
    "mixamorig:LeftHandPinky1":  "Pinky1_L",
    "mixamorig:LeftHandPinky2":  "Pinky2_L",
    "mixamorig:LeftHandPinky3":  "Pinky3_L",

    "mixamorig:RightHandThumb1":  "Thumb1_R",
    "mixamorig:RightHandThumb2":  "Thumb2_R",
    "mixamorig:RightHandThumb3":  "Thumb3_R",
    "mixamorig:RightHandIndex1":  "Index1_R",
    "mixamorig:RightHandIndex2":  "Index2_R",
    "mixamorig:RightHandIndex3":  "Index3_R",
    "mixamorig:RightHandMiddle1": "Middle1_R",
    "mixamorig:RightHandMiddle2": "Middle2_R",
    "mixamorig:RightHandMiddle3": "Middle3_R",
    "mixamorig:RightHandRing1":   "Ring1_R",
    "mixamorig:RightHandRing2":   "Ring2_R",
    "mixamorig:RightHandRing3":   "Ring3_R",
    "mixamorig:RightHandPinky1":  "Pinky1_R",
    "mixamorig:RightHandPinky2":  "Pinky2_R",
    "mixamorig:RightHandPinky3":  "Pinky3_R",
}


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python mixamo_to_glb.py -- input.fbx output.glb [target.vrm]")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    if len(args) < 2:
        print("Error: need input.fbx and output.glb")
        sys.exit(1)
    vrm_path = args[2] if len(args) >= 3 else None
    return args[0], args[1], vrm_path


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes) + list(bpy.data.armatures) + list(bpy.data.actions):
        bpy.data.batch_remove([block])


def import_fbx(path):
    print(f"Importing FBX: {path}")
    sys.path.append('/usr/share/blender/5.0/scripts/addons_core')
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
        primary_bone_axis = 'Y'
        secondary_bone_axis = 'X'
        use_prepost_rot = True
        axis_forward = '-Z'
        axis_up = 'Y'
        global_scale = 1.0
        bake_space_transform = False
        def report(self, level, msg): print(f"[{level}] {msg}")

    _fbx_loader.load(_FakeOp(), bpy.context, filepath=path)


def remap_bones():
    """Rename Mixamo bones and FCurve data_paths to VRM 0.x naming."""
    renamed = 0
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE':
            continue
        for bone in obj.data.bones:
            new_name = BONE_MAP.get(bone.name)
            if new_name:
                print(f"  bone: {bone.name} → {new_name}")
                bone.name = new_name
                renamed += 1

    for action in bpy.data.actions:
        fcurves = []
        if action.is_action_layered:
            for layer in action.layers:
                for strip in layer.strips:
                    if hasattr(strip, 'channelbags'):
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

    print(f"Renamed {renamed} bones.")
    return renamed


def get_armature(exclude=None):
    """Return first armature in scene, optionally excluding one by name."""
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj.name != exclude:
            return obj
    return None


def import_vrm(path):
    """Import VRM using the VRM addon (must be installed)."""
    print(f"Importing VRM: {path}")
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.vrm")
    bpy.ops.import_scene.vrm(filepath=path)


def retarget_with_constraints(source_arm, target_arm):
    """
    Set up Copy Rotation constraints on target armature bones pointing to
    the source (Mixamo) armature. Blender handles axis alignment automatically.

    Both armatures must share bone names (done by remap_bones()).
    """
    source_bone_names = {b.name for b in source_arm.data.bones}
    target_bone_names = {b.name for b in target_arm.data.bones}
    shared = source_bone_names & target_bone_names
    print(f"Shared bones: {len(shared)} (source={len(source_bone_names)}, target={len(target_bone_names)})")

    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.object.mode_set(mode='POSE')

    added = 0
    for bone_name in shared:
        pose_bone = target_arm.pose.bones.get(bone_name)
        if pose_bone is None:
            continue

        c = pose_bone.constraints.new('COPY_ROTATION')
        c.name = "MixamoRetarget"
        c.target = source_arm
        c.subtarget = bone_name
        # POSE 空間: T-pose を基準とした相対回転をコピーする。
        # LOCAL だとボーンのローカル軸がそのままコピーされて軸ズレが起きる。
        c.target_space = 'POSE'
        c.owner_space = 'POSE'
        c.use_x = True
        c.use_y = True
        c.use_z = True
        added += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Added {added} Copy Rotation constraints.")


def bake_to_target(source_arm, target_arm):
    """Bake the constrained animation onto the target armature."""
    # Get frame range from source action
    frame_start, frame_end = 1, 100
    if source_arm.animation_data and source_arm.animation_data.action:
        r = source_arm.animation_data.action.frame_range
        frame_start = int(r[0])
        frame_end   = int(r[1])
    print(f"Baking frames {frame_start}–{frame_end} onto {target_arm.name}")

    bpy.context.view_layer.objects.active = target_arm
    target_arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')

    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        visual_keying=True,      # 拘束込みの見た目通りのポーズを記録
        clear_constraints=True,  # ベイク後に拘束を自動削除
        clear_parents=False,
        use_current_action=True,
        bake_types={'POSE'},
    )

    bpy.ops.object.mode_set(mode='OBJECT')
    print("Bake complete.")


def delete_object_and_data(obj):
    """Delete object and its data block."""
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data and data.users == 0:
        if isinstance(data, bpy.types.Armature):
            bpy.data.armatures.remove(data)


def select_only(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def export_glb(path, arm_obj):
    """Export only the target armature as GLB."""
    print(f"Exporting: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Select only the target armature (and its children for safety)
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    for child in arm_obj.children_recursive:
        child.select_set(True)

    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_animations=True,
        export_anim_single_armature=True,
        export_frame_range=False,
        export_apply=False,
    )


def main():
    input_path, output_path, vrm_path = parse_args()

    if not os.path.exists(input_path):
        print(f"Error: input not found: {input_path}")
        sys.exit(1)

    clear_scene()
    import_fbx(input_path)
    remap_bones()

    source_arm = get_armature()
    if source_arm is None:
        print("Error: no armature found after FBX import")
        sys.exit(1)
    source_arm.name = "SourceArmature"
    print(f"Source armature: {source_arm.name} ({len(source_arm.data.bones)} bones)")

    if vrm_path:
        if not os.path.exists(vrm_path):
            print(f"Error: VRM not found: {vrm_path}")
            sys.exit(1)

        import_vrm(vrm_path)
        target_arm = get_armature(exclude="SourceArmature")
        if target_arm is None:
            print("Error: VRM armature not found after import")
            sys.exit(1)
        print(f"Target armature: {target_arm.name} ({len(target_arm.data.bones)} bones)")

        retarget_with_constraints(source_arm, target_arm)
        bake_to_target(source_arm, target_arm)

        # ソースアーマチュアを削除（ベイク済み）
        delete_object_and_data(source_arm)
        print("Source armature removed.")

        export_glb(output_path, target_arm)
    else:
        # VRM未指定: 従来通りソースをそのままエクスポート
        print("No VRM target specified — exporting source directly.")
        export_glb(output_path, source_arm)

    print(f"Done: {output_path}")


main()
