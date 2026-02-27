import bpy
import os
import sys
from mathutils import Matrix, Quaternion

def build_bone_map_for_target(target_arm):
    names = {b.name for b in target_arm.data.bones}
    use_underscore = "Shoulder_L" in names or "UpperArm_L" in names
    side_sep = "_" if use_underscore else "."

    def side(name, lr):
        return f"{name}{side_sep}{lr}"

    return {
        "mixamorig:Hips": "Hips",
        "mixamorig:Spine": "Spine",
        "mixamorig:Spine1": "Spine",
        "mixamorig:Spine2": "Chest",
        "mixamorig:Neck": "Neck",
        "mixamorig:Head": "Head",
        "mixamorig:LeftShoulder": side("Shoulder", "L"),
        "mixamorig:LeftArm": side("UpperArm", "L"),
        "mixamorig:LeftForeArm": side("LowerArm", "L"),
        "mixamorig:LeftHand": side("Hand", "L"),
        "mixamorig:RightShoulder": side("Shoulder", "R"),
        "mixamorig:RightArm": side("UpperArm", "R"),
        "mixamorig:RightForeArm": side("LowerArm", "R"),
        "mixamorig:RightHand": side("Hand", "R"),
        "mixamorig:LeftUpLeg": side("UpperLeg", "L"),
        "mixamorig:LeftLeg": side("LowerLeg", "L"),
        "mixamorig:LeftFoot": side("Foot", "L"),
        "mixamorig:LeftToeBase": side("Toe", "L"),
        "mixamorig:RightUpLeg": side("UpperLeg", "R"),
        "mixamorig:RightLeg": side("LowerLeg", "R"),
        "mixamorig:RightFoot": side("Foot", "R"),
        "mixamorig:RightToeBase": side("Toe", "R"),
        "mixamorig:LeftHandThumb1": side("ThumbProximal", "L"),
        "mixamorig:LeftHandThumb2": side("ThumbIntermediate", "L"),
        "mixamorig:LeftHandThumb3": side("ThumbDistal", "L"),
        "mixamorig:LeftHandIndex1": side("IndexProximal", "L"),
        "mixamorig:LeftHandIndex2": side("IndexIntermediate", "L"),
        "mixamorig:LeftHandIndex3": side("IndexDistal", "L"),
        "mixamorig:LeftHandMiddle1": side("MiddleProximal", "L"),
        "mixamorig:LeftHandMiddle2": side("MiddleIntermediate", "L"),
        "mixamorig:LeftHandMiddle3": side("MiddleDistal", "L"),
        "mixamorig:LeftHandRing1": side("RingProximal", "L"),
        "mixamorig:LeftHandRing2": side("RingIntermediate", "L"),
        "mixamorig:LeftHandRing3": side("RingDistal", "L"),
        "mixamorig:LeftHandPinky1": side("LittleProximal", "L"),
        "mixamorig:LeftHandPinky2": side("LittleIntermediate", "L"),
        "mixamorig:LeftHandPinky3": side("LittleDistal", "L"),
        "mixamorig:RightHandThumb1": side("ThumbProximal", "R"),
        "mixamorig:RightHandThumb2": side("ThumbIntermediate", "R"),
        "mixamorig:RightHandThumb3": side("ThumbDistal", "R"),
        "mixamorig:RightHandIndex1": side("IndexProximal", "R"),
        "mixamorig:RightHandIndex2": side("IndexIntermediate", "R"),
        "mixamorig:RightHandIndex3": side("IndexDistal", "R"),
        "mixamorig:RightHandMiddle1": side("MiddleProximal", "R"),
        "mixamorig:RightHandMiddle2": side("MiddleIntermediate", "R"),
        "mixamorig:RightHandMiddle3": side("MiddleDistal", "R"),
        "mixamorig:RightHandRing1": side("RingProximal", "R"),
        "mixamorig:RightHandRing2": side("RingIntermediate", "R"),
        "mixamorig:RightHandRing3": side("RingDistal", "R"),
        "mixamorig:RightHandPinky1": side("LittleProximal", "R"),
        "mixamorig:RightHandPinky2": side("LittleIntermediate", "R"),
        "mixamorig:RightHandPinky3": side("LittleDistal", "R"),
    }


def parse_args():
    if "--" not in sys.argv:
        print("Usage: blender --background --python tools/build_chocolat_mascot_glb.py -- chocolat.fbx idle.fbx greeting.fbx out.glb")
        sys.exit(1)
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 4:
        print("Error: need chocolat.fbx idle.fbx greeting.fbx out.glb")
        sys.exit(1)
    return tuple(os.path.abspath(a) for a in args)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


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


def find_target_armature():
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        return None
    return max(arms, key=lambda a: len(a.data.bones))


def find_new_source_armature(target_arm):
    cands = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.name != target_arm.name]
    if not cands:
        return None
    return min(cands, key=lambda a: len(a.data.bones))


def remap_mixamo_bones(source_arm, bone_map):
    for bone in source_arm.data.bones:
        new_name = bone_map.get(bone.name)
        if new_name:
            bone.name = new_name


def relink_chocolat_textures(chocolat_fbx_path):
    base_dir = os.path.dirname(os.path.dirname(chocolat_fbx_path))
    tex_dir = os.path.join(base_dir, "Texture")
    if not os.path.isdir(tex_dir):
        return
    remap = {
        "face": "Chocolat_Face.png",
        "face_effect": "Chocolat_Face_effect.png",
        "hair": "Chocolat_Hair.png",
        "tex": "Chocolat_Costume.png",
        "costume": "Chocolat_Costume.png",
        "body": "Chocolat_Body.png",
        "chiffon_body": "Chocolat_Body.png",
    }
    fixed = 0
    for img in bpy.data.images:
        if img.name == "Render Result":
            continue
        src = bpy.path.abspath(img.filepath)
        stem = os.path.splitext(os.path.basename(src))[0].lower()
        ext = os.path.splitext(src)[1].lower()
        if os.path.exists(src) and ext != ".psd":
            continue
        target = None
        if stem in remap:
            target = remap[stem]
        else:
            for k, v in remap.items():
                if k in stem:
                    target = v
                    break
        if not target:
            continue
        p = os.path.join(tex_dir, target)
        if not os.path.exists(p):
            continue
        img.filepath = p
        try:
            img.reload()
        except Exception:
            pass
        fixed += 1
    print(f"Relinked textures: {fixed}")


def retarget_action(target_arm, source_arm, action_name):
    frame_start, frame_end = 1, 100
    if source_arm.animation_data and source_arm.animation_data.action:
        r = source_arm.animation_data.action.frame_range
        frame_start, frame_end = int(r[0]), int(r[1])

    src = {b.name for b in source_arm.data.bones}
    dst = {b.name for b in target_arm.data.bones}
    shared = sorted(src & dst)
    print(f"Shared bones: {len(shared)}")

    def depth(name):
        d = 0
        b = target_arm.data.bones[name]
        while b.parent is not None:
            d += 1
            b = b.parent
        return d

    ordered = sorted(shared, key=depth)
    src_rest_world = {name: source_arm.data.bones[name].matrix_local.copy() for name in shared}
    dst_rest_world = {name: target_arm.data.bones[name].matrix_local.copy() for name in shared}
    dst_rest_local = {}
    for name in shared:
        tb = target_arm.data.bones[name]
        if tb.parent is None:
            dst_rest_local[name] = tb.matrix_local.copy()
        else:
            dst_rest_local[name] = tb.parent.matrix_local.inverted() @ tb.matrix_local
    adaptive_bones = {
        "UpperArm.L",
        "LowerArm.L",
        "Hand.L",
        "UpperArm.R",
        "LowerArm.R",
        "Hand.R",
        "UpperLeg.L",
        "LowerLeg.L",
        "UpperLeg.R",
        "LowerLeg.R",
    }
    finger_prefixes = (
        "ThumbProximal.",
        "ThumbIntermediate.",
        "ThumbDistal.",
        "IndexProximal.",
        "IndexIntermediate.",
        "IndexDistal.",
        "MiddleProximal.",
        "MiddleIntermediate.",
        "MiddleDistal.",
        "RingProximal.",
        "RingIntermediate.",
        "RingDistal.",
        "LittleProximal.",
        "LittleIntermediate.",
        "LittleDistal.",
    )
    finger_weight = 0.45
    world_corr = {}
    target_arm.animation_data_create()
    target_arm.animation_data.action = bpy.data.actions.new(action_name)
    target_action = target_arm.animation_data.action

    bpy.context.view_layer.objects.active = target_arm
    target_arm.select_set(True)
    source_arm.select_set(False)
    bpy.ops.object.mode_set(mode="POSE")
    scene = bpy.context.scene
    src_hips_base = None
    dst_hips_base = None
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        computed_world = {}
        for name in ordered:
            src_pb = source_arm.pose.bones.get(name)
            dst_pb = target_arm.pose.bones.get(name)
            if src_pb is None or dst_pb is None:
                continue

            # Transfer source rest->pose delta in armature space to target rest.
            src_delta = src_rest_world[name].inverted() @ src_pb.matrix
            dst_pose_world = dst_rest_world[name] @ src_delta
            dloc, drot, dscale = dst_pose_world.decompose()
            if name in adaptive_bones:
                if frame == frame_start and name not in world_corr:
                    srot = src_pb.matrix.to_quaternion().normalized()
                    world_corr[name] = (srot @ drot.inverted()).normalized()
                corr = world_corr.get(name)
                if corr is not None:
                    drot = (corr @ drot).normalized()
                    dst_pose_world = (
                        Matrix.Translation(dloc)
                        @ drot.to_matrix().to_4x4()
                        @ Matrix.Diagonal((dscale.x, dscale.y, dscale.z, 1.0))
                    )
            parent = target_arm.data.bones[name].parent
            if parent and parent.name in computed_world:
                dst_pose_local = computed_world[parent.name].inverted() @ dst_pose_world
            else:
                dst_pose_local = dst_pose_world
            dst_basis = dst_rest_local[name].inverted() @ dst_pose_local
            loc, rot, _ = dst_basis.decompose()
            if name.startswith(finger_prefixes):
                rot = Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(rot.normalized(), finger_weight)
            dst_pb.rotation_mode = "QUATERNION"
            dst_pb.rotation_quaternion = rot.normalized()
            if name == "Hips":
                if src_hips_base is None:
                    src_hips_base = src_pb.location.copy()
                    dst_hips_base = dst_pb.location.copy()
                dst_pb.location = dst_hips_base + (src_pb.location - src_hips_base)
            else:
                dst_pb.location = (0.0, 0.0, 0.0)
            computed_world[name] = dst_pose_world

            if name == "Hips":
                dst_pb.keyframe_insert(data_path="location", frame=frame)
            dst_pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")
    target_action.name = action_name


def delete_object_and_data(obj):
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data and data.users == 0 and isinstance(data, bpy.types.Armature):
        bpy.data.armatures.remove(data)


def prune_shape_keys(max_non_basis=500):
    expr_prefixes = (
        "vrc.",
        "eye_",
        "eyelid_",
        "lower_eyelid_",
        "brow_",
        "mouth_",
        "lip_",
        "jaw_",
        "tongue_",
        "cheek_",
        "nose_",
    )
    corrective_tokens = (
        "_OFF",
        "Skirt_",
        "Socks_",
        "Loafer_",
        "Bra_",
        "Shirt_",
        "SailorCollar_",
        "Foot_HighHeel",
        "Waist_slim",
        "Nail_point",
        "SharkSpine",
        "SharkChest",
    )
    keep_set = set()
    all_keys = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data.shape_keys:
            continue
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name == "Basis":
                continue
            all_keys.append((obj.name, kb.name))

    # Priority 1: facial/viseme expression keys (mainly Body).
    for obj_name, key_name in all_keys:
        n = key_name.lower()
        if any(n.startswith(p) for p in expr_prefixes):
            keep_set.add((obj_name, key_name))
    # Priority 2: clothing/body corrective keys.
    for obj_name, key_name in all_keys:
        if (obj_name, key_name) in keep_set:
            continue
        if any(tok in key_name for tok in corrective_tokens):
            keep_set.add((obj_name, key_name))

    # Enforce Bevy morph target limit with deterministic order.
    if len(keep_set) > max_non_basis:
        trimmed = set()
        for pair in all_keys:
            if pair in keep_set:
                trimmed.add(pair)
                if len(trimmed) >= max_non_basis:
                    break
        keep_set = trimmed

    removed = 0
    kept = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data.shape_keys:
            continue
        keep_names = {
            n for o, n in keep_set if o == obj.name
        } | {"Basis"}
        kb_names = [kb.name for kb in obj.data.shape_keys.key_blocks]
        remove_names = [n for n in kb_names if n not in keep_names]
        kept += len(kb_names) - len(remove_names)
        if not remove_names:
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for name in remove_names:
            idx = obj.data.shape_keys.key_blocks.find(name)
            if idx < 0:
                continue
            obj.active_shape_key_index = idx
            bpy.ops.object.shape_key_remove(all=False)
            removed += 1
        obj.select_set(False)
    print(f"Shape keys kept(non-basis): {len(keep_set)} removed: {removed} total_kept: {kept}")


def apply_default_shape_key_presets():
    # Prevent torso/waist cloth penetration in idle and greeting poses.
    presets = (
        ("Body_base", "Spine_1_OFF", 1.0),
        ("Body_base", "Spine_2_OFF", 1.0),
        ("Body_base", "Chest_1_OFF", 1.0),
        ("Body_base", "Chest_2_OFF", 1.0),
        ("Body_base", "Waist_slim", 1.0),
        ("Skirt", "Skirt_Corset_OFF", 1.0),
    )
    total = 0
    for obj_name, key_name, value in presets:
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != "MESH" or not obj.data.shape_keys:
            continue
        idx = obj.data.shape_keys.key_blocks.find(key_name)
        if idx < 0:
            continue
        obj.data.shape_keys.key_blocks[idx].value = value
        total += 1

    print(f"Shape key presets applied: {total}/{len(presets)}")


def keep_actions(names):
    allow = set(names)
    for action in list(bpy.data.actions):
        if action.name not in allow:
            bpy.data.actions.remove(action)


def export_glb(path, target_arm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    for c in target_arm.children_recursive:
        c.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_animations=True,
        use_selection=True,
        export_skins=True,
        export_def_bones=True,
    )


def bake_from_source_fbx(target_arm, source_fbx, action_name):
    bone_map = build_bone_map_for_target(target_arm)
    import_fbx(source_fbx)
    source_arm = find_new_source_armature(target_arm)
    if source_arm is None:
        print(f"Error: source armature not found for {source_fbx}")
        sys.exit(1)
    remap_mixamo_bones(source_arm, bone_map)
    retarget_action(target_arm, source_arm, action_name)
    delete_object_and_data(source_arm)


def main():
    chocolat_fbx, idle_fbx, greeting_fbx, out_glb = parse_args()
    for p in (chocolat_fbx, idle_fbx, greeting_fbx):
        if not os.path.exists(p):
            print(f"Error: not found: {p}")
            sys.exit(1)

    clear_scene()
    import_fbx(chocolat_fbx)
    target_arm = find_target_armature()
    if target_arm is None:
        print("Error: target armature not found in chocolat")
        sys.exit(1)

    relink_chocolat_textures(chocolat_fbx)
    bake_from_source_fbx(target_arm, idle_fbx, "Idle")
    bake_from_source_fbx(target_arm, greeting_fbx, "Greeting")

    keep_actions(["Idle", "Greeting"])
    apply_default_shape_key_presets()
    prune_shape_keys()
    export_glb(out_glb, target_arm)
    print(f"Done: {out_glb}")


main()
