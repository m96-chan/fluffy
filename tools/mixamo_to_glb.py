"""
Mixamo FBX → GLB converter with bone name remapping.

Usage:
    blender --background --python tools/mixamo_to_glb.py -- input.fbx output.glb

Renames Mixamo bones (mixamorig:Hips etc.) to MANUKA / VRM 0.x naming
so that the GLB can be loaded and retargeted in Fluffy.
"""

import bpy
import sys
import os

# ── Mixamo → VRM 0.x bone name map ──────────────────────────────────────────
# Left side uses _L suffix, right side uses _R suffix (MANUKA convention)
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

    # Fingers (optional, mapped to generic names)
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
    """Parse arguments after '--'."""
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python mixamo_to_glb.py -- input.fbx output.glb")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    if len(args) < 2:
        print("Error: need input.fbx and output.glb")
        sys.exit(1)
    return args[0], args[1]


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes) + list(bpy.data.armatures) + list(bpy.data.actions):
        bpy.data.batch_remove([block])


def import_fbx(path):
    print(f"Importing: {path}")
    # Blender 5.0 の bpy.ops.import_scene.fbx は CLI から呼ぶと
    # self.files 未初期化バグがあるため、モジュールを直接呼ぶ
    import sys
    sys.path.append('/usr/share/blender/5.0/scripts/addons_core')
    from io_scene_fbx import import_fbx as _fbx_loader

    class _FakeOp:
        """bpy.ops の代わりに直接 load() を呼ぶための最小スタブ。"""
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

        def report(self, level, msg):
            print(f"[{level}] {msg}")

    result = _fbx_loader.load(
        _FakeOp(),
        bpy.context,
        filepath=path,
    )
    print(f"Import result: {result}")


def remap_bones():
    """Rename all Mixamo bones in all armatures and actions."""
    renamed = 0

    # Rename bones in armatures
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE':
            continue
        arm = obj.data
        for bone in arm.bones:
            new_name = BONE_MAP.get(bone.name)
            if new_name:
                print(f"  bone: {bone.name} → {new_name}")
                bone.name = new_name
                renamed += 1

    # Rename bone channels in all animation actions
    # Blender 5.0: Layered Action API (action.fcurves は廃止)
    # strip.channelbags → channelbag.fcurves でアクセス
    for action in bpy.data.actions:
        fcurves = []
        if action.is_action_layered:
            for layer in action.layers:
                for strip in layer.strips:
                    if hasattr(strip, 'channelbags'):
                        for cb in strip.channelbags:
                            fcurves.extend(cb.fcurves)
        else:
            # 旧形式フォールバック
            fcurves = list(action.fcurves)

        for fcurve in fcurves:
            dp = fcurve.data_path
            for old, new in BONE_MAP.items():
                if f'"{old}"' in dp:
                    fcurve.data_path = dp.replace(f'"{old}"', f'"{new}"')
                    break

    print(f"Renamed {renamed} bones.")


def apply_tpose_as_rest():
    """
    Mixamo A-pose → T-pose に変換してレストポーズとして適用する。

    手順:
      1. 全フレームをビジュアルキーフレームにベイク（絶対座標で保存）
      2. 全ボーン回転クリア（= T-pose）
      3. T-pose をレストポーズとして適用

    ベイク後のカーブは絶対座標なのでレストポーズ変更の影響を受けない。
    """
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE':
            continue

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # フレーム範囲を取得
        frame_start, frame_end = 1, 100
        if obj.animation_data and obj.animation_data.action:
            r = obj.animation_data.action.frame_range
            frame_start = int(r[0])
            frame_end   = int(r[1])

        print(f"  Baking {obj.name}: frame {frame_start}–{frame_end}")

        # ① ビジュアルキーフレームにベイク
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            visual_keying=True,
            clear_constraints=False,
            clear_parents=False,
            use_current_action=True,
            bake_types={'POSE'},
        )

        # ② frame_start に移動して T-pose（回転ゼロ）に設定
        bpy.context.scene.frame_set(frame_start)
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.rot_clear()

        # ③ T-pose をレストポーズとして適用
        bpy.ops.pose.armature_apply()

        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  T-pose rest pose applied to: {obj.name}")


def export_glb(path):
    print(f"Exporting: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        export_animations=True,
        export_anim_single_armature=True,
        export_frame_range=False,        # export all frames
        export_apply=False,
    )


def main():
    input_path, output_path = parse_args()

    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    clear_scene()
    import_fbx(input_path)
    remap_bones()
    # apply_tpose_as_rest()  # Blender 5.0ではカーブが正しく更新されないため無効
    # TODO: 適切なリターゲット（issue #5）
    export_glb(output_path)
    print(f"Done: {output_path}")


main()
