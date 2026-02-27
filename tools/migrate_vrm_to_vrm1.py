"""
Backup + migrate VRM to VRM 1.0 using Blender VRM Add-on.

Usage:
    blender --background --python tools/migrate_vrm_to_vrm1.py -- input.vrm output.vrm [backup.vrm]
"""

import os
import shutil
import sys
from datetime import datetime

import bpy


def parse_args():
    if "--" not in sys.argv:
        print("Usage: blender --background --python migrate_vrm_to_vrm1.py -- input.vrm output.vrm [backup.vrm]")
        sys.exit(1)
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) < 2 or len(args) > 3:
        print("Error: need input.vrm output.vrm [backup.vrm]")
        sys.exit(1)
    input_path = os.path.abspath(args[0])
    output_path = os.path.abspath(args[1])
    if len(args) == 3:
        backup_path = os.path.abspath(args[2])
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        root, ext = os.path.splitext(input_path)
        backup_path = f"{root}.backup-{ts}{ext}"
    return input_path, output_path, backup_path


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes) + list(bpy.data.armatures) + list(bpy.data.actions):
        bpy.data.batch_remove([block])


def find_armature():
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def migrate_to_vrm1(armature_obj):
    arm_data = armature_obj.data
    ext = arm_data.vrm_addon_extension

    # Switch spec version to VRM 1.0
    ext.spec_version = ext.SPEC_VERSION_VRM1

    # Auto assign VRM 1.0 required human bones
    result = bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(
        armature_object_name=armature_obj.name
    )
    print(f"assign_vrm1_humanoid_human_bones_automatically: {result}")

    ext = arm_data.vrm_addon_extension
    if not ext.is_vrm1():
        print("Error: failed to switch spec_version to VRM 1.0")
        sys.exit(1)

    human_bones = ext.vrm1.humanoid.human_bones
    if not human_bones.all_required_bones_are_assigned():
        print("Error: required VRM 1.0 human bones are not fully assigned")
        sys.exit(1)


def export_vrm(output_path, armature_obj):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = bpy.ops.export_scene.vrm(
        filepath=output_path,
        armature_object_name=armature_obj.name,
        ignore_warning=True,
    )
    print(f"export_scene.vrm: {result}")
    if not os.path.exists(output_path):
        print(f"Error: export did not create file: {output_path}")
        sys.exit(1)


def main():
    input_path, output_path, backup_path = parse_args()
    if not os.path.exists(input_path):
        print(f"Error: input not found: {input_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(input_path, backup_path)
    print(f"Backup created: {backup_path}")

    clear_scene()
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.vrm")
    bpy.ops.import_scene.vrm(filepath=input_path)

    armature = find_armature()
    if armature is None:
        print("Error: armature not found after VRM import")
        sys.exit(1)

    migrate_to_vrm1(armature)
    export_vrm(output_path, armature)
    print(f"Done: {output_path}")


main()
