import bpy
import os
import sys


def parse_args():
    if "--" not in sys.argv:
        print("Usage: blender --background --python tools/fbx_to_glb.py -- input.fbx output.glb")
        sys.exit(1)
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        print("Error: need input.fbx output.glb")
        sys.exit(1)
    return os.path.abspath(args[0]), os.path.abspath(args[1])


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


def remove_all_shape_keys():
    removed = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.data.shape_keys:
            continue
        key_blocks = obj.data.shape_keys.key_blocks
        removed += len(key_blocks)
        with bpy.context.temp_override(
            object=obj,
            active_object=obj,
            selected_objects=[obj],
            selected_editable_objects=[obj],
        ):
            bpy.ops.object.shape_key_remove(all=True)
    print(f"Removed shape keys: {removed}")


def relink_chocolat_textures(input_fbx_path):
    base_dir = os.path.dirname(os.path.dirname(input_fbx_path))
    tex_dir = os.path.join(base_dir, "Texture")
    if not os.path.isdir(tex_dir):
        print(f"Texture directory not found: {tex_dir}")
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
        needs_relink = (not os.path.exists(src)) or ext == ".psd"
        if not needs_relink:
            continue

        target_name = None
        if stem in remap:
            target_name = remap[stem]
        else:
            # fallback for noisy source names
            for key, val in remap.items():
                if key in stem:
                    target_name = val
                    break
        if not target_name:
            continue

        target_path = os.path.join(tex_dir, target_name)
        if not os.path.exists(target_path):
            continue
        img.filepath = target_path
        try:
            img.reload()
        except Exception:
            pass
        fixed += 1
    print(f"Relinked textures: {fixed}")


def main():
    input_path, output_path = parse_args()
    if not os.path.exists(input_path):
        print(f"Error: input not found: {input_path}")
        sys.exit(1)

    clear_scene()
    import_fbx(input_path)
    relink_chocolat_textures(input_path)
    remove_all_shape_keys()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_animations=True,
    )
    if not os.path.exists(output_path):
        print(f"Error: export did not create file: {output_path}")
        sys.exit(1)
    print(f"Done: {output_path}")


main()
