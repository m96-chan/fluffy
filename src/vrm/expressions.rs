/// VRM expression (blend shape) map.
///
/// Supports:
/// - VRM 0.x: extensions.VRM.blendShapeMaster.blendShapeGroups
/// - VRM 1.x: extensions.VRMC_vrm.expressions.preset

use std::collections::HashMap;
use bevy::prelude::*;
use bevy::gltf::Gltf;

/// A single morph target bind: which mesh entity and which morph index.
#[derive(Debug, Clone)]
pub struct ExpressionBind {
    /// Node name as it appears in the spawned Bevy scene.
    pub node_name: String,
    /// Morph target index within that mesh.
    pub morph_index: usize,
}

/// Maps expression preset names → list of morph target binds.
/// Attached as a Component on the mascot entity after the VRM is parsed.
#[derive(Component, Debug, Default)]
pub struct VrmExpressionMap {
    pub binds: HashMap<String, Vec<ExpressionBind>>,
}

/// Parse the VRM expression map from a loaded Gltf asset.
/// Tries VRM 1.x first, then falls back to VRM 0.x.
pub fn parse_expressions(gltf: &Gltf) -> VrmExpressionMap {
    let Some(doc) = gltf.source.as_ref() else {
        warn!("VRM: no source document available");
        return VrmExpressionMap::default();
    };

    // Build mesh-index → node-name mapping from the document
    let node_by_mesh_index = build_node_by_mesh_index(doc);

    // Serialize the gltf::json::Root to serde_json::Value to access raw extension data.
    // gltf crate's Document::as_json() returns a typed struct, not a raw Value.
    let raw: serde_json::Value = match serde_json::to_value(doc.as_json()) {
        Ok(v) => v,
        Err(e) => {
            warn!("VRM: failed to serialize GLTF JSON: {}", e);
            return VrmExpressionMap::default();
        }
    };
    let extensions = match raw.get("extensions") {
        Some(e) => e,
        None => {
            warn!("VRM: no extensions in GLTF document");
            return VrmExpressionMap::default();
        }
    };

    if let Some(vrmc) = extensions.get("VRMC_vrm") {
        info!("VRM: detected VRM 1.x format");
        parse_vrm1(vrmc, doc)
    } else if let Some(vrm0) = extensions.get("VRM") {
        info!("VRM: detected VRM 0.x format");
        parse_vrm0(vrm0, &node_by_mesh_index)
    } else {
        warn!("VRM: neither VRMC_vrm nor VRM extension found");
        VrmExpressionMap::default()
    }
}

/// VRM 1.x: extensions.VRMC_vrm.expressions.preset
fn parse_vrm1(vrmc: &serde_json::Value, doc: &gltf::Document) -> VrmExpressionMap {
    let mut map = VrmExpressionMap::default();

    let Some(presets) = vrmc
        .get("expressions")
        .and_then(|e| e.get("preset"))
        .and_then(|p| p.as_object())
    else {
        warn!("VRM 1.x: no expressions.preset found");
        return map;
    };

    let nodes: Vec<_> = doc.nodes().collect();

    for (preset_name, preset) in presets {
        let Some(binds_arr) = preset
            .get("morphTargetBinds")
            .and_then(|b| b.as_array())
        else {
            continue;
        };

        let mut expr_binds = Vec::new();
        for bind in binds_arr {
            let Some(node_idx) = bind.get("node").and_then(|v| v.as_u64()) else {
                continue;
            };
            let Some(morph_index) = bind.get("index").and_then(|v| v.as_u64()) else {
                continue;
            };
            if let Some(node) = nodes.get(node_idx as usize) {
                if let Some(name) = node.name() {
                    expr_binds.push(ExpressionBind {
                        node_name: name.to_string(),
                        morph_index: morph_index as usize,
                    });
                }
            }
        }

        if !expr_binds.is_empty() {
            map.binds.insert(preset_name.clone(), expr_binds);
        }
    }

    info!("VRM 1.x: parsed {} expressions", map.binds.len());
    map
}

/// VRM 0.x: extensions.VRM.blendShapeMaster.blendShapeGroups
fn parse_vrm0(
    vrm: &serde_json::Value,
    node_by_mesh_index: &HashMap<usize, String>,
) -> VrmExpressionMap {
    let mut map = VrmExpressionMap::default();

    let Some(groups) = vrm
        .get("blendShapeMaster")
        .and_then(|b| b.get("blendShapeGroups"))
        .and_then(|g| g.as_array())
    else {
        warn!("VRM 0.x: no blendShapeMaster.blendShapeGroups found");
        return map;
    };

    for group in groups {
        // Use presetName if available, otherwise fallback to name
        let preset_name = group
            .get("presetName")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty() && *s != "unknown")
            .unwrap_or_else(|| group.get("name").and_then(|v| v.as_str()).unwrap_or(""))
            .to_lowercase();

        if preset_name.is_empty() {
            continue;
        }

        let Some(binds_arr) = group.get("binds").and_then(|b| b.as_array()) else {
            continue;
        };

        let mut expr_binds = Vec::new();
        for bind in binds_arr {
            let Some(mesh_idx) = bind.get("mesh").and_then(|v| v.as_u64()) else {
                continue;
            };
            let Some(morph_index) = bind.get("index").and_then(|v| v.as_u64()) else {
                continue;
            };
            // VRM 0.x weight is 0-100, we ignore it (we'll set 0.0 or 1.0)
            if let Some(node_name) = node_by_mesh_index.get(&(mesh_idx as usize)) {
                expr_binds.push(ExpressionBind {
                    node_name: node_name.clone(),
                    morph_index: morph_index as usize,
                });
            }
        }

        if !expr_binds.is_empty() {
            map.binds.insert(preset_name, expr_binds);
        }
    }

    info!("VRM 0.x: parsed {} expressions", map.binds.len());
    map
}

/// Build a map from mesh index → node name.
/// In GLTF, nodes reference meshes; we invert this mapping.
fn build_node_by_mesh_index(doc: &gltf::Document) -> HashMap<usize, String> {
    let mut map = HashMap::new();
    for node in doc.nodes() {
        if let Some(mesh) = node.mesh() {
            if let Some(name) = node.name() {
                map.insert(mesh.index(), name.to_string());
            }
        }
    }
    map
}
