/// Bevy plugin that relays async pipeline messages into Bevy's message system.

use bevy::prelude::*;

use crate::events::PipelineMessage;
use crate::state::PipelineState;

pub struct PipelineRelayPlugin;

impl Plugin for PipelineRelayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, relay_pipeline_messages);
    }
}

/// Each frame, drain messages from the async pipeline channel and forward them
/// as Bevy messages so that mascot systems can react.
fn relay_pipeline_messages(
    pipeline: Res<PipelineState>,
    mut writer: MessageWriter<PipelineMessage>,
) {
    let Some(receiver) = &pipeline.receiver else {
        return;
    };

    // Try to acquire the mutex without blocking (pipeline runs on a different thread).
    let Ok(mut rx) = receiver.try_lock() else {
        return;
    };

    // Drain all pending messages.
    loop {
        match rx.try_recv() {
            Ok(msg) => { writer.write(msg); }
            Err(_) => break,
        }
    }
}
