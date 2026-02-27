/// Bevy UI plugin for the chat panel.
/// Layout: left 400px = character (3D viewport), right 350px = chat panel.
/// Window is fixed at 750×600. T key toggles chat panel visibility.

use bevy::prelude::*;

use crate::chat::state::{ChatState, MessageRole, MASCOT_WIDTH};
use crate::events::PipelineMessage;

pub struct ChatOverlayPlugin;

impl Plugin for ChatOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ChatState::default())
            .add_systems(Startup, setup_ui)
            .add_systems(
                Update,
                (handle_toggle_key, handle_pipeline_messages, update_chat_ui).chain(),
            );
    }
}

#[derive(Component)]
struct ChatPanel;

#[derive(Component)]
struct MessageList;

#[derive(Component)]
struct MessageItem(usize);

fn setup_ui(mut commands: Commands) {
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            ..default()
        })
        .with_children(|root| {
            // Left spacer — 3D camera renders here
            root.spawn(Node {
                width: Val::Px(MASCOT_WIDTH as f32),
                height: Val::Percent(100.0),
                flex_shrink: 0.0,
                ..default()
            });

            // Right: chat panel
            root.spawn((
                ChatPanel,
                Node {
                    flex_direction: FlexDirection::Column,
                    flex_grow: 1.0,
                    height: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(10.0)),
                    overflow: Overflow::clip_y(),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.04, 0.04, 0.10, 0.88)),
                Visibility::Hidden,
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("Chat  [ T to close ]"),
                    TextFont { font_size: 10.0, ..default() },
                    TextColor(Color::srgba(0.5, 0.5, 0.55, 1.0)),
                    Node { margin: UiRect::bottom(Val::Px(6.0)), ..default() },
                ));

                panel.spawn((
                    MessageList,
                    Node {
                        flex_direction: FlexDirection::Column,
                        flex_grow: 1.0,
                        overflow: Overflow::clip_y(),
                        row_gap: Val::Px(6.0),
                        ..default()
                    },
                ));
            });
        });
}

fn handle_toggle_key(keys: Res<ButtonInput<KeyCode>>, mut chat: ResMut<ChatState>) {
    if keys.just_pressed(KeyCode::KeyT) {
        chat.toggle();
    }
}

fn update_chat_ui(
    chat: Res<ChatState>,
    mut panel_q: Query<&mut Visibility, With<ChatPanel>>,
    list_q: Query<Entity, With<MessageList>>,
    mut commands: Commands,
) {
    if !chat.is_changed() {
        return;
    }

    for mut vis in panel_q.iter_mut() {
        *vis = if chat.visible { Visibility::Visible } else { Visibility::Hidden };
    }

    if !chat.visible {
        return;
    }

    let Ok(list_entity) = list_q.single() else { return };
    commands.entity(list_entity).despawn_related::<Children>();

    commands.entity(list_entity).with_children(|list| {
        for (i, msg) in chat.messages.iter().enumerate() {
            let (label, label_color, text_color, bg) = match msg.role {
                MessageRole::User => (
                    "You",
                    Color::srgb(0.5, 0.8, 1.0),
                    Color::WHITE,
                    Color::srgba(0.12, 0.18, 0.28, 0.75),
                ),
                MessageRole::Assistant => (
                    if msg.is_streaming { "Fluffy ●" } else { "Fluffy" },
                    Color::srgb(0.75, 0.55, 1.0),
                    Color::srgb(0.92, 0.92, 0.92),
                    Color::srgba(0.08, 0.08, 0.16, 0.75),
                ),
            };

            list.spawn((
                MessageItem(i),
                Node {
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(6.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(bg),
            ))
            .with_children(|bubble| {
                bubble.spawn((
                    Text::new(label),
                    TextFont { font_size: 9.0, ..default() },
                    TextColor(label_color),
                    Node { margin: UiRect::bottom(Val::Px(2.0)), ..default() },
                ));
                bubble.spawn((
                    Text::new(msg.text.clone()),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(text_color),
                ));
            });
        }
    });
}

fn handle_pipeline_messages(
    mut reader: MessageReader<PipelineMessage>,
    mut chat: ResMut<ChatState>,
) {
    for msg in reader.read() {
        match msg {
            PipelineMessage::SttResult { text } => {
                chat.push_user(text.clone());
            }
            PipelineMessage::LlmToken { token } => {
                let needs_new = chat
                    .messages
                    .last()
                    .map(|m| m.role != MessageRole::Assistant || !m.is_streaming)
                    .unwrap_or(true);
                if needs_new {
                    chat.start_assistant_message();
                }
                chat.append_token(token);
            }
            PipelineMessage::LlmDone => {
                chat.finish_assistant_message();
            }
            _ => {}
        }
    }
}
