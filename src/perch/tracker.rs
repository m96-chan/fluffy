/// X11 background thread that polls foreign window positions.
///
/// Uses `_NET_CLIENT_LIST_STACKING` to enumerate windows, then
/// `GetGeometry` + `TranslateCoordinates` for absolute coordinates.
/// Sends `WindowSnapshot` through a `std::sync::mpsc` channel.
use std::sync::mpsc;
use std::time::Duration;

use tracing::{debug, info, warn};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{self, Atom, ConnectionExt, MapState};

/// A foreign (non-Fluffy) window's geometry in root coordinates.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ForeignWindow {
    pub x11_id: u32,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Snapshot of all visible foreign windows at a point in time.
#[derive(Debug, Clone)]
pub struct WindowSnapshot {
    pub windows: Vec<ForeignWindow>,
    pub screen_width: u32,
    pub screen_height: u32,
}

/// Spawn a background thread that polls X11 for foreign windows.
///
/// Returns `None` if the X11 connection fails (e.g. pure Wayland without XWayland).
pub fn spawn_tracker(
    own_title: String,
    poll_interval: Duration,
) -> Option<mpsc::Receiver<WindowSnapshot>> {
    // Test connection first before spawning thread
    let (conn, screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(e) => {
            warn!("Perch: cannot connect to X11: {} — perching disabled", e);
            return None;
        }
    };

    let screen = conn.setup().roots[screen_num].clone();
    let root = screen.root;
    let screen_width = screen.width_in_pixels as u32;
    let screen_height = screen.height_in_pixels as u32;

    // Resolve atoms we need
    let net_client_list = match intern_atom(&conn, b"_NET_CLIENT_LIST_STACKING") {
        Some(a) => a,
        None => {
            warn!("Perch: _NET_CLIENT_LIST_STACKING not available");
            return None;
        }
    };
    let net_wm_name = intern_atom(&conn, b"_NET_WM_NAME");
    let utf8_string = intern_atom(&conn, b"UTF8_STRING");

    drop(conn); // we'll reconnect in the thread

    let (tx, rx) = mpsc::channel();

    std::thread::Builder::new()
        .name("perch-x11-tracker".into())
        .spawn(move || {
            let Ok((conn, _)) = x11rb::connect(None) else {
                warn!("Perch: X11 reconnection failed in tracker thread");
                return;
            };

            info!(
                "Perch: X11 tracker started (poll={}ms, screen={}x{})",
                poll_interval.as_millis(),
                screen_width,
                screen_height
            );

            loop {
                let snapshot = poll_windows(
                    &conn,
                    root,
                    screen_width,
                    screen_height,
                    net_client_list,
                    net_wm_name,
                    utf8_string,
                    &own_title,
                );

                if let Some(snap) = snapshot {
                    if tx.send(snap).is_err() {
                        // Receiver dropped (app shutting down)
                        break;
                    }
                }

                std::thread::sleep(poll_interval);
            }

            info!("Perch: X11 tracker thread exiting");
        })
        .ok()?;

    Some(rx)
}

fn intern_atom(conn: &impl Connection, name: &[u8]) -> Option<Atom> {
    conn.intern_atom(false, name)
        .ok()?
        .reply()
        .ok()
        .map(|r| r.atom)
}

fn poll_windows(
    conn: &impl Connection,
    root: u32,
    screen_width: u32,
    screen_height: u32,
    net_client_list: Atom,
    net_wm_name: Option<Atom>,
    utf8_string: Option<Atom>,
    own_title: &str,
) -> Option<WindowSnapshot> {
    // Get stacking order window list
    let reply = conn
        .get_property(false, root, net_client_list, xproto::AtomEnum::WINDOW, 0, 1024)
        .ok()?
        .reply()
        .ok()?;

    let window_ids: Vec<u32> = reply
        .value32()?
        .collect();

    let mut windows = Vec::new();

    for &wid in &window_ids {
        // Check if window is viewable (not minimized)
        let attrs = match conn.get_window_attributes(wid).ok().and_then(|c| c.reply().ok()) {
            Some(a) => a,
            None => continue,
        };
        if attrs.map_state != MapState::VIEWABLE {
            continue;
        }

        // Get title and skip our own window
        let title = get_window_title(conn, wid, net_wm_name, utf8_string);
        if let Some(ref t) = title {
            if t == own_title {
                continue;
            }
        }

        // Get geometry in root coordinates
        let geom = match conn.get_geometry(wid).ok().and_then(|c| c.reply().ok()) {
            Some(g) => g,
            None => continue,
        };

        let translated = match conn
            .translate_coordinates(wid, root, 0, 0)
            .ok()
            .and_then(|c| c.reply().ok())
        {
            Some(t) => t,
            None => continue,
        };

        windows.push(ForeignWindow {
            x11_id: wid,
            x: translated.dst_x as i32,
            y: translated.dst_y as i32,
            width: geom.width as u32,
            height: geom.height as u32,
        });
    }

    debug!("Perch: polled {} foreign windows", windows.len());

    Some(WindowSnapshot {
        windows,
        screen_width,
        screen_height,
    })
}

fn get_window_title(
    conn: &impl Connection,
    wid: u32,
    net_wm_name: Option<Atom>,
    utf8_string: Option<Atom>,
) -> Option<String> {
    // Try _NET_WM_NAME (UTF-8) first
    if let (Some(atom), Some(utf8)) = (net_wm_name, utf8_string) {
        if let Some(title) = get_string_property(conn, wid, atom, utf8) {
            return Some(title);
        }
    }

    // Fallback to WM_NAME
    get_string_property(conn, wid, xproto::AtomEnum::WM_NAME.into(), xproto::AtomEnum::STRING.into())
}

fn get_string_property(
    conn: &impl Connection,
    wid: u32,
    property: Atom,
    type_: Atom,
) -> Option<String> {
    let reply = conn
        .get_property(false, wid, property, type_, 0, 256)
        .ok()?
        .reply()
        .ok()?;

    if reply.value.is_empty() {
        return None;
    }

    String::from_utf8(reply.value).ok()
}
