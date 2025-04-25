#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]


mod engine;
mod model;

use anyhow::Result;
use engine::setup::ApplicationEngine;
use winit::event_loop::EventLoop;

fn main() -> Result<()> {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = ApplicationEngine::new(&event_loop);
    event_loop.run_app(&mut app)?;

    Ok(())
}