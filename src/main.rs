#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

mod engine;
mod game;
mod model;

use std::sync::Arc;

use anyhow::Result;
use engine::setup::ApplicationEngine;
use game::world::World;
use winit::event_loop::EventLoop;

fn main() -> Result<()> {
    let world = Arc::new(World::new());

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = ApplicationEngine::new(&event_loop, world);
    event_loop.run_app(&mut app)?;

    Ok(())
}
