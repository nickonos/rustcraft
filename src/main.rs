#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use anyhow::Result;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::library;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Minecraft, Also try Terraria!")
            .with_inner_size(LogicalSize::new(800, 600));

        self.window = Some(ActiveEventLoop::create_window(event_loop, window_attributes).unwrap());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed {:?}", id);
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::default();

    event_loop.run_app(&mut app)?;

    // Create VKInstance
    let library = library::VulkanLibrary::new().expect("No local Vulkan library found");

    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        ..Default::default()
    };

    let instance = Instance::new(library, instance_create_info).expect("Failed to create instance");

    // Select Physical Device
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices")
        .next()
        .expect("Could not find a physical device");

    // Get Queue Family Index
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|properties| properties.queue_flags.contains(QueueFlags::GRAPHICS))
        .expect("Could not find a queue family that supports graphics")
        as u32;

    // Create Logical Device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Failed to create logical device");

    let queue = queues.next().unwrap();

    // Create Window Surface

    // Create Swapchain

    // Create Image Views

    // Create Framebuffers

    // Create Render Pass

    // Create Graphics Pipeline

    // Create Command Pool

    // Create Command Buffers

    // Create Main loop

    Ok(())
}
