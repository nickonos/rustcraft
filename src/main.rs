#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use std::sync::Arc;

use anyhow::Result;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::{library, swapchain};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
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

    // Create VKInstance
    let library = library::VulkanLibrary::new().expect("No local Vulkan library found");
    let required_extensions =
        Surface::required_extensions(&event_loop).expect("unable to get required extensions");

    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions: required_extensions,
        ..Default::default()
    };

    let instance = Instance::new(library, instance_create_info).expect("Failed to create instance");

    // Make sure physical device has swapchain support
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // Create window
    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    let window = app.window.expect("window was not created");

    let dimenisons = window.inner_size();

    // Create Window Surface
    let surface = Surface::from_window(instance.clone(), Arc::new(window))
        .expect("Unable to create surface from window");

    // Select Physical Device
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("Could not find a physical device");

    

    let capabilities = physical_device.surface_capabilities(&surface, Default::default()).expect("could not get physical device capabilities");
    let composite_alpha = capabilities.supported_composite_alpha.into_iter().next().unwrap();
    let image_format = physical_device.surface_formats(&surface, Default::default()).unwrap()[0].0;

    // Create Logical Device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .expect("Failed to create logical device");

    let queue = queues.next().unwrap();

    // Create Swapchain
    let (mut swapchain, images) = Swapchain::new(device.clone(), surface.clone(), SwapchainCreateInfo {
        min_image_count: capabilities.min_image_count + 1,
        image_format: image_format,
        image_extent: dimenisons.into(),
        image_usage: ImageUsage::COLOR_ATTACHMENT,
        composite_alpha: composite_alpha,
        ..Default::default()
    }).expect("Unable to create swapchain");
    
    // Create Image Views

    // Create Framebuffers

    // Create Render Pass

    // Create Graphics Pipeline

    // Create Command Pool

    // Create Command Buffers

    // Create Main loop

    Ok(())
}
