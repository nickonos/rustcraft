#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use std::sync::Arc;
use std::{clone, future};

use anyhow::Result;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    self, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
    QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::{
    Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, Version, VulkanError, library, swapchain};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[VertexPoint]>,
    rctx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct VertexPoint {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> App {
        // Create VKInstance
        let library = library::VulkanLibrary::new().expect("No local Vulkan library found");
        let required_extensions =
            Surface::required_extensions(event_loop).expect("unable to get required extensions");

        let instance_create_info = InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        };

        let instance =
            Instance::new(library, instance_create_info).expect("Failed to create instance");

        // Make sure physical device has swapchain support
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // Select Physical Device
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("Could not enumerate physical devices")
            // Make sure that the physical device has the dynamic rendering feature enabled
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            // Make sure that the system has the required extensions enabled
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            // For each physical device, we try to find a suitable queue family that will
            // execute our draw commands.
            //
            // Devices can provide multiple queues to run commands in parallel (for example a
            // draw queue and a compute queue), similar to CPU threads. This is something you
            // have to have to manage manually in Vulkan. Queues of the same type belong to the
            // same queue family.
            //
            // Here, we look for a single queue family that is suitable for our purposes. In a
            // real-world application, you may want to use a separate dedicated transfer queue
            // to handle data transfers in parallel with graphics operations. You may also need
            // a separate queue for compute operations, if your application uses those.
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing
                        // to a window surface, as we do in this example, we also need to check
                        // that queues in this queue family are capable of presenting images to the
                        // surface.
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                                .unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            // Search for the most suitable physical device
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("Could not find a suitable physical device");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to create logical device");

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let vertices = [
            VertexPoint {
                position: [-0.5, 0.25],
            },
            VertexPoint {
                position: [0.0, 0.5],
            },
            VertexPoint {
                position: [0.25, -0.1],
            },
        ];

        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .expect("Unable to create vertex buffer");

        App {
            instance,
            device,
            queue,
            command_buffer_allocator,
            vertex_buffer,
            rctx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Minecraft")
            .with_inner_size(LogicalSize::new(800, 600));

        let window = Arc::new(
            ActiveEventLoop::create_window(event_loop, window_attributes)
                .expect("Could not create window"),
        );

        // Create Window Surface
        let surface = Surface::from_window(self.instance.clone(), window.clone())
            .expect("Unable to create surface from window");

        let window_size = window.inner_size();

        let capabilities = self
            .device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .expect("could not get physical device capabilities");

        let composite_alpha = capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap();

        let image_format = self
            .device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            self.device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: capabilities.min_image_count.max(2),
                image_format,
                image_extent: window_size.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )
        .expect("Unable to create swapchain");

        let attachment_image_views = window_size_dependent_setup(&images);

        mod vertex_shader {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec2 position;

                    void main(){
                        gl_Position = vec4(position, 0.0, 1.0);
                    }
                ",
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) out vec4 f_color;

                    void main() {
                        f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                "
            }
        }

        let pipeline = {
            let vertex_shader = vertex_shader::load(self.device.clone())
                .expect("Unable to load the vertex shader")
                .entry_point("main")
                .expect("Unable to find vertex shader entrypoint");

            let fragment_shader = fragment_shader::load(self.device.clone())
                .expect("Unable to load the fragment shader")
                .entry_point("main")
                .expect("Unable to find fragment shader entrypoint");

            let vertex_input_state = VertexPoint::per_vertex()
                .definition(&vertex_shader)
                .expect("Unable to create vertex shader input");

            let stages = [
                PipelineShaderStageCreateInfo::new(vertex_shader),
                PipelineShaderStageCreateInfo::new(fragment_shader),
            ];

            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .expect("Unable to create pipeline layout info from stages"),
            )
            .expect("Unable to create pipeline layout");

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(swapchain.image_format())],
                ..Default::default()
            };

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.color_attachment_formats.len() as u32,
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .expect("Unable to create graphics pipeline")
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rctx = Some(RenderContext {
            window,
            swapchain,
            attachment_image_views,
            pipeline,
            viewport,
            recreate_swapchain: false,
            previous_frame_end: previous_frame_end,
        })
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let rctx = self.rctx.as_mut().expect("Unable to find render context");

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rctx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rctx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rctx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if rctx.recreate_swapchain {
                    let (new_swapchain, new_images) = rctx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rctx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rctx.swapchain = new_swapchain;
                    rctx.attachment_image_views = window_size_dependent_setup(&new_images);
                    rctx.viewport.extent = window_size.into();

                    rctx.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(rctx.swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            rctx.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("unable to acquire next image: {e}"),
                    };
                if suboptimal {
                    rctx.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .expect("Unable to create command buffer builder");

                builder
                    .begin_rendering(RenderingInfo {
                        color_attachments: vec![Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                            ..RenderingAttachmentInfo::image_view(
                                rctx.attachment_image_views[image_index as usize].clone(),
                            )
                        })],
                        ..Default::default()
                    })
                    .expect("Unable to start rendering")
                    .set_viewport(0, [rctx.viewport.clone()].into_iter().collect())
                    .expect("Unable to set viewport of command buffer")
                    .bind_pipeline_graphics(rctx.pipeline.clone())
                    .expect("Unable to bind pipeline")
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .expect("Unable to bind vertex buffers");

                // Add a draw command
                unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }
                    .expect("Unable to draw");

                builder.end_rendering().expect("Unable to end rendering");

                let command_buffer = builder.build().expect("Unable to build command buffer");

                let future = rctx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .expect("Unable to execute command buffer")
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rctx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rctx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rctx.recreate_swapchain = true;
                        rctx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        print!("Unable to flush future: {e}");
                        rctx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // Create window
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app)?;

    Ok(())
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(images: &[Arc<Image>]) -> Vec<Arc<ImageView>> {
    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}
