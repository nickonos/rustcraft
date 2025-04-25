use std::{collections::HashMap, sync::Arc, time::Instant};

use glam::{Mat3, Mat4, Vec3};
use vulkano::{
    Validated, Version, VulkanError,
    buffer::{
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, DeviceOwned, Queue,
        QueueCreateInfo, QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    library,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

use crate::{engine::movement::MovementDirection, model::{Normal, Position, INDICES, NORMALS, POSITIONS}};

use super::movement::get_movement_direction;

pub struct ApplicationEngine {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[Position]>,
    normals_buffer: Subbuffer<[Normal]>,
    index_buffer: Subbuffer<[u16]>,
    uniform_buffer_allocator: SubbufferAllocator,
    rctx: Option<RenderContext>,
    stored_movement_input: HashMap<MovementDirection, bool>,
    last_rendered_at: Instant,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    vertex_shader: EntryPoint,
    fragment_shader: EntryPoint,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl ApplicationEngine {
    pub fn new(event_loop: &EventLoop<()>) -> ApplicationEngine {
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
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // Select a queue family that supports graphics operations
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

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            POSITIONS,
        )
        .expect("Unable to create vertex buffer");

        let normals_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            NORMALS,
        )
        .expect("Unable to create normals buffer");

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            INDICES,
        )
        .expect("Unable to create index buffer");

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let pos_vec = POSITIONS.iter().map(|p| p.to_vec3().normalize());
        println!("normalized: {:?}", pos_vec);

        let stored_movement_input = HashMap::from([
            (MovementDirection::Forward, false),
            (MovementDirection::Backward, false),
            (MovementDirection::Left, false),
            (MovementDirection::Right, false),
            (MovementDirection::Up, false),
            (MovementDirection::Down, false),
        ]);


        ApplicationEngine {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            vertex_buffer,
            normals_buffer,
            index_buffer,
            uniform_buffer_allocator,
            rctx: None,
            stored_movement_input,
            last_rendered_at: Instant::now(),
        }
    }

    fn create_render_context(&mut self, event_loop: &ActiveEventLoop) {
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

        let vertex_shader = vertex_shader::load(self.device.clone())
            .expect("Unable to load the vertex shader")
            .entry_point("main")
            .expect("Unable to find vertex shader entrypoint");

        let fragment_shader = fragment_shader::load(self.device.clone())
            .expect("Unable to load the fragment shader")
            .entry_point("main")
            .expect("Unable to find fragment shader entrypoint");

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil}
            },
        )
        .expect("Unable to create render pass");

        let (framebuffers, pipeline) = window_size_dependent_setup(
            window_size,
            &images,
            &render_pass,
            &self.memory_allocator,
            &vertex_shader,
            &fragment_shader,
        );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        let rotation_start = Instant::now();

        self.rctx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            vertex_shader,
            fragment_shader,
            pipeline,
            viewport,
            recreate_swapchain: false,
            previous_frame_end,
        })
    }

    fn draw_screen(&mut self, event_loop: &ActiveEventLoop) {
        let rctx = self.rctx.as_mut().expect("Render Context not set");
        let window_size = rctx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        let elapsed = self.last_rendered_at.elapsed();
        let dt = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;

        let mut movement_direction: Vec3 = Vec3::new(0.0, 0.0, 0.0);
        for (dir, enabled) in self.stored_movement_input.iter().filter(|(_, enabled)| **enabled){
            movement_direction += dir.to_vec3();
        };
        // println!("{dt}");
        println!("{movement_direction}");

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

            let image_format = self
                .device
                .physical_device()
                .surface_formats(&rctx.swapchain.surface(), Default::default())
                .unwrap()[0]
                .0;

            (rctx.framebuffers, rctx.pipeline) = window_size_dependent_setup(
                window_size,
                &new_images,
                &rctx.render_pass,
                &self.memory_allocator,
                &rctx.vertex_shader,
                &rctx.fragment_shader,
            );
            rctx.viewport.extent = window_size.into();

            rctx.recreate_swapchain = false;
        }

        let uniform_buffer = {
            let rotation = Mat3::from_rotation_y(0 as f32);

            let aspect_ratio =
                rctx.swapchain.image_extent()[0] as f32 / rctx.swapchain.image_extent()[1] as f32;

            let projection =
                Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.01, 100.0);

            let view = Mat4::look_at_rh(
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
            );
            let scale = Mat4::from_scale(Vec3::splat(0.5));

            let uniform_data = vertex_shader::Data {
                world: Mat4::from_mat3(rotation).to_cols_array_2d(),
                view: (view * scale).to_cols_array_2d(),
                projection: projection.to_cols_array_2d(),
            };

            println!("world: {:?}", uniform_data.world);
            println!("view: {:?}", uniform_data.view);
            println!("projection: {:?}", uniform_data.projection);

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let layout = &rctx.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )
        .expect("Unable to create descriptor set");

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rctx.swapchain.clone(), None).map_err(Validated::unwrap) {
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
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.1, 1.0, 1.0].into()), Some(1f32.into())],
                    ..RenderPassBeginInfo::framebuffer(
                        rctx.framebuffers[image_index as usize].clone(),
                    )
                },
                Default::default(),
            )
            .expect("Unable to set viewport of command buffer")
            //
            .bind_pipeline_graphics(rctx.pipeline.clone())
            .expect("Unable to bind pipeline")
            //
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                rctx.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .expect("Unable to bind descriptor set")
            //
            .bind_vertex_buffers(0, (self.vertex_buffer.clone(), self.normals_buffer.clone()))
            .expect("Unable to bind normals buffers")
            //
            .bind_index_buffer(self.index_buffer.clone())
            .expect("Unable to bind index buffer");

        // Add a draw command
        unsafe { builder.draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0) }
            .expect("Unable to draw");

        builder
            .end_render_pass(Default::default())
            .expect("Unable to end render pass");

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
                SwapchainPresentInfo::swapchain_image_index(rctx.swapchain.clone(), image_index),
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
        self.last_rendered_at = Instant::now();
    }
}

impl ApplicationHandler for ApplicationEngine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.create_render_context(event_loop);
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
                self.draw_screen(event_loop);
            }
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {}
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
                
            } => {
                match event.logical_key.as_ref() {
                    _ => {}
                }

                let direction = get_movement_direction(event.logical_key.as_ref());
                if direction.is_some() {
                    self.stored_movement_input.insert(direction.unwrap(), event.state.is_pressed());
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let rctx = self.rctx.as_mut().unwrap();
        rctx.window.request_redraw();
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    window_size: PhysicalSize<u32>,
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    vs: &EntryPoint,
    fs: &EntryPoint,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
    let device = memory_allocator.device();

    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let pipeline = {
        let vertex_input_state = [Position::per_vertex(), Normal::per_vertex()]
            .definition(vs)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0, 0.0],
                        extent: window_size.into(),
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    (framebuffers, pipeline)
}

mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/vertex.glsl"
    }
}

mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/fragment.glsl"
    }
}
