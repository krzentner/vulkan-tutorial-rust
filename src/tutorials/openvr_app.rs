use std::time::Duration;
use std::time::Instant;
use vulkan_tutorial_rust::{
    utility, // the mod define some fixed functions that have been learned before.
    utility::constants::*,
    utility::debug::*,
    utility::multisampling::{
        create_color_resources, create_graphics_pipeline, create_texture_image,
        create_texture_sampler, get_max_usable_sample_count,
    },
    utility::share,
    utility::structures::*,
    utility::window::ProgramProc,
};

use ash::version::DeviceV1_0;
use ash::version::InstanceV1_0;
use ash::vk;
use ash::vk::Handle;
use cgmath::{Deg, Matrix4, Point3, Vector3};
use openvr;

use std::convert::TryInto;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

use ash::version::EntryV1_0;

use std::os::raw::c_void;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

const IS_PAINT_FPS_COUNTER: bool = true;

use crate::utility::debug;
use crate::utility::platforms;

// Constants
const WINDOW_TITLE: &'static str = "OpenVR App";
const MODEL_PATH: &'static str = "assets/chalet.obj";
const TEXTURE_PATH: &'static str = "assets/chalet.jpg";
const NEAR_Z: f32 = 0.01;
const FAR_Z: f32 = 10000.0;
// 90fps +
const TIME_TO_PHOTONS: f32 = (2. + 11.1) / 1000.0;
const TRACKING_UNIVERSE: openvr::TrackingUniverseOrigin = openvr::TrackingUniverseOrigin::Standing;
const N_EYES: usize = 2;
const VR_FRAMES: usize = 2;

struct DevicePoses {
    hmd: Option<Matrix4<f32>>,
    left_hand: Option<Matrix4<f32>>,
    right_hand: Option<Matrix4<f32>>,
    lighthouses: Vec<Matrix4<f32>>,
}

impl DevicePoses {
    fn at_time_offset(system: &openvr::System, time_to_photons: f32) -> Self {
        Self::from_tracked_device_poses(
            system,
            system.device_to_absolute_tracking_pose(TRACKING_UNIVERSE, time_to_photons),
        )
    }

    fn from_tracked_device_poses(
        system: &openvr::System,
        poses: openvr::TrackedDevicePoses,
    ) -> Self {
        let mut hmd = None;
        let mut left_hand = None;
        let mut right_hand = None;
        let mut lighthouses = Vec::with_capacity(2);
        for (index, pose) in poses.iter().enumerate() {
            match system.tracked_device_class(index as u32) {
                openvr::TrackedDeviceClass::HMD => {
                    hmd = Some(pose_to_mat(pose));
                }
                openvr::TrackedDeviceClass::Controller => {
                    match system.get_controller_role_for_tracked_device_index(index as u32) {
                        Some(openvr::TrackedControllerRole::RightHand) => {
                            right_hand = Some(pose_to_mat(pose));
                        }
                        Some(openvr::TrackedControllerRole::LeftHand) => {
                            left_hand = Some(pose_to_mat(pose));
                        }
                        None => {
                            eprintln!("Controller at index {} has no role", index);
                        }
                    }
                }
                openvr::TrackedDeviceClass::TrackingReference => {
                    lighthouses.push(pose_to_mat(pose));
                }
                openvr::TrackedDeviceClass::Invalid => { /* Expected. */ }
                other => {
                    println!("Extra device of class {:?}", other);
                }
            }
        }
        DevicePoses {
            hmd,
            left_hand,
            right_hand,
            lighthouses,
        }
    }
}

fn pose_to_mat(tracked_device: &openvr::TrackedDevicePose) -> Matrix4<f32> {
    convert_ovr_mat(tracked_device.device_to_absolute_tracking())
}

fn convert_ovr_mat(mat: &[[f32; 4]; 3]) -> Matrix4<f32> {
    Matrix4::<f32>::from([mat[0], mat[1], mat[2], [0., 0., 0., 1.]])
}

//fn convert_mat_ovr(mat: Matrix4<f32>) -> &[[f32; 4]; 3] {
//let m: [[f32; 4], 4] = mat.into();
//m[0..3].to_owned()
//}

struct Viewer {
    eye: Option<openvr::Eye>,
    model: Matrix4<f32>,
    projection: Matrix4<f32>,
    pose: Matrix4<f32>,
}

impl Viewer {
    fn from_eye(system: &openvr::System, eye: openvr::Eye) -> Self {
        let eye_to_head = convert_ovr_mat(&system.eye_to_head_transform(eye));
        let pose = DevicePoses::at_time_offset(system, TIME_TO_PHOTONS)
            .hmd
            .unwrap_or(Matrix4::from_scale(1.0));
        Viewer {
            eye: Some(eye),
            model: Matrix4::from_angle_z(Deg(90.0)),
            projection: eye_to_head * Matrix4::from(system.projection_matrix(eye, NEAR_Z, FAR_Z)),
            pose,
        }
    }

    fn from_width_height(width: f32, height: f32) -> Self {
        Viewer {
            eye: None,
            model: Matrix4::from_angle_z(Deg(90.0)),
            projection: {
                let mut proj = cgmath::perspective(Deg(45.0), width / height, NEAR_Z, FAR_Z);
                proj[1][1] = proj[1][1] * -1.0;
                proj
            },
            pose: Matrix4::look_at(
                Point3::new(2.0, 2.0, 2.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ),
        }
    }

    fn uniform_transform(&self) -> UniformBufferObject {
        UniformBufferObject {
            model: self.model,
            view: self.pose.clone(),
            proj: self.projection.clone(),
        }
    }
}

struct OpenVRStuff {
    context: openvr::Context,
    system: openvr::System,
    compositor: openvr::Compositor,
    render_models: openvr::RenderModels,
    chaperone: openvr::Chaperone,
}

impl OpenVRStuff {
    fn new() -> Self {
        let context = unsafe { openvr::init(openvr::ApplicationType::Scene) }
            .expect("Could not initialize OpenVR");
        let system = context.system().expect("Could not get OpenVR system");
        let compositor = context
            .compositor()
            .expect("Could not get OpenVR compositor");
        let render_models = context
            .render_models()
            .expect("Could not get OpenVR render_models");
        let chaperone = context.chaperone().expect("Could not get OpenVR chaperone");
        OpenVRStuff {
            context,
            system,
            compositor,
            render_models,
            chaperone,
        }
    }

    fn select_physical_device(
        &self,
        instance: &ash::Instance,
        entry: &ash::Entry,
        surface_stuff: &SurfaceStuff,
    ) -> (vk::PhysicalDevice, Vec<CString>) {
        // init vulkan stuff
        let instance_handle = instance.handle().as_raw() as *mut openvr::VkInstance_T;
        let physical_device;
        let base_ext = vec![CString::new("VK_KHR_swapchain").unwrap()];
        let mut required_ext = base_ext.clone();
        if let Some(device_handle) = self.system.vulkan_output_device(instance_handle) {
            required_ext = unsafe {
                self.compositor
                    .vulkan_device_extensions_required(device_handle)
            };
            required_ext.extend(base_ext.iter().cloned());
            // Workaround OpenVR not listing all required extensions.
            dbg!(&required_ext);
            physical_device = vk::PhysicalDevice::from_raw(device_handle as u64);
            if !share::is_physical_device_suitable(
                &instance,
                physical_device,
                &surface_stuff,
                &required_ext,
            ) {
                panic!("OpenVR output device does not support required vulkan extensions");
            }
            println!("Got OpenVR output device");
        } else {
            println!("No OpenVR output device specified, picking appropriate device");
            let physical_devices = unsafe {
                instance
                    .enumerate_physical_devices()
                    .expect("Failed to enumerate Physical Devices!")
            };

            let result = physical_devices.iter().find(|physical_device| {
                let device_handle = physical_device.as_raw() as *mut openvr::VkPhysicalDevice_T;
                required_ext = unsafe {
                    self.compositor
                        .vulkan_device_extensions_required(device_handle)
                };
                required_ext.extend(base_ext.iter().cloned());

                dbg!(&required_ext);
                let is_suitable = share::is_physical_device_suitable(
                    &instance,
                    **physical_device,
                    &surface_stuff,
                    &required_ext,
                );

                if is_suitable {
                    let device_properties =
                        unsafe { instance.get_physical_device_properties(**physical_device) };
                    let device_name = utility::tools::vk_to_string(&device_properties.device_name);
                    println!("Using GPU: {}", device_name);
                }

                is_suitable
            });

            if let Some(p_physical_device) = result {
                physical_device = *p_physical_device;
            } else {
                panic!("Failed to find a suitable GPU.");
            }
        }
        // The above may cause a validation error, since the VK_KHR_external_semaphore_capabilities
        // extension wasn't requested, but the VK_KHR_external_semaphore extension was.
        // This works around a bug where the latter is supported, even though the prior isn't.
        return (physical_device, required_ext);
    }
}

struct EyeFramebuffer {
    color: vk::Image,
    color_memory: vk::DeviceMemory,
    color_layout: vk::ImageLayout,
    color_view: vk::ImageView,
    depth: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_layout: vk::ImageLayout,
    depth_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    uniform_buffers: [vk::Buffer; VR_FRAMES],
    uniform_buffers_memory: [vk::DeviceMemory; VR_FRAMES],
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; VR_FRAMES],
    image_available_semaphores: [vk::Semaphore; VR_FRAMES],
    render_finished_semaphores: [vk::Semaphore; VR_FRAMES],
    in_flight_fences: [vk::Fence; VR_FRAMES],
    current_frame: usize,
    viewer: Viewer,
    msaa_samples: vk::SampleCountFlags,
    width: u32,
    height: u32,
    eye: openvr::Eye,
}

impl EyeFramebuffer {
    fn create(
        device: &ash::Device,
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        ubo_layout: vk::DescriptorSetLayout,
        queue_family: &QueueFamilyIndices,
        viewer: Viewer,
        model: &Model,
    ) -> Option<Self> {
        let mip_levels = 1;
        let (color, color_memory) = share::v1::create_image(
            device,
            width,
            height,
            mip_levels,
            msaa_samples,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );
        let color_view = share::v1::create_image_view(
            device,
            color,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        );
        let (depth, depth_memory) = share::v1::create_image(
            device,
            width,
            height,
            mip_levels,
            msaa_samples,
            vk::Format::D32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );
        let depth_view = share::v1::create_image_view(
            device,
            depth,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
            mip_levels,
        );

        let color_attachment = vk::AttachmentDescription {
            format: vk::Format::R8G8B8A8_SRGB,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpasses = [vk::SubpassDescription {
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            p_resolve_attachments: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        }];

        let renderpass_attachments = [color_attachment, depth_attachment];

        let subpass_dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];

        let renderpass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            flags: vk::RenderPassCreateFlags::empty(),
            p_next: ptr::null(),
            attachment_count: renderpass_attachments.len() as u32,
            p_attachments: renderpass_attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr(),
            //dependency_count: 0,
            //p_dependencies: ptr::null(),
        };

        let render_pass = unsafe {
            device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create render pass!")
        };

        let framebuffer_attachments = [color_view, depth_view];

        let framebuffer_create_info = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass,
            attachment_count: framebuffer_attachments.len() as u32,
            p_attachments: framebuffer_attachments.as_ptr(),
            width,
            height,
            layers: 1,
        };

        let framebuffer = unsafe {
            device
                .create_framebuffer(&framebuffer_create_info, None)
                .expect("Failed to create Framebuffer!")
        };

        let (uniform_buffers, uniform_buffers_memory) =
            share::v1::create_uniform_buffers(&device, &device_memory_properties, VR_FRAMES);
        let command_pool = share::v1::create_command_pool(&device, &queue_family);

        let descriptor_pool = share::v2::create_descriptor_pool(&device, VR_FRAMES);
        let descriptor_sets = share::v2::create_descriptor_sets(
            &device,
            descriptor_pool,
            ubo_layout,
            &uniform_buffers,
            model.texture_image_view,
            model.texture_sampler,
            VR_FRAMES,
        );
        let extent = vk::Extent2D { width, height };
        let (pipeline, pipeline_layout) =
            create_graphics_pipeline(&device, render_pass, extent, ubo_layout, msaa_samples);
        let command_buffers = OpenVRApp::create_command_buffers(
            &device,
            command_pool,
            pipeline,
            &vec![framebuffer],
            render_pass,
            extent,
            model.vertex_buffer,
            model.index_buffer,
            pipeline_layout,
            &descriptor_sets,
            model.indices.len() as u32,
        );
        let sync_ojbects = share::v1::create_sync_objects(&device, VR_FRAMES);
        let eye = viewer.eye.unwrap();

        return Some(EyeFramebuffer {
            color,
            color_memory,
            color_layout: vk::ImageLayout::UNDEFINED,
            color_view,
            depth,
            depth_memory,
            depth_layout: vk::ImageLayout::UNDEFINED,
            depth_view,
            render_pass,
            framebuffer,
            pipeline,
            pipeline_layout,
            command_pool,
            command_buffers,
            uniform_buffers: uniform_buffers
                .try_into()
                .expect("Wrong number of uniform buffers"),
            uniform_buffers_memory: uniform_buffers_memory
                .try_into()
                .expect("Wrong number of uniform buffer allocations"),
            descriptor_pool,
            descriptor_sets: descriptor_sets
                .try_into()
                .expect("Wrong number of descriptor sets"),
            image_available_semaphores: sync_ojbects
                .image_available_semaphores
                .try_into()
                .expect("Wrong number of image available semaphores created"),
            render_finished_semaphores: sync_ojbects
                .render_finished_semaphores
                .try_into()
                .expect("Wrong number of render finished semaphores created"),
            in_flight_fences: sync_ojbects
                .inflight_fences
                .try_into()
                .expect("Wrong number of fences created"),
            current_frame: 0,
            viewer,
            width,
            height,
            msaa_samples,
            eye,
        });
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_image(self.color, None);
        device.destroy_image_view(self.color_view, None);
        device.free_memory(self.color_memory, None);

        device.destroy_image(self.depth, None);
        device.destroy_image_view(self.depth_view, None);
        device.free_memory(self.depth_memory, None);
        device.destroy_render_pass(self.render_pass, None);
        device.destroy_framebuffer(self.framebuffer, None);

        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);

        //device.free_command_buffers(self.command_pool, &self.command_buffers);
        device.destroy_command_pool(self.command_pool, None);
        for i in 0..self.uniform_buffers.len() {
            device.destroy_buffer(self.uniform_buffers[i], None);
            device.free_memory(self.uniform_buffers_memory[i], None);
        }
        device.destroy_descriptor_pool(self.descriptor_pool, None);

        for i in 0..VR_FRAMES {
            device.destroy_semaphore(self.image_available_semaphores[i], None);
            device.destroy_semaphore(self.render_finished_semaphores[i], None);
            device.destroy_fence(self.in_flight_fences[i], None);
        }
    }
}

struct CompanionWindowStuff {
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,

    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    color_image: vk::Image,
    color_image_view: vk::ImageView,
    color_image_memory: vk::DeviceMemory,

    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    check_framebuffer: bool,
    viewer: Viewer,
}

impl CompanionWindowStuff {
    fn create(
        instance: &ash::Instance,
        device: &ash::Device,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        physical_device: vk::PhysicalDevice,
        window: &winit::window::Window,
        surface_stuff: SurfaceStuff,
        queue_family: &QueueFamilyIndices,
        msaa_samples: vk::SampleCountFlags,
        ubo_layout: vk::DescriptorSetLayout,

        model: &Model,
    ) -> Option<Self> {
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family.present_family.unwrap(), 0) };
        let command_pool = share::v1::create_command_pool(&device, &queue_family);
        let swapchain_stuff = share::create_swapchain(
            instance,
            device,
            physical_device,
            window,
            &surface_stuff,
            queue_family,
        );
        let swapchain_imageviews = share::v1::create_image_views(
            &device,
            swapchain_stuff.swapchain_format,
            &swapchain_stuff.swapchain_images,
        );
        let render_pass = OpenVRApp::create_render_pass(
            &instance,
            &device,
            physical_device,
            swapchain_stuff.swapchain_format,
            msaa_samples,
        );
        let (graphics_pipeline, pipeline_layout) = create_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
            ubo_layout,
            msaa_samples,
        );
        let (color_image, color_image_view, color_image_memory) = create_color_resources(
            &device,
            swapchain_stuff.swapchain_format,
            swapchain_stuff.swapchain_extent,
            device_memory_properties,
            msaa_samples,
        );
        let (depth_image, depth_image_view, depth_image_memory) = share::v1::create_depth_resources(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            swapchain_stuff.swapchain_extent,
            device_memory_properties,
            msaa_samples,
        );
        let swapchain_framebuffers = OpenVRApp::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            depth_image_view,
            color_image_view,
            swapchain_stuff.swapchain_extent,
        );
        let (uniform_buffers, uniform_buffers_memory) = share::v1::create_uniform_buffers(
            &device,
            device_memory_properties,
            swapchain_stuff.swapchain_images.len(),
        );
        let descriptor_pool =
            share::v2::create_descriptor_pool(&device, swapchain_stuff.swapchain_images.len());
        let descriptor_sets = share::v2::create_descriptor_sets(
            &device,
            descriptor_pool,
            ubo_layout,
            &uniform_buffers,
            model.texture_image_view,
            model.texture_sampler,
            swapchain_stuff.swapchain_images.len(),
        );
        let command_buffers = OpenVRApp::create_command_buffers(
            &device,
            command_pool,
            graphics_pipeline,
            &swapchain_framebuffers,
            render_pass,
            swapchain_stuff.swapchain_extent,
            model.vertex_buffer,
            model.index_buffer,
            pipeline_layout,
            &descriptor_sets,
            model.indices.len() as u32,
        );
        let swapchain_loader = swapchain_stuff.swapchain_loader;
        let swapchain = swapchain_stuff.swapchain;
        let swapchain_images = swapchain_stuff.swapchain_images;
        let swapchain_format = swapchain_stuff.swapchain_format;
        let swapchain_extent = swapchain_stuff.swapchain_extent;
        let sync_ojbects = share::v1::create_sync_objects(&device, MAX_FRAMES_IN_FLIGHT);
        Some(CompanionWindowStuff {
            surface: surface_stuff.surface,
            surface_loader: surface_stuff.surface_loader,

            swapchain_loader,
            swapchain,
            swapchain_format,
            swapchain_images,
            swapchain_extent,
            swapchain_framebuffers,
            swapchain_imageviews,

            color_image,
            color_image_view,
            color_image_memory,
            depth_image,
            depth_image_view,
            depth_image_memory,
            render_pass,

            pipeline_layout,
            graphics_pipeline,
            command_pool,
            command_buffers,

            uniform_buffers,
            uniform_buffers_memory,

            descriptor_pool,
            descriptor_sets,

            graphics_queue,
            present_queue,

            //image_available_semaphores,
            //render_finished_semaphores,
            //in_flight_fences,
            image_available_semaphores: sync_ojbects.image_available_semaphores,
            render_finished_semaphores: sync_ojbects.render_finished_semaphores,
            in_flight_fences: sync_ojbects.inflight_fences,
            current_frame: 0,
            check_framebuffer: false,
            viewer: Viewer::from_width_height(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32),
        })
    }
    unsafe fn destroy(&self, device: &ash::Device) {
        self.destroy_swapchain(device);
        for i in 0..self.uniform_buffers.len() {
            device.destroy_buffer(self.uniform_buffers[i], None);
            device.free_memory(self.uniform_buffers_memory[i], None);
        }
        device.destroy_descriptor_pool(self.descriptor_pool, None);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            device.destroy_semaphore(self.image_available_semaphores[i], None);
            device.destroy_semaphore(self.render_finished_semaphores[i], None);
            device.destroy_fence(self.in_flight_fences[i], None);
        }
        self.surface_loader.destroy_surface(self.surface, None);
    }

    unsafe fn destroy_swapchain(&self, device: &ash::Device) {
        device.free_command_buffers(self.command_pool, &self.command_buffers);
        device.destroy_command_pool(self.command_pool, None);
        device.destroy_image(self.depth_image, None);
        device.destroy_image_view(self.depth_image_view, None);
        device.free_memory(self.depth_image_memory, None);

        device.destroy_image(self.color_image, None);
        device.destroy_image_view(self.color_image_view, None);
        device.free_memory(self.color_image_memory, None);

        for &framebuffer in self.swapchain_framebuffers.iter() {
            device.destroy_framebuffer(framebuffer, None);
        }
        for &image_view in self.swapchain_imageviews.iter() {
            device.destroy_image_view(image_view, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        device.destroy_render_pass(self.render_pass, None);
        device.destroy_pipeline(self.graphics_pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
    }

    fn update_uniform_buffer(
        &mut self,
        device: &ash::Device,
        current_image: usize,
        delta_time: f32,
    ) {
        self.viewer.model =
            Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Deg(90.0) * delta_time)
                * self.viewer.model;

        let ubos = [self.viewer.uniform_transform()];

        let buffer_size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

        unsafe {
            let data_ptr = device
                .map_memory(
                    self.uniform_buffers_memory[current_image],
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory")
                as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            device.unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }
}

struct Model {
    _mip_levels: u32,
    texture_image: vk::Image,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    texture_image_memory: vk::DeviceMemory,

    _vertices: Vec<VertexV3>,
    indices: Vec<u32>,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
}

impl Model {
    fn new(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue_family: &QueueFamilyIndices,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let (vertices, indices) = share::load_model(&Path::new(MODEL_PATH));
        let (texture_image, texture_image_memory, mip_levels) = create_texture_image(
            device,
            command_pool,
            graphics_queue,
            device_memory_properties,
            &Path::new(TEXTURE_PATH),
        );
        let texture_image_view =
            share::v1::create_texture_image_view(&device, texture_image, mip_levels);
        let texture_sampler = create_texture_sampler(&device, mip_levels);
        let (vertex_buffer, vertex_buffer_memory) = share::v1::create_vertex_buffer(
            device,
            device_memory_properties,
            command_pool,
            graphics_queue,
            &vertices,
        );
        let (index_buffer, index_buffer_memory) = share::v1::create_index_buffer(
            &device,
            device_memory_properties,
            command_pool,
            graphics_queue,
            &indices,
        );
        return Model {
            _mip_levels: mip_levels,
            texture_image,
            texture_image_view,
            texture_sampler,
            texture_image_memory,

            _vertices: vertices,
            indices,

            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
        };
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_buffer(self.index_buffer, None);
        device.free_memory(self.index_buffer_memory, None);

        device.destroy_buffer(self.vertex_buffer, None);
        device.free_memory(self.vertex_buffer_memory, None);

        device.destroy_sampler(self.texture_sampler, None);
        device.destroy_image_view(self.texture_image_view, None);

        device.destroy_image(self.texture_image, None);
        device.free_memory(self.texture_image_memory, None);
    }
}

struct OpenVRApp {
    window: winit::window::Window,

    // OpenVR stuff
    ovr_stuff: OpenVRStuff,
    eye_framebuffers: [EyeFramebuffer; N_EYES],
    companion_stuff: Mutex<CompanionWindowStuff>,
    model: Model,

    // vulkan stuff
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    ubo_layout: vk::DescriptorSetLayout,
    msaa_samples: vk::SampleCountFlags,

    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,

    queue_family: QueueFamilyIndices,
}

fn create_instance(
    entry: &ash::Entry,
    window_title: &str,
    is_enable_debug: bool,
    required_validation_layers: &Vec<&str>,
    required_vr_extensions: &Vec<CString>,
) -> ash::Instance {
    if is_enable_debug
        && debug::check_validation_layer_support(entry, required_validation_layers) == false
    {
        panic!("Validation layers requested, but not available!");
    }

    let app_name = CString::new(window_title).unwrap();
    let engine_name = CString::new("Vulkan Engine").unwrap();
    let app_info = vk::ApplicationInfo {
        p_application_name: app_name.as_ptr(),
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        application_version: APPLICATION_VERSION,
        p_engine_name: engine_name.as_ptr(),
        engine_version: ENGINE_VERSION,
        api_version: API_VERSION,
    };

    // This create info used to debug issues in vk::createInstance and vk::destroyInstance.
    let debug_utils_create_info = debug::populate_debug_messenger_create_info();

    // VK_EXT debug report has been requested here.
    let mut extension_names = platforms::required_extension_names();
    extension_names.extend(required_vr_extensions.iter().map(|s| s.as_ptr()));

    let required_layers = if is_enable_debug {
        required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect()
    } else {
        vec![]
    };

    let layer_names = share::raw_cstr_array(&required_layers);

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: if VALIDATION.is_enable {
            &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
        } else {
            ptr::null()
        },
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        pp_enabled_layer_names: layer_names.as_ptr(),
        enabled_layer_count: layer_names.len() as u32,
        pp_enabled_extension_names: extension_names.as_ptr(),
        enabled_extension_count: extension_names.len() as u32,
    };

    let instance: ash::Instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create instance!")
    };

    instance
}

impl OpenVRApp {
    fn create_render_pass(
        instance: &ash::Instance,
        device: &ash::Device,
        physcial_device: vk::PhysicalDevice,
        surface_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
    ) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: share::find_depth_format(instance, physcial_device),
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_resolve = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_resolve_ref = vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpasses = [vk::SubpassDescription {
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            p_resolve_attachments: &color_attachment_resolve_ref,
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        }];

        let render_pass_attachments =
            [color_attachment, depth_attachment, color_attachment_resolve];

        let subpass_dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];

        let renderpass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            flags: vk::RenderPassCreateFlags::empty(),
            p_next: ptr::null(),
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr(),
        };

        unsafe {
            device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create render pass!")
        }
    }

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        swapchain_image_views: &Vec<vk::ImageView>,
        depth_image_view: vk::ImageView,
        color_image_view: vk::ImageView,
        swapchain_extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = vec![];

        for &image_view in swapchain_image_views.iter() {
            let attachments = [color_image_view, depth_image_view, image_view];

            let framebuffer_create_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                layers: 1,
            };

            let framebuffer = unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create Framebuffer!")
            };

            framebuffers.push(framebuffer);
        }

        framebuffers
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        graphics_pipeline: vk::Pipeline,
        framebuffers: &Vec<vk::Framebuffer>,
        render_pass: vk::RenderPass,
        surface_extent: vk::Extent2D,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &Vec<vk::DescriptorSet>,
        index_count: u32,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: framebuffers.len() as u32,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        };

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                p_inheritance_info: ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            };

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [
                vk::ClearValue {
                    // clear value for color buffer
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    // clear value for depth buffer
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass,
                framebuffer: framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
            };

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );

                let vertex_buffers = [vertex_buffer];
                let offsets = [0_u64];
                let descriptor_sets_to_bind = [descriptor_sets[i]];

                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.cmd_draw_indexed(command_buffer, index_count, 1, 0, 0, 0);

                device.cmd_end_render_pass(command_buffer);

                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        command_buffers
    }
}

impl Drop for OpenVRApp {
    fn drop(&mut self) {
        unsafe {
            self.companion_stuff.lock().unwrap().destroy(&self.device);
            for eye in &self.eye_framebuffers {
                eye.destroy(&self.device);
            }
            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);

            self.model.destroy(&self.device);

            self.device.destroy_device(None);

            if VALIDATION.is_enable {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl OpenVRApp {
    fn recreate_swapchain(&self) {
        todo!();
        /*
         */
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.companion_stuff
                .lock()
                .unwrap()
                .destroy_swapchain(&self.device);
        }
    }

    fn render_vr(&self) {}
}

fn draw_eye(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    ovr_stuff: &OpenVRStuff,
    queue: &vk::Queue,
    queue_family: &QueueFamilyIndices,
    buf: &mut EyeFramebuffer,
    msaa_samples: vk::SampleCountFlags,
) {
    let instance = instance.handle().as_raw() as *mut openvr::VkInstance_T;
    let device = device.handle().as_raw() as *mut openvr::VkDevice_T;
    let physical_device = physical_device.as_raw() as *mut openvr::VkPhysicalDevice_T;
    let color_space = openvr::compositor::texture::ColorSpace::Auto;
    let vulkan_output_texture = openvr::compositor::texture::vulkan::Texture {
        image: buf.color.as_raw(),
        device,
        physical_device,
        instance,
        queue: queue.as_raw() as *mut openvr::VkQueue_T,
        queue_family_index: queue_family.graphics_family.unwrap(),
        width: buf.width,
        height: buf.height,
        format: vk::Format::R8G8B8A8_SRGB.as_raw() as u32,
        sample_count: msaa_samples.as_raw(),
    };
    let handle = openvr::compositor::texture::Handle::Vulkan(vulkan_output_texture);
    let output_texture = openvr::compositor::texture::Texture {
        handle,
        color_space,
    };
    unsafe {
        ovr_stuff
            .compositor
            .submit(buf.eye, &output_texture, None, None)
            .expect("Could not submit texture to compositor");
    }

    buf.current_frame = (buf.current_frame + 1) % VR_FRAMES;
}

fn draw_frame(device: &ash::Device, companion_stuff: &mut CompanionWindowStuff, delta_time: f32) {
    let wait_fences = [companion_stuff.in_flight_fences[companion_stuff.current_frame]];

    unsafe {
        device
            .wait_for_fences(&wait_fences, true, std::u64::MAX)
            .expect("Failed to wait for Fence!");
    }

    let (image_index, _is_sub_optimal) = unsafe {
        let result = companion_stuff.swapchain_loader.acquire_next_image(
            companion_stuff.swapchain,
            std::u64::MAX,
            companion_stuff.image_available_semaphores[companion_stuff.current_frame],
            vk::Fence::null(),
        );
        match result {
            Ok(image_index) => image_index,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    recreate_swapchain(companion_stuff);
                    return;
                }
                _ => panic!("Failed to acquire Swap Chain Image!"),
            },
        }
    };

    companion_stuff.update_uniform_buffer(&device, image_index as usize, delta_time);

    let wait_semaphores =
        [companion_stuff.image_available_semaphores[companion_stuff.current_frame]];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let signal_semaphores =
        [companion_stuff.render_finished_semaphores[companion_stuff.current_frame]];

    let submit_infos = [vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: wait_semaphores.len() as u32,
        p_wait_semaphores: wait_semaphores.as_ptr(),
        p_wait_dst_stage_mask: wait_stages.as_ptr(),
        command_buffer_count: 1,
        p_command_buffers: &companion_stuff.command_buffers[image_index as usize],
        signal_semaphore_count: signal_semaphores.len() as u32,
        p_signal_semaphores: signal_semaphores.as_ptr(),
    }];

    unsafe {
        device
            .reset_fences(&wait_fences)
            .expect("Failed to reset Fence!");

        device
            .queue_submit(
                companion_stuff.graphics_queue,
                &submit_infos,
                companion_stuff.in_flight_fences[companion_stuff.current_frame],
            )
            .expect("Failed to execute queue submit.");
    }

    let swapchains = [companion_stuff.swapchain];

    let present_info = vk::PresentInfoKHR {
        s_type: vk::StructureType::PRESENT_INFO_KHR,
        p_next: ptr::null(),
        wait_semaphore_count: 1,
        p_wait_semaphores: signal_semaphores.as_ptr(),
        swapchain_count: 1,
        p_swapchains: swapchains.as_ptr(),
        p_image_indices: &image_index,
        p_results: ptr::null_mut(),
    };

    let result = unsafe {
        companion_stuff
            .swapchain_loader
            .queue_present(companion_stuff.present_queue, &present_info)
    };

    let is_resized = match result {
        Ok(_) => companion_stuff.check_framebuffer,
        Err(vk_result) => match vk_result {
            vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
            _ => panic!("Failed to execute queue present."),
        },
    };
    if is_resized {
        companion_stuff.check_framebuffer = false;
        recreate_swapchain(companion_stuff);
    }

    companion_stuff.current_frame = (companion_stuff.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

fn wait_device_idle(device: &ash::Device) {
    unsafe {
        device
            .device_wait_idle()
            .expect("Failed to wait device idle!")
    };
}

fn recreate_swapchain(companion_stuff: &mut CompanionWindowStuff) {
    //companion_stuff.destroy_swapchain(device);

    //let surface_suff = SurfaceStuff {
    //surface_loader: self.surface_loader.clone(),
    //surface: self.surface,
    //screen_width: WINDOW_WIDTH,
    //screen_height: WINDOW_HEIGHT,
    //};
    //*companion_stuff = CompanionWindowStuff::create(
    //&companion_stuff.instance,
    //&companion_stuff.device,
    //&companion_stuff.device_memory_properties,
    //&companion_stuff.physical_device,
    //companion_stuff.window,
    //companion_stuff.surface_stuff,
    //&companion_stuff.queue_family,
    //companion_stuff.msaa_samples,
    //companion_stuff.ubo_layout,
    //companion_stuff.model,
    //)
    //.expect("Could not setup companion window");

    //unsafe {
    //self.device
    //.device_wait_idle()
    //.expect("Failed to wait device idle!")
    //};
    //self.cleanup_swapchain();

    //let swapchain_stuff = share::create_swapchain(
    //&self.instance,
    //&self.device,
    //self.physical_device,
    //&self.window,
    //&surface_suff,
    //&self.queue_family,
    //);
    //self.swapchain_loader = swapchain_stuff.swapchain_loader;
    //self.swapchain = swapchain_stuff.swapchain;
    //self.swapchain_images = swapchain_stuff.swapchain_images;
    //self.swapchain_format = swapchain_stuff.swapchain_format;
    //self.swapchain_extent = swapchain_stuff.swapchain_extent;

    //self.swapchain_imageviews = share::v1::create_image_views(
    //&self.device,
    //self.swapchain_format,
    //&self.swapchain_images,
    //);
    //self.render_pass = OpenVRApp::create_render_pass(
    //&self.instance,
    //&self.device,
    //self.physical_device,
    //self.swapchain_format,
    //self.msaa_samples,
    //);
    //let (graphics_pipeline, pipeline_layout) = OpenVRApp::create_graphics_pipeline(
    //&self.device,
    //self.render_pass,
    //companion_stuff.swapchain_extent,
    //self.ubo_layout,
    //self.msaa_samples,
    //);
    //self.graphics_pipeline = graphics_pipeline;
    //self.pipeline_layout = pipeline_layout;

    //let color_resources = create_color_resources(
    //&self.device,
    //self.swapchain_format,
    //self.swapchain_extent,
    //&self.memory_properties,
    //self.msaa_samples,
    //);
    //self.color_image = color_resources.0;
    //self.color_image_view = color_resources.1;
    //self.color_image_memory = color_resources.2;

    //let depth_resources = share::v1::create_depth_resources(
    //&self.instance,
    //&self.device,
    //self.physical_device,
    //self.command_pool,
    //self.graphics_queue,
    //self.swapchain_extent,
    //&self.memory_properties,
    //self.msaa_samples,
    //);
    //self.depth_image = depth_resources.0;
    //self.depth_image_view = depth_resources.1;
    //self.depth_image_memory = depth_resources.2;

    //self.swapchain_framebuffers = OpenVRApp::create_framebuffers(
    //&self.device,
    //self.render_pass,
    //&self.swapchain_imageviews,
    //self.depth_image_view,
    //self.color_image_view,
    //self.swapchain_extent,
    //);
    //self.command_buffers = OpenVRApp::create_command_buffers(
    //&self.device,
    //self.command_pool,
    //self.graphics_pipeline,
    //&self.swapchain_framebuffers,
    //self.render_pass,
    //self.swapchain_extent,
    //self.vertex_buffer,
    //self.index_buffer,
    //self.pipeline_layout,
    //&self.descriptor_sets,
    //self.indices.len() as u32,
    //);
}

fn run_event_loop(
    mut companion_stuff: CompanionWindowStuff,
    device: ash::Device,
    window: winit::window::Window,
    program_proc: ProgramProc,
) {
    let mut tick_counter = utility::fps_limiter::FPSLimiter::new();
    let mut last_draw_time = Instant::now();
    let mut next_draw_time = Instant::now();
    program_proc.event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    wait_device_idle(&device);
                    *control_flow = ControlFlow::Exit
                }
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                            wait_device_idle(&device);
                            *control_flow = ControlFlow::Exit
                        }
                        _ => {}
                    },
                },
                WindowEvent::Resized(_new_size) => {
                    wait_device_idle(&device);
                    companion_stuff.check_framebuffer = true;
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_window_id) => {}
            Event::LoopDestroyed => {
                companion_stuff.check_framebuffer = true;
            }
            _ => (),
        }
        let frame_start = Instant::now();
        if frame_start >= next_draw_time {
            let delta_time = tick_counter.delta_time();
            draw_frame(&device, &mut companion_stuff, delta_time);

            if IS_PAINT_FPS_COUNTER {
                print!("FPS: {}\r", tick_counter.fps());
            }

            tick_counter.tick_frame();
            let frame_end = Instant::now();

            let frame_draw_duration = frame_end - frame_start;

            next_draw_time =
                frame_start + Duration::from_micros(1000_000 / 32) - frame_draw_duration;
            last_draw_time = frame_start;
        }
        match control_flow {
            ControlFlow::Exit => {}
            ctrl => *ctrl = ControlFlow::WaitUntil(next_draw_time),
        }
    })
}

fn main() {
    // Winit wants to own the main thread, and OpenVR insists on running in the thread it's
    // initialized in, but they need to collaborate to pick a GPU, so we need to do a weird dance.
    let (window_instance_tx, window_instance_rx) = channel();
    let (loader_instance_tx, loader_instance_rx) = channel();
    let (surface_stuff_tx, surface_stuff_rx) = channel();
    let (device_tx, device_rx) = channel();
    let (loader_dev_tx, loader_dev_rx) = channel();
    let (openvr_model_tx, openvr_model_rx) = channel();
    let (window_model_tx, window_model_rx) = channel();
    // OpenVR thread
    thread::spawn(move || {
        // Step 1:
        let ovr_stuff = OpenVRStuff::new();
        let entry = ash::Entry::new().unwrap();
        let instance = create_instance(
            &entry,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers.to_vec(),
            &ovr_stuff.compositor.vulkan_instance_extensions_required(),
        );
        for sender in &[&window_instance_tx, &loader_instance_tx] {
            sender.send((instance.clone(), entry.clone())).unwrap();
        }

        let (debug_utils_loader, debug_messenger) =
            setup_debug_utils(VALIDATION.is_enable, &entry, &instance);

        // Step 2 happens on main thread

        // Step 3:
        let surface_stuff: Arc<SurfaceStuff> = surface_stuff_rx.recv().unwrap();

        let (physical_device, required_ext) =
            ovr_stuff.select_physical_device(&instance, &entry, &surface_stuff);
        share::check_mipmap_support(&instance, physical_device, vk::Format::R8G8B8A8_UNORM);
        let msaa_samples = get_max_usable_sample_count(&instance, physical_device);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, queue_family) = share::create_logical_device(
            &instance,
            physical_device,
            &VALIDATION,
            &required_ext,
            &surface_stuff,
        );
        // Make sure we're not using surface_stuff anymore once we send device over.
        drop(surface_stuff);
        let queue_family = Arc::new(queue_family);
        for sender in &[&device_tx, &loader_dev_tx] {
            sender
                .send((
                    device.clone(),
                    Arc::clone(&queue_family),
                    physical_device.clone(),
                ))
                .unwrap();
        }

        let ubo_layout = share::v2::create_descriptor_set_layout(&device);
        let (eye_width, eye_height) = ovr_stuff.system.recommended_render_target_size();
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };

        let model: Arc<Model> = openvr_model_rx.recv().unwrap();

        let mut eye_framebuffers = [
            EyeFramebuffer::create(
                &device,
                eye_width,
                eye_height,
                msaa_samples,
                &physical_device_memory_properties,
                ubo_layout,
                &queue_family,
                Viewer::from_eye(&ovr_stuff.system, openvr::Eye::Left),
                &model,
            )
            .expect("Failed to allocated left eye framebuffer"),
            EyeFramebuffer::create(
                &device,
                eye_width,
                eye_height,
                msaa_samples,
                &physical_device_memory_properties,
                ubo_layout,
                &queue_family,
                Viewer::from_eye(&ovr_stuff.system, openvr::Eye::Right),
                &model,
            )
            .expect("Failed to allocate right eye framebuffer."),
        ];

        loop {
            let poses = ovr_stuff.compositor.wait_get_poses().unwrap();
            let dev_poses = DevicePoses::from_tracked_device_poses(&ovr_stuff.system, poses.render);
            if let Some(hmd_pose) = dev_poses.hmd {
                for eye in 0..N_EYES {
                    let buf = &mut eye_framebuffers[eye];
                    buf.viewer.pose = hmd_pose;
                    let ubos = [buf.viewer.uniform_transform()];

                    let buffer_size =
                        (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

                    unsafe {
                        let data_ptr = device
                            .map_memory(
                                buf.uniform_buffers_memory[buf.current_frame],
                                0,
                                buffer_size,
                                vk::MemoryMapFlags::empty(),
                            )
                            .expect("Failed to Map Memory")
                            as *mut UniformBufferObject;

                        data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

                        device.unmap_memory(buf.uniform_buffers_memory[buf.current_frame]);
                    }
                    // Only draw one frame for now, for debugging purposes
                    if buf.current_frame == 0 {
                        draw_eye(
                            &device,
                            &instance,
                            &physical_device,
                            &ovr_stuff,
                            &graphics_queue,
                            &queue_family,
                            buf,
                            msaa_samples,
                        );
                    }
                }
            }
        }
    });

    // Loader thread
    thread::spawn(move || {
        let (instance, _) = loader_instance_rx.recv().unwrap();
        let (device, queue_family, physical_device) = loader_dev_rx.recv().unwrap();
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let command_pool = share::v1::create_command_pool(&device, &queue_family);
        let model = Arc::new(Model::new(
            &device,
            command_pool,
            &queue_family,
            &physical_device_memory_properties,
        ));
        for sender in &[&openvr_model_tx, &window_model_tx] {
            sender.send(Arc::clone(&model)).unwrap();
        }
    });

    // Step 2:
    let program_proc = ProgramProc::new();

    let window = utility::window::init_window(
        &program_proc.event_loop,
        WINDOW_TITLE,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    );

    let (instance, entry) = window_instance_rx.recv().unwrap();

    let surface_stuff = Arc::new(share::create_surface(
        &entry,
        &instance,
        &window,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    ));
    surface_stuff_tx.send(Arc::clone(&surface_stuff)).unwrap();

    // Step 3 happens on OpenVR thread

    // Step 4:
    let (device, queue_family, physical_device) = device_rx.recv().unwrap();

    let msaa_samples = get_max_usable_sample_count(&instance, physical_device);
    let physical_device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let ubo_layout = share::v2::create_descriptor_set_layout(&device);

    let model: Arc<Model> = window_model_rx.recv().unwrap();

    let surface_stuff: SurfaceStuff = Arc::try_unwrap(surface_stuff)
        .ok()
        .expect("OpenVR thread should be done with surface now");

    let companion_stuff = CompanionWindowStuff::create(
        &instance,
        &device,
        &physical_device_memory_properties,
        physical_device,
        &window,
        surface_stuff,
        &queue_family,
        msaa_samples,
        ubo_layout,
        &model,
    )
    .expect("Could not setup companion window");

    run_event_loop(companion_stuff, device, window, program_proc);
}
