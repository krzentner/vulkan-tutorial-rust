use crate::utility::structures::*;
use ash::version::DeviceV1_0;
use ash::version::InstanceV1_0;
use ash::vk;
use image::GenericImageView;

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use super::*;

pub fn create_texture_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    image_path: &Path,
) -> (vk::Image, vk::DeviceMemory, u32) {
    let mut image_object = image::open(image_path).unwrap(); // this function is slow in debug mode.
    image_object = image_object.flipv();
    let (image_width, image_height) = (image_object.width(), image_object.height());
    let image_data = match &image_object {
        image::DynamicImage::ImageBgr8(_)
        | image::DynamicImage::ImageLuma8(_)
        | image::DynamicImage::ImageRgb8(_) => image_object.to_rgba().into_raw(),
        image::DynamicImage::ImageBgra8(_)
        | image::DynamicImage::ImageLumaA8(_)
        | image::DynamicImage::ImageRgba8(_) => image_object.raw_pixels(),
    };
    let image_size =
        (::std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
    let mip_levels = ((::std::cmp::max(image_width, image_height) as f32)
        .log2()
        .floor() as u32)
        + 1;

    if image_size <= 0 {
        panic!("Failed to load texture image!")
    }

    let (staging_buffer, staging_buffer_memory) = share::create_buffer(
        device,
        image_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        device_memory_properties,
    );

    unsafe {
        let data_ptr = device
            .map_memory(
                staging_buffer_memory,
                0,
                image_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to Map Memory") as *mut u8;

        data_ptr.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());

        device.unmap_memory(staging_buffer_memory);
    }

    let (texture_image, texture_image_memory) = share::v1::create_image(
        device,
        image_width,
        image_height,
        mip_levels,
        vk::SampleCountFlags::TYPE_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        device_memory_properties,
    );

    share::v1::transition_image_layout(
        device,
        command_pool,
        submit_queue,
        texture_image,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
    );

    share::copy_buffer_to_image(
        device,
        command_pool,
        submit_queue,
        staging_buffer,
        texture_image,
        image_width,
        image_height,
    );

    share::v1::generate_mipmaps(
        device,
        command_pool,
        submit_queue,
        texture_image,
        image_width,
        image_height,
        mip_levels,
    );

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    (texture_image, texture_image_memory, mip_levels)
}

pub fn get_max_usable_sample_count(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::SampleCountFlags {
    let physical_device_properties =
        unsafe { instance.get_physical_device_properties(physical_device) };

    let count = std::cmp::min(
        physical_device_properties
            .limits
            .framebuffer_color_sample_counts,
        physical_device_properties
            .limits
            .framebuffer_depth_sample_counts,
    );

    if count.contains(vk::SampleCountFlags::TYPE_64) {
        return vk::SampleCountFlags::TYPE_64;
    }
    if count.contains(vk::SampleCountFlags::TYPE_32) {
        return vk::SampleCountFlags::TYPE_32;
    }
    if count.contains(vk::SampleCountFlags::TYPE_16) {
        return vk::SampleCountFlags::TYPE_16;
    }
    if count.contains(vk::SampleCountFlags::TYPE_8) {
        return vk::SampleCountFlags::TYPE_8;
    }
    if count.contains(vk::SampleCountFlags::TYPE_4) {
        return vk::SampleCountFlags::TYPE_4;
    }
    if count.contains(vk::SampleCountFlags::TYPE_2) {
        return vk::SampleCountFlags::TYPE_2;
    }

    vk::SampleCountFlags::TYPE_1
}

pub fn create_color_resources(
    device: &ash::Device,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    msaa_samples: vk::SampleCountFlags,
) -> (vk::Image, vk::ImageView, vk::DeviceMemory) {
    let color_format = swapchain_format;

    let (color_image, color_image_memory) = share::v1::create_image(
        device,
        swapchain_extent.width,
        swapchain_extent.height,
        1,
        msaa_samples,
        color_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        device_memory_properties,
    );

    let color_image_view = share::v1::create_image_view(
        device,
        color_image,
        color_format,
        vk::ImageAspectFlags::COLOR,
        1,
    );

    (color_image, color_image_view, color_image_memory)
}

pub fn create_texture_sampler(device: &ash::Device, mip_levels: u32) -> vk::Sampler {
    let sampler_create_info = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SamplerCreateFlags::empty(),
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        anisotropy_enable: vk::TRUE,
        max_anisotropy: 16.0,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::ALWAYS,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        min_lod: 0.0,
        max_lod: mip_levels as f32,
        mip_lod_bias: 0.0,
        border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        unnormalized_coordinates: vk::FALSE,
    };

    unsafe {
        device
            .create_sampler(&sampler_create_info, None)
            .expect("Failed to create Sampler!")
    }
}

pub fn create_graphics_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    ubo_set_layout: vk::DescriptorSetLayout,
    msaa_samples: vk::SampleCountFlags,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let vert_shader_module = share::create_shader_module(
        device,
        include_bytes!("../../shaders/spv/26-shader-depth.vert.spv").to_vec(),
    );
    let frag_shader_module = share::create_shader_module(
        device,
        include_bytes!("../../shaders/spv/26-shader-depth.frag.spv").to_vec(),
    );

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            // Vertex Shader
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: vert_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::VERTEX,
        },
        vk::PipelineShaderStageCreateInfo {
            // Fragment Shader
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: frag_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::FRAGMENT,
        },
    ];

    let binding_description = VertexV3::get_binding_descriptions();
    let attribute_description = VertexV3::get_attribute_descriptions();

    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        vertex_attribute_description_count: attribute_description.len() as u32,
        p_vertex_attribute_descriptions: attribute_description.as_ptr(),
        vertex_binding_description_count: binding_description.len() as u32,
        p_vertex_binding_descriptions: binding_description.as_ptr(),
    };
    let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
        p_next: ptr::null(),
        primitive_restart_enable: vk::FALSE,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
    };

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    }];

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::empty(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
    };

    let rasterization_statue_create_info = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineRasterizationStateCreateFlags::empty(),
        depth_clamp_enable: vk::FALSE,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        rasterizer_discard_enable: vk::FALSE,
        depth_bias_clamp: 0.0,
        depth_bias_constant_factor: 0.0,
        depth_bias_enable: vk::FALSE,
        depth_bias_slope_factor: 0.0,
    };

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        p_next: ptr::null(),
        rasterization_samples: msaa_samples,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        p_sample_mask: ptr::null(),
        alpha_to_one_enable: vk::FALSE,
        alpha_to_coverage_enable: vk::FALSE,
    };

    let stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
    };

    let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
        depth_test_enable: vk::TRUE,
        depth_write_enable: vk::TRUE,
        depth_compare_op: vk::CompareOp::LESS,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: stencil_state,
        back: stencil_state,
        max_depth_bounds: 1.0,
        min_depth_bounds: 0.0,
    };

    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        color_write_mask: vk::ColorComponentFlags::all(),
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
    }];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: color_blend_attachment_states.len() as u32,
        p_attachments: color_blend_attachment_states.as_ptr(),
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };

    let set_layouts = [ubo_set_layout];

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineLayoutCreateFlags::empty(),
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Failed to create pipeline layout!")
    };

    let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &vertex_input_state_create_info,
        p_input_assembly_state: &vertex_input_assembly_state_info,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state_create_info,
        p_rasterization_state: &rasterization_statue_create_info,
        p_multisample_state: &multisample_state_create_info,
        p_depth_stencil_state: &depth_state_create_info,
        p_color_blend_state: &color_blend_state,
        p_dynamic_state: ptr::null(),
        layout: pipeline_layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: -1,
    }];

    let graphics_pipelines = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &graphic_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Graphics Pipeline!.")
    };

    unsafe {
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
    }

    (graphics_pipelines[0], pipeline_layout)
}
