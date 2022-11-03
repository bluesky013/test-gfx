#include "ShaderUtil.h"
#include "gfx-base/SPIRVUtils.h"
#include "gfx-base/GFXDevice.h"
#include "spirv_cross/spirv_msl.hpp"
#include "spirv_cross/spirv_cross.hpp"

namespace cc {
static void buildReflection(gfx::ShaderInfo &out, const std::vector<uint32_t> &spv)
{
    spirv_cross::Compiler compiler(spv.data(), spv.size());
    using SpvResources = spirv_cross::SmallVector<spirv_cross::Resource>;

    auto resourceFn = [&compiler](const SpvResources &resources, std::function<void(uint32_t, uint32_t, const std::string&)> fn) {
        for (auto &res : resources) {
            auto set     = compiler.get_decoration(res.id, spv::DecorationDescriptorSet);
            auto binding = compiler.get_decoration(res.id, spv::DecorationBinding);
            auto &name   = compiler.get_name(res.id);
            fn(set, binding, name);
        }
    };

    auto resources = compiler.get_shader_resources();
    resourceFn(resources.uniform_buffers, [&out](uint32_t set, uint32_t binding, const std::string& name) {
        out.blocks.emplace_back(gfx::UniformBlock{set, binding, name, {}, 1});
    });
    resourceFn(resources.storage_buffers, [&out](uint32_t set, uint32_t binding, const std::string& name){
        out.buffers.emplace_back(gfx::UniformStorageBuffer{set, binding, name, 1});
    });
    resourceFn(resources.sampled_images, [&out](uint32_t set, uint32_t binding, const std::string& name){
        out.samplerTextures.emplace_back(gfx::UniformSamplerTexture{set, binding, name, gfx::Type::SAMPLER2D, 1});
    });
    resourceFn(resources.storage_images, [&out](uint32_t set, uint32_t binding, const std::string& name){
        out.images.emplace_back(gfx::UniformStorageImage{set, binding, name, gfx::Type::SAMPLER2D, 1});
    });
    resourceFn(resources.separate_samplers, [&out](uint32_t set, uint32_t binding, const std::string& name){
        out.samplers.emplace_back(gfx::UniformSampler{set, binding, name, 1});
    });
    resourceFn(resources.separate_images, [&out](uint32_t set, uint32_t binding, const std::string& name){
        out.textures.emplace_back(gfx::UniformTexture{set, binding, name, gfx::Type::TEXTURE2D, 1});
    });
    resourceFn(resources.subpass_inputs, [&out](uint32_t set, uint32_t binding, const std::string& name){
        out.subpassInputs.emplace_back(gfx::UniformInputAttachment{set, binding, name, 1});
    });
}

gfx::Shader *createShader(gfx::ShaderInfo &shaderInfo, gfx::Device *device, const gfx::ShaderStageList &shaderList) {
    gfx::SPIRVUtils *spvUtils = gfx::SPIRVUtils::getInstance();
    std::unordered_map<gfx::ShaderStageFlagBit, std::vector<uint32_t>> spvMap;

//    gfx::ShaderStageFlagBit shaderListOut;
    for (auto &shader : shaderList) {
        std::string source = "#version 450 core \n" + shader.source;
        spvUtils->compileGLSL(shader.stage, source);
        auto &spv = spvMap[shader.stage];
        spv.resize(spvUtils->getOutputSize() / sizeof(uint32_t));
        memcpy(spv.data(), spvUtils->getOutputData(), spvUtils->getOutputSize());

        // relfect
        buildReflection(shaderInfo, spv);
    }

    auto api = device->getGfxAPI();
    if (api == gfx::API::VULKAN || api == gfx::API::METAL) {
        shaderInfo.stages = shaderList;
    }

    return device->createShader(shaderInfo);
}
}
