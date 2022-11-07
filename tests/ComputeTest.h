#pragma once

#include "TestBase.h"

namespace cc {

class ComputeTest : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(ComputeTest)
    using TestBaseI::TestBaseI;

    bool onInit() override;
    void onTick() override;
    void onDestroy() override;

private:
    void createShader();
    void createUniformBuffer();
    void createPipeline();
    void createInputAssembler();

    void createComputeVBPipeline();
    void createComputeBGPipeline();

    IntrusivePtr<gfx::Shader>              _shader;
    IntrusivePtr<gfx::Buffer>              _uniformBufferMVP;
    IntrusivePtr<gfx::DescriptorSet>       _descriptorSet;
    IntrusivePtr<gfx::DescriptorSetLayout> _descriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _pipelineLayout;
    IntrusivePtr<gfx::PipelineState>       _pipelineState;
    IntrusivePtr<gfx::InputAssembler>      _inputAssembler;

    IntrusivePtr<gfx::Buffer>              _compConstantsBuffer;
    IntrusivePtr<gfx::Buffer>              _compStorageBuffer;
    IntrusivePtr<gfx::Shader>              _compShader;
    IntrusivePtr<gfx::DescriptorSetLayout> _compDescriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _compPipelineLayout;
    IntrusivePtr<gfx::PipelineState>       _compPipelineState;
    IntrusivePtr<gfx::DescriptorSet>       _compDescriptorSet;

    IntrusivePtr<gfx::Shader>              _compBGShader;
    IntrusivePtr<gfx::DescriptorSetLayout> _compBGDescriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _compBGPipelineLayout;
    IntrusivePtr<gfx::PipelineState>       _compBGPipelineState;
    IntrusivePtr<gfx::DescriptorSet>       _compBGDescriptorSet;
};

} // namespace cc
