#pragma once

#include "TestBase.h"

namespace cc {

class StressTest : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(StressTest)
    using TestBaseI::TestBaseI;

    bool onInit() override;
    void onTick() override;
    void onDestroy() override;

protected:
    static const gfx::Color CLEAR_COLORS[];

    void onSpacePressed() override;

    void createShader();
    void createVertexBuffer();
    void createPipeline();
    void createInputAssembler();

    void recordRenderPass(uint jobIdx);

    struct ViewProjUBO {
        Mat4 matViewProj;
        Vec4 color;
    };
    ViewProjUBO _uboVP;

    IntrusivePtr<gfx::Shader>                       _shader;
    IntrusivePtr<gfx::Buffer>                       _vertexBuffer;
    IntrusivePtr<gfx::Buffer>                       _uniformBufferVP;
    IntrusivePtr<gfx::Buffer>                       _uniWorldBuffer;
    IntrusivePtr<gfx::Buffer>                       _uniWorldBufferView;
    IntrusivePtr<gfx::DescriptorSet>                _uniDescriptorSet;
    IntrusivePtr<gfx::DescriptorSetLayout>          _descriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>               _pipelineLayout;
    IntrusivePtr<gfx::PipelineState>                _pipelineState;
    IntrusivePtr<gfx::InputAssembler>               _inputAssembler;
    ccstd::vector<IntrusivePtr<gfx::Buffer>>        _worldBuffers;
    ccstd::vector<IntrusivePtr<gfx::DescriptorSet>> _descriptorSets;
    ccstd::vector<IntrusivePtr<gfx::CommandBuffer>> _parallelCBContainer;
    ccstd::vector<gfx::CommandBuffer*>              _parallelCBs;

    uint _worldBufferStride = 0U;
    uint _threadCount       = 1U;
};

} // namespace cc
