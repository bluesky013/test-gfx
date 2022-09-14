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

    gfx::Shader *_shader          = nullptr;
    gfx::Buffer *_vertexBuffer    = nullptr;
    gfx::Buffer *_uniformBufferVP = nullptr;

    struct ViewProjUBO {
        Mat4 matViewProj;
        Vec4 color;
    };
    ViewProjUBO _uboVP;

    gfx::Buffer *       _uniWorldBuffer = nullptr, *_uniWorldBufferView = nullptr;
    gfx::DescriptorSet *_uniDescriptorSet = nullptr;

    ccstd::vector<gfx::Buffer *>        _worldBuffers;
    ccstd::vector<gfx::DescriptorSet *> _descriptorSets;

    gfx::DescriptorSetLayout *_descriptorSetLayout = nullptr;
    gfx::PipelineLayout *     _pipelineLayout      = nullptr;
    gfx::PipelineState *      _pipelineState       = nullptr;
    gfx::InputAssembler *     _inputAssembler      = nullptr;

    ccstd::vector<gfx::CommandBuffer *> _parallelCBs;

    uint _worldBufferStride = 0U;
    uint _threadCount       = 1U;
};

} // namespace cc
