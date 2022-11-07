#pragma once

#include "TestBase.h"

namespace cc {

class StencilTest : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(StencilTest)
    using TestBaseI::TestBaseI;

    bool onInit() override;
    void onTick() override;
    void onDestroy() override;

private:
    void createShader();
    void createBuffers();
    void createTextures();
    void createInputAssembler();
    void createPipelineState();

    const static uint                      BINDING_COUNT = 2;
    const static uint                      PIPELIE_COUNT = 6;
    IntrusivePtr<gfx::Shader>              _shader;
    IntrusivePtr<gfx::Buffer>              _vertexBuffer;
    IntrusivePtr<gfx::InputAssembler>      _inputAssembler;
    IntrusivePtr<gfx::Buffer>              _uniformBuffer[BINDING_COUNT] = {nullptr};
    IntrusivePtr<gfx::DescriptorSet>       _descriptorSet[BINDING_COUNT] = {nullptr};
    IntrusivePtr<gfx::DescriptorSetLayout> _descriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _pipelineLayout;
    IntrusivePtr<gfx::PipelineState>       _pipelineState[PIPELIE_COUNT] = {nullptr};

    struct MatrixUBO {
        Mat4 world;
        Mat4 viewProj;
    };
    MatrixUBO _uboData[BINDING_COUNT];
};

} // namespace cc
