#pragma once

#include "TestBase.h"

namespace cc {

struct Vertex {
    float pos[2];
    float col[3];
};

class BasicTriangle : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(BasicTriangle)
    using TestBaseI::TestBaseI;

    bool onInit() override;
    void onTick() override;
    void onDestroy() override;

private:
    void createShader();
    void createVertexBuffer();
    void createPipeline();
    void createInputAssembler();

    IntrusivePtr<gfx::Shader>              _shader;
    IntrusivePtr<gfx::Buffer>              _vertexBuffer;
    IntrusivePtr<gfx::Buffer>              _uniformBuffer;
    IntrusivePtr<gfx::Buffer>              _uniformBufferMVP;
    IntrusivePtr<gfx::DescriptorSet>       _descriptorSet;
    IntrusivePtr<gfx::DescriptorSetLayout> _descriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _pipelineLayout;
    IntrusivePtr<gfx::PipelineState>       _pipelineState;
    IntrusivePtr<gfx::PipelineState>       _invisiblePipelineState;
    IntrusivePtr<gfx::InputAssembler>      _inputAssembler;
    IntrusivePtr<gfx::Buffer>              _indirectBuffer;
    IntrusivePtr<gfx::Buffer>              _indexBuffer;
};

} // namespace cc
