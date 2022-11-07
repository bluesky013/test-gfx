#pragma once

#include "TestBase.h"

namespace cc {

class BasicTexture : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(BasicTexture)
    using TestBaseI::TestBaseI;

    void onTick() override;
    bool onInit() override;
    void onDestroy() override;

private:
    void createShader();
    void createVertexBuffer();
    void createPipeline();
    void createInputAssembler();
    void createTexture();

    ccstd::vector<IntrusivePtr<gfx::Texture>> _textureViews = {};

    IntrusivePtr<gfx::Shader>              _shader;
    IntrusivePtr<gfx::Buffer>              _vertexBuffer;
    IntrusivePtr<gfx::Buffer>              _uniformBuffer;
    IntrusivePtr<gfx::InputAssembler>      _inputAssembler;
    IntrusivePtr<gfx::DescriptorSet>       _descriptorSet;
    IntrusivePtr<gfx::DescriptorSetLayout> _descriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _pipelineLayout;
    IntrusivePtr<gfx::PipelineState>       _pipelineState;

    uint32_t _oldTime = 0;
};

} // namespace cc
