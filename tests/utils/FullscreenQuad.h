#pragma once

#include "../TestBase.h"
#include "base/Macros.h"
#include "gfx-base/GFXCommandBuffer.h"

namespace cc {

class FullscreenQuad : public cc::CCObject {
public:
    FullscreenQuad(gfx::Device *device, gfx::RenderPass *renderPass, gfx::Texture *texture);
    ~FullscreenQuad() override;
    CC_DISALLOW_COPY_MOVE_ASSIGN(FullscreenQuad)

    void draw(gfx::CommandBuffer *commandBuffer);

private:
    IntrusivePtr<gfx::Shader>              _shader;
    IntrusivePtr<gfx::InputAssembler>      _inputAssembler;
    IntrusivePtr<gfx::DescriptorSetLayout> _descriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _pipelineLayout;
    IntrusivePtr<gfx::PipelineState>       _pipelineState;
    IntrusivePtr<gfx::DescriptorSet>       _descriptorSet;
    IntrusivePtr<gfx::Buffer>              _vertexBuffer;
};

} // namespace cc
