#pragma once

#include "TestBase.h"
#include "base/Ptr.h"
#include "gfx-base/GFXTransientPool.h"

namespace cc {

struct RasterPass {
    RasterPass()          = default;
    virtual ~RasterPass() = default;

    gfx::Framebuffer              *realFrameBuffer = nullptr;
    IntrusivePtr<gfx::RenderPass>  renderPass;
    IntrusivePtr<gfx::Framebuffer> frameBuffer;
    ccstd::vector<gfx::Color>      clearColors;
    IntrusivePtr<gfx::Texture>     depthStencil;
    gfx::Rect                      renderArea;
    float                          clearDepth   = 1.0;
    uint32_t                       clearStencil = 0.0;

    virtual void execute(gfx::CommandBuffer *commandBuffer, std::function<void(gfx::CommandBuffer*)> &&func);
};

struct FullscreenPass : public RasterPass {
    IntrusivePtr<gfx::Shader>              shader;
    IntrusivePtr<gfx::DescriptorSetLayout> setLayout;
    IntrusivePtr<gfx::PipelineLayout>      pipelineLayout;
    IntrusivePtr<gfx::PipelineState>       pipeline;
    IntrusivePtr<gfx::DescriptorSet>       set;
    IntrusivePtr<gfx::InputAssembler>      ia;

    void execute(gfx::CommandBuffer *commandBuffer, std::function<void(gfx::CommandBuffer*)> &&func) override;
};

struct BloomPass {
    std::unique_ptr<FullscreenPass>  blurPassH;
    std::unique_ptr<FullscreenPass>  blurPassV;
    IntrusivePtr<gfx::DescriptorSet> set;
};

struct ComputePass {
    IntrusivePtr<gfx::Shader>              computeShader;
    IntrusivePtr<gfx::DescriptorSetLayout> setLayout;
    IntrusivePtr<gfx::PipelineLayout>      pipelineLayout;
    IntrusivePtr<gfx::PipelineState>       computePipeline;
    IntrusivePtr<gfx::DescriptorSet>       set;
    gfx::DispatchInfo dispatchInfo;

    void execute(gfx::CommandBuffer *commandBuffer);
};

class TransientPoolTest : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(TransientPoolTest)
    using TestBaseI::TestBaseI;
    ~TransientPoolTest() override = default;

    bool onInit() override;
    void onTick() override;
    void onDestroy() override;

    void onSpacePressed() override;

protected:
    struct ParticleData {
        Vec4 pos;
        Vec4 vel;
    };

    struct ParticleRenderData {
        Vec4 pos;
    };

    struct  FrameData {
        float time  = 0;
        float delta = 0;
    };

    static constexpr uint32_t POOL_SIZE = 1024 * 4;
    struct ParticleSystem {
        IntrusivePtr<gfx::Shader>              shader;
        IntrusivePtr<gfx::DescriptorSetLayout> setLayout;
        IntrusivePtr<gfx::PipelineLayout>      pipelineLayout;
        IntrusivePtr<gfx::PipelineState>       pipeline;
        IntrusivePtr<gfx::InputAssembler>      ia;
        IntrusivePtr<gfx::DescriptorSet>       set;

        IntrusivePtr<gfx::Buffer> particleBuffer;
        IntrusivePtr<gfx::Buffer> particleRenderBuffer;

        void execute(gfx::CommandBuffer *commandBuffer);
    };

    void prepareParticleSystem();
    void prepareParticleTexture();
    void prepareParticlePipeline();
    void prepareComputePass1();
    void prepareComputePass2();
    void resetParticlePool();
    void prepareColorPass();
    void prepareBlurPass();
    void preparePresentPass();

    void updateTransientResourceParticleRenderBuffer();
    void updateTransientResourceParticleStorageTexture();
    void updateTransientResourceBlurImage();

    using TexturePtr = IntrusivePtr<gfx::Texture>;
    enum class TextureId : uint32_t {
        STORAGE_PARTICLE_IMAGE,
        COLOR_PASS_IMAGE0,
        COLOR_PASS_IMAGE1,
        BLUR_IMAGE0,
        BLUR_IMAGE1
    };

    FrameData                                     _frameData{};
    IntrusivePtr<gfx::Buffer>                     _ubo;
    IntrusivePtr<gfx::TransientPool>              _transientPool;
    ccstd::unordered_map<TextureId, TexturePtr>   _textureMap;
    gfx::Sampler                                 *_sampler = nullptr;
    std::unique_ptr<ParticleSystem>               _particleSystem;
    std::unique_ptr<ComputePass>                  _computePass1;
    std::unique_ptr<ComputePass>                  _computePass2;
    std::unique_ptr<RasterPass>                   _colorPass;
    std::unique_ptr<FullscreenPass>               _presentPass;
    BloomPass                                     _blurGroup;
};

} // namespace cc
