#include "TransientPoolTest.h"
#include "tiny_obj_loader.h"
#include "utils/ShaderUtil.h"
#include "base/StringUtil.h"

namespace cc {
/**
 *  |Compute1|Compute2|Color|Blur|Composite|
 *  |--ParticleRenderData---|
 *           |ParticleImage-|
 *                    |-------Color0-------|
 *                    |--Color1--|
 *                          |Blr0|
 *                          |-----Blr1-----|
 */
gfx::AttributeList STANDARD_VERTEX_ATTRIBUTES{
    {"inPos", gfx::Format::RGBA32F, false, 0, true, 0},
};

template <typename T>
static gfx::Buffer *createBuffer(gfx::Device *device, const std::vector<T> &data, gfx::BufferUsage usage, uint32_t stride) {
    uint32_t size   = static_cast<uint32_t>(data.size() * sizeof(T));
    gfx::BufferInfo bufferInfo = {};
    bufferInfo.usage = usage;
    bufferInfo.memUsage = gfx::MemoryUsage::DEVICE;
    bufferInfo.size = size;
    bufferInfo.stride = stride;

    auto buffer = device->createBuffer(bufferInfo);
    buffer->update(data.data(), size);
    return buffer;
}

static gfx::Buffer *createBuffer(gfx::Device *device, gfx::BufferUsage usage, gfx::MemoryUsage mem, uint32_t size, uint32_t stride) {
    gfx::BufferInfo bufferInfo = {};
    bufferInfo.usage = usage;
    bufferInfo.memUsage = mem;
    bufferInfo.size = size;
    bufferInfo.stride = stride;

    auto buffer = device->createBuffer(bufferInfo);
    return buffer;
}

static gfx::Buffer *createTransientBuffer(gfx::TransientPool *pool, gfx::BufferUsage usage, gfx::MemoryUsage mem, uint32_t size, uint32_t stride) {
    gfx::BufferInfo bufferInfo = {};
    bufferInfo.usage = usage;
    bufferInfo.memUsage = mem;
    bufferInfo.size = size;
    bufferInfo.stride = stride;
    bufferInfo.flags = gfx::BufferFlagBit::TRANSIENT;
    return pool->requestBuffer(bufferInfo);
}

static constexpr uint32_t IMAGE_LOCAL_GROUP = 32;
static constexpr uint32_t IMAGE_EXTENT = 128;
static constexpr uint32_t PARTICLE_LOCAL_GROUP = 256;

const std::string FULL_SCREEN_VS = StringUtil::format(R"(
    layout(location = 0) out vec2 outUv;

    vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2(-1.0,  3.0),
    vec2( 3.0, -1.0)
    );

    vec2 uv[3] = vec2[](
    vec2(0.0, 0.0),
    vec2(0.0, 2.0),
    vec2(2.0, 0.0)
    );

    void main() {
        gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        outUv = uv[gl_VertexIndex];
    })");

const std::string VS = StringUtil::format(R"(
    layout(location = 0) in vec4 inPos;
    layout(location = 0) out vec2 outUv;
    #define PARTICLE_SCALE 0.015
    vec2 positions[6] = vec2[](
    vec2(-PARTICLE_SCALE, -PARTICLE_SCALE),
    vec2( PARTICLE_SCALE, -PARTICLE_SCALE),
    vec2( PARTICLE_SCALE,  PARTICLE_SCALE),
    vec2( PARTICLE_SCALE,  PARTICLE_SCALE),
    vec2(-PARTICLE_SCALE,  PARTICLE_SCALE),
    vec2(-PARTICLE_SCALE, -PARTICLE_SCALE)
    );

    vec2 uv[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),

    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0)
    );

    void main() {
        vec2 pos = positions[gl_VertexIndex];
        float angle = inPos.z;
        vec2 offset = inPos.xy;

        mat2 rot = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
        vec2 outPos = rot * pos;

        gl_Position = vec4(outPos + offset, 0.0, 1.0);
        outUv = uv[gl_VertexIndex];
    })");

const std::string FS = StringUtil::format(R"(
    layout (location = 0) in vec2 inUv;
    layout (location = 0) out vec4 outFragColor1;
    layout (location = 1) out vec4 outFragColor2;
    layout (set = 0, binding = 0) uniform sampler2D sampler0;
    void main() {
        outFragColor1 = texture(sampler0, inUv);
        float brightness = dot(outFragColor1.rgb, vec3(0.2126, 0.7152, 0.0722));
        outFragColor2 = brightness > 0.5 ? outFragColor1 : vec4(0, 0, 0, 0);
    })");

const std::string COMPOSITE_FS = StringUtil::format(R"(
    layout (location = 0) in vec2 inUv;
    layout (location = 0) out vec4 outFragColor;
    layout (set = 0, binding = 0) uniform sampler2D sampler0;
    layout (set = 0, binding = 1) uniform sampler2D sampler1;
    void main() {
        outFragColor = texture(sampler0, inUv) + texture(sampler1, inUv);
    })");

const char* BLUR_DATA = R"(
layout (location = 0) in vec2 inUv;
layout (location = 0) out vec4 outFragColor;
layout (set = 0, binding = 0) uniform sampler2D sampler0;
#define VERTICAL %u
void main() {
    float weight[5];
    weight[0] = 0.227027;
    weight[1] = 0.1945946;
    weight[2] = 0.1216216;
    weight[3] = 0.054054;
    weight[4] = 0.016216;

    vec2 offset = 1.0 / textureSize(sampler0, 0);
    vec4 tex = texture(sampler0, inUv);
    vec3 res = tex.rgb * weight[0];
    for (int i = 1; i < 5; ++i) {
#if(VERTICAL)
        res += texture(sampler0, inUv + vec2(offset.y * i, 0.0)).rgb * weight[i];
        res += texture(sampler0, inUv - vec2(offset.y * i, 0.0)).rgb * weight[i];
#else
        res += texture(sampler0, inUv + vec2(offset.x * i, 0.0)).rgb * weight[i];
        res += texture(sampler0, inUv - vec2(offset.x * i, 0.0)).rgb * weight[i];
#endif
    }
    outFragColor = vec4(res, tex.a);
})";

const std::string BLUR_FS_H = StringUtil::format(BLUR_DATA, 0);

const std::string BLUR_FS_V = StringUtil::format(BLUR_DATA, 1);

const std::string CS1 = StringUtil::format(R"(
    layout (local_size_x = %u, local_size_y = 1, local_size_z = 1) in;
    struct Particle {
        vec4 pos;
        vec4 vel;
    };

    struct ParticleRender {
        vec4 pos;
    };

    layout(set = 0, binding = 0) uniform FrameData {
        float time;
        float delta;
    };

    layout(set = 0, binding = 1) buffer ParticleData {
        Particle particlesIn[];
    };

    layout(set = 0, binding = 2) writeonly buffer ParticleRenderData {
        ParticleRender particlesOut[];
    };

    void main() {
        particlesIn[gl_GlobalInvocationID.x].pos = particlesIn[gl_GlobalInvocationID.x].pos + particlesIn[gl_GlobalInvocationID.x].vel * vec4(delta);
        particlesIn[gl_GlobalInvocationID.x].vel = particlesIn[gl_GlobalInvocationID.x].vel + vec4(0, 0.5 * delta, 0.0, 0.0);

        vec2 dir = normalize(particlesIn[gl_GlobalInvocationID.x].vel.xy);
        float val = atan(dir.y, dir.x);

        particlesOut[gl_GlobalInvocationID.x].pos = vec4(particlesIn[gl_GlobalInvocationID.x].pos.xy, val, 0.0);
    })", PARTICLE_LOCAL_GROUP);

const std::string CS2 = StringUtil::format(R"(
    #define LOCAL_GROUP %u
    layout (local_size_x = LOCAL_GROUP, local_size_y = LOCAL_GROUP, local_size_z = 1) in;
    layout(set = 0, binding = 0) uniform FrameData {
        float time;
        float delta;
    };
    layout(set = 0, binding = 1, rgba8) uniform writeonly image2D resultImage;

    void main() {
        const float IMAGE_EXT = %u;
        float u = gl_GlobalInvocationID.x / IMAGE_EXT * 2.f - 1.f;
        float v = gl_GlobalInvocationID.y / IMAGE_EXT * 2.f - 1.f;
        float v1 = u - 4 * v - 1;
        float v2 = u + 4 * v - 1;
        bool val = v1 <= 0 && v2 <= 0;
        float timeScale = sin(time * 5.0) / 2.0 + 1.0;
        vec4 color = val ? vec4(0, timeScale, 0, 1) : vec4(0, 0, 0, 0);
        imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), color);
    })", IMAGE_LOCAL_GROUP, IMAGE_EXTENT);

bool TransientPoolTest::onInit() {
    gfx::TransientPoolInfo poolInfo = {};
    poolInfo.blockSize = 64 * 1024 * 1024;
    _transientPool.reset(device->createTransientPool(poolInfo));

    _ubo = createBuffer(device,
                        gfx::BufferUsageBit::UNIFORM, gfx::MemoryUsageBit::DEVICE | gfx::MemoryUsageBit::HOST,
                        sizeof(FrameData),
                        sizeof(FrameData));

    gfx::SamplerInfo samplerInfo;
    samplerInfo.addressU  = gfx::Address::CLAMP;
    samplerInfo.addressV  = gfx::Address::CLAMP;
    samplerInfo.mipFilter = gfx::Filter::LINEAR;
    samplerInfo.magFilter = gfx::Filter::LINEAR;
    _sampler = device->getSampler(samplerInfo);

    prepareColorPass();
    prepareBlurPass();
    preparePresentPass();

    prepareParticleSystem();
    prepareParticleTexture();
    prepareParticlePipeline();
    return true;
}

void TransientPoolTest::onTick() {
    auto *swapchain = swapchains[0];

    auto                                         now     = std::chrono::steady_clock::now();
    static std::chrono::steady_clock::time_point current = std::chrono::steady_clock::now();
    _frameData.delta                                     = std::chrono::duration<float, std::milli>(now - current).count() / 1000.f;
    if (_frameData.delta < 0) _frameData.delta = 1 / 60.f;
    _frameData.time += _frameData.delta;
    current = now;
    _ubo->update(&_frameData, sizeof(FrameData));

    //    Mat4 matrices = {};
    //    gfx::Extent orientedSize = TestBaseI::getOrientedSurfaceSize(swapchain);
    //    TestBaseI::createPerspective(60.0F,
    //                                 static_cast<float>(orientedSize.width) / static_cast<float>(orientedSize.height),
    //                                 0.01F, 1000.0F, &matrices, swapchain);

    device->acquire(&swapchain, 1);

    auto *commandBuffer = commandBuffers[0];
    commandBuffer->begin();

    // update transient resource.
    {
        updateTransientResourceParticleRenderBuffer();
        updateTransientResourceParticleStorageTexture();
        updateTransientResourceBlurImage();
    }

    // compute pass
    _computePass1->execute(commandBuffer);
    {
        std::vector<gfx::Texture *>        textures        = {_textureMap[TextureId::STORAGE_PARTICLE_IMAGE]};
        std::vector<gfx::TextureBarrier *> textureBarriers = {device->getTextureBarrier({
            gfx::AccessFlagBit::NONE,
            gfx::AccessFlagBit::COMPUTE_SHADER_WRITE,
        })};
        commandBuffer->pipelineBarrier(nullptr, {}, {}, textureBarriers, textures);
        _computePass2->execute(commandBuffer);
    }

    // color pass
    {
        std::vector<gfx::Buffer *>         buffers         = {_particleSystem->particleRenderBuffer};
        std::vector<gfx::Texture *>        textures        = {_textureMap[TextureId::STORAGE_PARTICLE_IMAGE]};
        std::vector<gfx::TextureBarrier *> textureBarriers = {device->getTextureBarrier({
            gfx::AccessFlagBit::COMPUTE_SHADER_WRITE,
            gfx::AccessFlagBit::FRAGMENT_SHADER_READ_OTHER,
        })};
        std::vector<gfx::BufferBarrier *>  bufferBarriers  = {device->getBufferBarrier({
            gfx::AccessFlagBit::COMPUTE_SHADER_WRITE,
            gfx::AccessFlagBit::VERTEX_BUFFER,
            gfx::BarrierType::FULL,
            0, _particleSystem->particleRenderBuffer->getSize()
        })};
        commandBuffer->pipelineBarrier(nullptr, bufferBarriers, buffers, textureBarriers, textures);

        _particleSystem->set->bindTexture(0, _textureMap[TextureId::STORAGE_PARTICLE_IMAGE]);
        _particleSystem->set->bindSampler(0, _sampler);
        _particleSystem->set->update();

        if (_particleSystem->ia) {
            _particleSystem->ia->updateVertexBuffer(0, _particleSystem->particleRenderBuffer);
        } else {
            gfx::InputAssemblerInfo inputAssemblerInfo;
            inputAssemblerInfo.attributes = STANDARD_VERTEX_ATTRIBUTES;
            inputAssemblerInfo.vertexBuffers.emplace_back(_particleSystem->particleRenderBuffer);
            _particleSystem->ia = device->createInputAssembler(inputAssemblerInfo);
        }

        _colorPass->execute(commandBuffer, [this](gfx::CommandBuffer *commandBuffer) {
            _particleSystem->execute(commandBuffer);
        });
    }

    // blur pass
    {
        gfx::Texture   *tex[2]  = {_textureMap[TextureId::BLUR_IMAGE0], _textureMap[TextureId::BLUR_IMAGE1]};
        FullscreenPass *pass[2] = {_blurGroup.blurPassH.get(), _blurGroup.blurPassV.get()};
        for (uint32_t i = 0; i < 10; ++i) {
            gfx::Texture *input  = i == 0 ? _textureMap[TextureId::COLOR_PASS_IMAGE1].get() : tex[(i - 1) % 2];
            gfx::Texture *output = tex[i % 2];

            std::vector<gfx::Texture *>        textures         = {input, output};
            std::vector<gfx::TextureBarrier *> textureBarriers = {
                device->getTextureBarrier({
                    gfx::AccessFlagBit::COLOR_ATTACHMENT_WRITE,
                    gfx::AccessFlagBit::FRAGMENT_SHADER_READ_TEXTURE,
                }),
                device->getTextureBarrier({
                    i < 2 ? gfx::AccessFlagBit::NONE : gfx::AccessFlagBit::FRAGMENT_SHADER_READ_TEXTURE,
                    gfx::AccessFlagBit::COLOR_ATTACHMENT_WRITE,
                })};
            commandBuffer->pipelineBarrier(nullptr, {}, {}, textureBarriers, textures);
            auto fn = [this](gfx::CommandBuffer *cmd) {
                cmd->bindDescriptorSet(0, _blurGroup.set);
            };

            if (i == 0) {
                pass[i % 2]->execute(commandBuffer, fn);
            } else {
                pass[i % 2]->execute(commandBuffer, {});
            }
        }
    }

    // composite
    {
        std::vector<gfx::Texture *> textures = {
            _textureMap[TextureId::COLOR_PASS_IMAGE0],
            _textureMap[TextureId::BLUR_IMAGE1]};
        std::vector<gfx::TextureBarrier *> textureBarriers = {
            device->getTextureBarrier({
            gfx::AccessFlagBit::COLOR_ATTACHMENT_WRITE,
            gfx::AccessFlagBit::FRAGMENT_SHADER_READ_TEXTURE,
            }),
            device->getTextureBarrier({
            gfx::AccessFlagBit::COLOR_ATTACHMENT_WRITE,
            gfx::AccessFlagBit::FRAGMENT_SHADER_READ_TEXTURE,
            })};
        commandBuffer->pipelineBarrier(nullptr, {}, {}, textureBarriers, textures);
        _presentPass->set->bindTexture(0, _textureMap[TextureId::COLOR_PASS_IMAGE0]);
        _presentPass->set->bindSampler(0, _sampler);
        _presentPass->set->bindTexture(1, _textureMap[TextureId::BLUR_IMAGE1]);
        _presentPass->set->bindSampler(1, _sampler);
        _presentPass->set->update();

        _presentPass->execute(commandBuffer, {});
    }
    commandBuffer->end();

    device->flushCommands(commandBuffers);
    device->getQueue()->submit(commandBuffers);
    device->present();
}

void TransientPoolTest::onDestroy() {
    _ubo = nullptr;
    _transientPool = nullptr;
    _textureMap.clear();
    _particleSystem = nullptr;
    _computePass1 = nullptr;
    _computePass2 = nullptr;
    _colorPass = nullptr;
    _presentPass = nullptr;
    _blurGroup = {};
}

void TransientPoolTest::onSpacePressed() {
    resetParticlePool();
}

void TransientPoolTest::resetParticlePool() {
    std::vector<ParticleData> particles(POOL_SIZE);

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::normal_distribution<float> distrib(100, 25);

    for (uint32_t i = 0; i < POOL_SIZE; ++i) {
        particles[i].pos.x = 0;
        particles[i].pos.y = 0;

        float angle        = (i / static_cast<float>(POOL_SIZE) * 2 - 1.F) * 3.14F / 8.F + (90 / 180.F * 3.14F);
        particles[i].vel.x = distrib(gen) / 120.0f * cos(angle);
        particles[i].vel.y = -distrib(gen) / 120.0f * sin(angle);
    }
    _particleSystem->particleBuffer->update(particles.data(), particles.size() * sizeof(ParticleData));
}

void TransientPoolTest::prepareParticleSystem() {
    prepareComputePass1();

    _particleSystem = std::make_unique<ParticleSystem>();
    // prepare buffer
    _particleSystem->particleBuffer = createBuffer(device,
                                                   gfx::BufferUsageBit::STORAGE,
                                                   gfx::MemoryUsageBit::DEVICE,
                                                   POOL_SIZE * sizeof(ParticleData),
                                                   sizeof(ParticleData));
    resetParticlePool();

    _computePass1->dispatchInfo.groupCountX = POOL_SIZE / PARTICLE_LOCAL_GROUP;
    _computePass1->dispatchInfo.groupCountY = 1;
    _computePass1->dispatchInfo.groupCountZ = 1;
};

void TransientPoolTest::prepareParticleTexture() {
    prepareComputePass2();

    _computePass2->dispatchInfo.groupCountX = IMAGE_EXTENT / IMAGE_LOCAL_GROUP;
    _computePass2->dispatchInfo.groupCountY = IMAGE_EXTENT / IMAGE_LOCAL_GROUP;
    _computePass2->dispatchInfo.groupCountZ = 1;
}

void TransientPoolTest::prepareParticlePipeline() {
    gfx::ShaderStageList shaderLists = {
        {gfx::ShaderStageFlagBit::VERTEX, VS},
        {gfx::ShaderStageFlagBit::FRAGMENT, FS},
    };

    gfx::ShaderInfo shaderInfo = {};
    shaderInfo.attributes      = STANDARD_VERTEX_ATTRIBUTES;
    _particleSystem->shader    = createShader(shaderInfo, device, shaderLists);

    gfx::DescriptorSetLayoutInfo dslInfo;
    dslInfo.bindings.push_back({0, gfx::DescriptorType::SAMPLER_TEXTURE, 1, gfx::ShaderStageFlagBit::FRAGMENT});
    _particleSystem->setLayout      = device->createDescriptorSetLayout(dslInfo);
    _particleSystem->pipelineLayout = device->createPipelineLayout({{_particleSystem->setLayout}});

    gfx::PipelineStateInfo pipelineInfo;
    pipelineInfo.primitive                           = gfx::PrimitiveMode::TRIANGLE_LIST;
    pipelineInfo.shader                              = _particleSystem->shader;
    pipelineInfo.inputState                          = {STANDARD_VERTEX_ATTRIBUTES};
    pipelineInfo.renderPass                          = _colorPass->realFrameBuffer->getRenderPass();
    pipelineInfo.rasterizerState.cullMode            = gfx::CullMode::NONE;
    pipelineInfo.depthStencilState.depthTest         = false;
    pipelineInfo.depthStencilState.depthWrite        = false;
    pipelineInfo.blendState.targets[0].blend         = true;
    pipelineInfo.blendState.targets[0].blendEq       = gfx::BlendOp::ADD;
    pipelineInfo.blendState.targets[0].blendAlphaEq  = gfx::BlendOp::ADD;
    pipelineInfo.blendState.targets[0].blendSrc      = gfx::BlendFactor::SRC_ALPHA;
    pipelineInfo.blendState.targets[0].blendDst      = gfx::BlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineInfo.blendState.targets[0].blendSrcAlpha = gfx::BlendFactor::ONE;
    pipelineInfo.blendState.targets[0].blendDstAlpha = gfx::BlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineInfo.pipelineLayout                      = _particleSystem->pipelineLayout.get();
    _particleSystem->pipeline                        = device->createPipelineState(pipelineInfo);

    _particleSystem->set = device->createDescriptorSet({_particleSystem->setLayout});
}

void TransientPoolTest::prepareComputePass1() {
    _computePass1 = std::make_unique<ComputePass>();

    gfx::ShaderStageList shaderLists = {
        {gfx::ShaderStageFlagBit::COMPUTE, CS1},
    };

    gfx::ShaderInfo shaderInfo = {};
    _computePass1->computeShader = createShader(shaderInfo, device, shaderLists);

    gfx::DescriptorSetLayoutInfo dslInfo;
    dslInfo.bindings.push_back({0, gfx::DescriptorType::UNIFORM_BUFFER, 1, gfx::ShaderStageFlagBit::COMPUTE});
    dslInfo.bindings.push_back({1, gfx::DescriptorType::STORAGE_BUFFER, 1, gfx::ShaderStageFlagBit::COMPUTE});
    dslInfo.bindings.push_back({2, gfx::DescriptorType::STORAGE_BUFFER, 1, gfx::ShaderStageFlagBit::COMPUTE});
    _computePass1->setLayout = device->createDescriptorSetLayout(dslInfo);
    _computePass1->pipelineLayout = device->createPipelineLayout({{_computePass1->setLayout}});
    _computePass1->set = device->createDescriptorSet({_computePass1->setLayout});

    gfx::PipelineStateInfo pipelineStateInfo = {};
    pipelineStateInfo.shader = _computePass1->computeShader;
    pipelineStateInfo.pipelineLayout = _computePass1->pipelineLayout;
    pipelineStateInfo.bindPoint = gfx::PipelineBindPoint::COMPUTE;
    _computePass1->computePipeline = device->createPipelineState(pipelineStateInfo);
}

void TransientPoolTest::updateTransientResourceParticleRenderBuffer() {
    _particleSystem->particleRenderBuffer = createTransientBuffer(_transientPool,
                                                         gfx::BufferUsageBit::STORAGE | gfx::BufferUsageBit::VERTEX,
                                                         gfx::MemoryUsageBit::DEVICE,
                                                         POOL_SIZE * sizeof(ParticleRenderData),
                                                         sizeof(ParticleRenderData));

    _computePass1->set->bindBuffer(0, _ubo);
    _computePass1->set->bindBuffer(1, _particleSystem->particleBuffer);
    _computePass1->set->bindBuffer(2, _particleSystem->particleRenderBuffer);
    _computePass1->set->update();
}

void TransientPoolTest::updateTransientResourceParticleStorageTexture() {
    gfx::TextureInfo textureInfo = {};
    textureInfo.usage = gfx::TextureUsageBit::SAMPLED | gfx::TextureUsageBit::STORAGE;
    textureInfo.format = gfx::Format::RGBA8;
    textureInfo.width = IMAGE_EXTENT;
    textureInfo.height = IMAGE_EXTENT;
    textureInfo.flags = gfx::TextureFlagBit::GENERAL_LAYOUT | gfx::TextureFlagBit::TRANSIENT;
    _textureMap[TextureId::STORAGE_PARTICLE_IMAGE] = _transientPool->requestTexture(textureInfo);

    _computePass2->set->bindBuffer(0, _ubo);
    _computePass2->set->bindTexture(1, _textureMap[TextureId::STORAGE_PARTICLE_IMAGE]);
    _computePass2->set->bindSampler(1, _sampler);
    _computePass2->set->update();
}

void TransientPoolTest::updateTransientResourceBlurImage() {
    _transientPool->resetBuffer(_particleSystem->particleRenderBuffer);
    _transientPool->resetTexture(_textureMap[TextureId::STORAGE_PARTICLE_IMAGE]);

    auto *swapChain = swapchains[0];

    gfx::TextureInfo textureInfo = {};
    textureInfo.width = swapChain->getWidth() / 4;
    textureInfo.height = swapChain->getHeight() / 4;
    textureInfo.format = swapChain->getColorTexture()->getFormat();
    textureInfo.usage = gfx::TextureUsageBit::COLOR_ATTACHMENT | gfx::TextureUsageBit::SAMPLED;
    textureInfo.flags = gfx::TextureFlagBit::TRANSIENT;
    _textureMap[TextureId::BLUR_IMAGE0] = _transientPool->requestTexture(textureInfo);
    _textureMap[TextureId::BLUR_IMAGE1] = _transientPool->requestTexture(textureInfo);

    _blurGroup.set->bindTexture(0, _textureMap[TextureId::COLOR_PASS_IMAGE1]);
    _blurGroup.set->bindSampler(0, _sampler);
    _blurGroup.set->update();
    {
        gfx::FramebufferInfo fbInfo = {};
        fbInfo.renderPass = _blurGroup.blurPassH->renderPass;
        fbInfo.colorTextures.emplace_back(_textureMap[TextureId::BLUR_IMAGE0]);
        _blurGroup.blurPassH->frameBuffer = device->createFramebuffer(fbInfo);
        _blurGroup.blurPassH->realFrameBuffer = _blurGroup.blurPassH->frameBuffer;
        _blurGroup.blurPassH->renderArea = {0, 0, textureInfo.width, textureInfo.height};
        _blurGroup.blurPassH->set->bindTexture(0, _textureMap[TextureId::BLUR_IMAGE1]);
        _blurGroup.blurPassH->set->bindSampler(0, _sampler);
        _blurGroup.blurPassH->set->update();
    }
    {
        gfx::FramebufferInfo fbInfo = {};
        fbInfo.renderPass = _blurGroup.blurPassV->renderPass;
        fbInfo.colorTextures.emplace_back(_textureMap[TextureId::BLUR_IMAGE1]);
        _blurGroup.blurPassV->frameBuffer = device->createFramebuffer(fbInfo);
        _blurGroup.blurPassV->realFrameBuffer = _blurGroup.blurPassV->frameBuffer;
        _blurGroup.blurPassV->renderArea = {0, 0, textureInfo.width, textureInfo.height};
        _blurGroup.blurPassV->set->bindTexture(0, _textureMap[TextureId::BLUR_IMAGE0]);
        _blurGroup.blurPassV->set->bindSampler(0, _sampler);
        _blurGroup.blurPassV->set->update();
    }

    _transientPool->resetTexture(_textureMap[TextureId::BLUR_IMAGE0]);
    _transientPool->resetTexture(_textureMap[TextureId::BLUR_IMAGE1]);
}

void TransientPoolTest::prepareComputePass2() {
    _computePass2 = std::make_unique<ComputePass>();

    gfx::ShaderStageList shaderLists = {
        {gfx::ShaderStageFlagBit::COMPUTE, CS2},
    };

    gfx::ShaderInfo shaderInfo = {};
    _computePass2->computeShader = createShader(shaderInfo, device, shaderLists);

    gfx::DescriptorSetLayoutInfo dslInfo;
    dslInfo.bindings.push_back({0, gfx::DescriptorType::UNIFORM_BUFFER, 1, gfx::ShaderStageFlagBit::COMPUTE});
    dslInfo.bindings.push_back({1, gfx::DescriptorType::STORAGE_IMAGE, 1, gfx::ShaderStageFlagBit::COMPUTE});
    _computePass2->setLayout = device->createDescriptorSetLayout(dslInfo);
    _computePass2->pipelineLayout = device->createPipelineLayout({{_computePass2->setLayout}});
    _computePass2->set = device->createDescriptorSet({_computePass2->setLayout});

    gfx::PipelineStateInfo pipelineStateInfo = {};
    pipelineStateInfo.shader = _computePass2->computeShader;
    pipelineStateInfo.pipelineLayout = _computePass2->pipelineLayout;
    pipelineStateInfo.bindPoint = gfx::PipelineBindPoint::COMPUTE;
    _computePass2->computePipeline = device->createPipelineState(pipelineStateInfo);
}

void TransientPoolTest::prepareColorPass() {
    auto *swapChain = swapchains[0];

    _colorPass = std::make_unique<RasterPass>();

    gfx::RenderPassInfo renderPassInfo;
    renderPassInfo.colorAttachments.emplace_back().format = swapChain->getColorTexture()->getFormat();
    renderPassInfo.colorAttachments.emplace_back().format = swapChain->getColorTexture()->getFormat();
    _colorPass->renderPass = device->createRenderPass(renderPassInfo);

    gfx::TextureInfo textureInfo = {};
    textureInfo.width = swapChain->getWidth();
    textureInfo.height = swapChain->getHeight();
    textureInfo.format = swapChain->getColorTexture()->getFormat();
    textureInfo.usage = gfx::TextureUsageBit::COLOR_ATTACHMENT | gfx::TextureUsageBit::SAMPLED;
    _textureMap[TextureId::COLOR_PASS_IMAGE0] = device->createTexture(textureInfo);
    _textureMap[TextureId::COLOR_PASS_IMAGE1] = device->createTexture(textureInfo);

    gfx::FramebufferInfo fbInfo = {};
    fbInfo.renderPass = _colorPass->renderPass;
    fbInfo.colorTextures.emplace_back(_textureMap[TextureId::COLOR_PASS_IMAGE0]);
    fbInfo.colorTextures.emplace_back(_textureMap[TextureId::COLOR_PASS_IMAGE1]);
    _colorPass->frameBuffer = device->createFramebuffer(fbInfo);
    _colorPass->realFrameBuffer = _colorPass->frameBuffer;

    _colorPass->clearColors.emplace_back(gfx::Color{0.F, 0.F, 0.F, 0.F});
    _colorPass->clearColors.emplace_back(gfx::Color{0.F, 0.F, 0.F, 0.F});
    _colorPass->renderArea = {0, 0, swapChain->getWidth(), swapChain->getHeight()};
}

void TransientPoolTest::prepareBlurPass() {
    auto *swapChain      = swapchains[0];
    _blurGroup.blurPassV = std::make_unique<FullscreenPass>();
    _blurGroup.blurPassH = std::make_unique<FullscreenPass>();

    gfx::ShaderStageList shaderListsH = {
        {gfx::ShaderStageFlagBit::VERTEX, FULL_SCREEN_VS},
        {gfx::ShaderStageFlagBit::FRAGMENT, BLUR_FS_H},
    };

    gfx::ShaderStageList shaderListsV = {
        {gfx::ShaderStageFlagBit::VERTEX, FULL_SCREEN_VS},
        {gfx::ShaderStageFlagBit::FRAGMENT, BLUR_FS_V},
    };

    gfx::ShaderInfo shaderInfo   = {};
    shaderInfo.attributes        = {};
    _blurGroup.blurPassH->shader = createShader(shaderInfo, device, shaderListsH);
    _blurGroup.blurPassV->shader = createShader(shaderInfo, device, shaderListsV);

    gfx::DescriptorSetLayoutInfo dslInfo = {};
    dslInfo.bindings.push_back({0, gfx::DescriptorType::SAMPLER_TEXTURE, 1, gfx::ShaderStageFlagBit::FRAGMENT});
    auto *setLayout                 = device->createDescriptorSetLayout(dslInfo);
    auto *pipelineLayout            = device->createPipelineLayout({{setLayout}});
    _blurGroup.blurPassV->setLayout = _blurGroup.blurPassH->setLayout = setLayout;
    _blurGroup.blurPassV->pipelineLayout = _blurGroup.blurPassH->pipelineLayout = pipelineLayout;

    _blurGroup.blurPassV->set = device->createDescriptorSet({setLayout});
    _blurGroup.blurPassH->set = device->createDescriptorSet({setLayout});
    _blurGroup.set            = device->createDescriptorSet({setLayout});

    gfx::InputAssemblerInfo assemblerInfo = {};

    auto *ia                 = device->createInputAssembler(assemblerInfo);
    _blurGroup.blurPassV->ia = _blurGroup.blurPassH->ia = ia;

    gfx::RenderPassInfo renderPassInfo;
    renderPassInfo.colorAttachments.emplace_back().format = swapChain->getColorTexture()->getFormat();
    auto *pass                                            = device->createRenderPass(renderPassInfo);
    _blurGroup.blurPassV->renderPass = _blurGroup.blurPassH->renderPass = pass;

    gfx::PipelineStateInfo pipelineInfo       = {};
    pipelineInfo.primitive                    = gfx::PrimitiveMode::TRIANGLE_LIST;
    pipelineInfo.inputState                   = {{}};
    pipelineInfo.renderPass                   = pass;
    pipelineInfo.rasterizerState.cullMode     = gfx::CullMode::NONE;
    pipelineInfo.depthStencilState.depthWrite = false;
    pipelineInfo.depthStencilState.depthTest  = false;
    pipelineInfo.pipelineLayout               = pipelineLayout;

    pipelineInfo.shader            = _blurGroup.blurPassV->shader;
    _blurGroup.blurPassV->pipeline = device->createPipelineState(pipelineInfo);

    pipelineInfo.shader            = _blurGroup.blurPassH->shader;
    _blurGroup.blurPassH->pipeline = device->createPipelineState(pipelineInfo);

    _blurGroup.blurPassV->clearColors.emplace_back(gfx::Color{0.F, 0.F, 0.F, 0.F});
    _blurGroup.blurPassH->clearColors.emplace_back(gfx::Color{0.F, 0.F, 0.F, 0.F});
}

void TransientPoolTest::preparePresentPass() {
    auto *swapChain = swapchains[0];
    _presentPass    = std::make_unique<FullscreenPass>();

    gfx::ShaderStageList shaderLists = {
        {gfx::ShaderStageFlagBit::VERTEX, FULL_SCREEN_VS},
        {gfx::ShaderStageFlagBit::FRAGMENT, COMPOSITE_FS},
    };

    gfx::ShaderInfo shaderInfo = {};
    shaderInfo.attributes      = {};
    _presentPass->shader       = createShader(shaderInfo, device, shaderLists);

    gfx::DescriptorSetLayoutInfo dslInfo = {};
    dslInfo.bindings.push_back({0, gfx::DescriptorType::SAMPLER_TEXTURE, 1, gfx::ShaderStageFlagBit::FRAGMENT});
    dslInfo.bindings.push_back({1, gfx::DescriptorType::SAMPLER_TEXTURE, 1, gfx::ShaderStageFlagBit::FRAGMENT});
    _presentPass->setLayout      = device->createDescriptorSetLayout(dslInfo);
    _presentPass->pipelineLayout = device->createPipelineLayout({{_presentPass->setLayout}});
    _presentPass->set            = device->createDescriptorSet({_presentPass->setLayout});

    gfx::InputAssemblerInfo assemblerInfo = {};
    _presentPass->ia                      = device->createInputAssembler(assemblerInfo);

    gfx::PipelineStateInfo pipelineInfo       = {};
    pipelineInfo.primitive                    = gfx::PrimitiveMode::TRIANGLE_LIST;
    pipelineInfo.shader                       = _presentPass->shader;
    pipelineInfo.inputState                   = {{}};
    pipelineInfo.renderPass                   = renderPass;
    pipelineInfo.rasterizerState.cullMode     = gfx::CullMode::NONE;
    pipelineInfo.depthStencilState.depthWrite = false;
    pipelineInfo.depthStencilState.depthTest  = false;
    pipelineInfo.pipelineLayout               = _presentPass->pipelineLayout.get();
    _presentPass->pipeline                    = device->createPipelineState(pipelineInfo);

    _presentPass->realFrameBuffer = fbos[0];
    _presentPass->clearColors.emplace_back(gfx::Color{0.F, 0.F, 0.F, 0.F});
    _presentPass->renderArea = {0, 0, swapChain->getWidth(), swapChain->getHeight()};
}

void RasterPass::execute(gfx::CommandBuffer *commandBuffer, std::function<void(gfx::CommandBuffer*)> &&func) {
    commandBuffer->beginRenderPass(realFrameBuffer->getRenderPass(),
                                   realFrameBuffer,
                                   renderArea,
                                   clearColors.data(),
                                   clearDepth,
                                   clearStencil);

    // draw call
    if (func) {
        func(commandBuffer);
    }
    commandBuffer->endRenderPass();
}

void FullscreenPass::execute(gfx::CommandBuffer *commandBuffer, std::function<void(gfx::CommandBuffer*)> &&func) {
    commandBuffer->beginRenderPass(realFrameBuffer->getRenderPass(),
                                   realFrameBuffer,
                                   renderArea,
                                   clearColors.data(),
                                   clearDepth,
                                   clearStencil);

    commandBuffer->bindPipelineState(pipeline);
//    commandBuffer->bindInputAssembler(ia);

    if (func) {
        func(commandBuffer);
    } else {
        commandBuffer->bindDescriptorSet(0, set);
    }

    gfx::DrawInfo drawInfo = {};
    drawInfo.vertexCount = 3;
    drawInfo.instanceCount = 1;
    commandBuffer->draw(drawInfo);

    commandBuffer->endRenderPass();
}

void ComputePass::execute(gfx::CommandBuffer *commandBuffer) {
    commandBuffer->bindPipelineState(computePipeline);
    commandBuffer->bindDescriptorSet(0, set);
    commandBuffer->dispatch(dispatchInfo);
}

void TransientPoolTest::ParticleSystem::execute(gfx::CommandBuffer *commandBuffer) {
    commandBuffer->bindPipelineState(pipeline);
    commandBuffer->bindInputAssembler(ia);
    commandBuffer->bindDescriptorSet(0, set);

    gfx::DrawInfo drawInfo = {};
    drawInfo.vertexCount = 6;
    drawInfo.firstVertex = 0;
    drawInfo.indexCount  = 0;
    drawInfo.firstIndex  = 0;
    drawInfo.vertexOffset = 0;
    drawInfo.instanceCount = POOL_SIZE;
    drawInfo.firstInstance = 0;
    commandBuffer->draw(drawInfo);
}
} // namespace cc
