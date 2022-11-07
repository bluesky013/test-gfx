#pragma once

#include "TestBase.h"

namespace cc {

class ParticleTest : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(ParticleTest)
    using TestBaseI::TestBaseI;

    bool onInit() override;
    void onTick() override;
    void onDestroy() override;

private:
    void createShader();
    void createVertexBuffer();
    void createPipeline();
    void createInputAssembler();
    void createTexture();

    IntrusivePtr<gfx::Shader>              _shader;
    IntrusivePtr<gfx::Buffer>              _vertexBuffer;
    IntrusivePtr<gfx::Buffer>              _indexBuffer;
    IntrusivePtr<gfx::Buffer>              _uniformBuffer;
    IntrusivePtr<gfx::PipelineState>       _pipelineState;
    IntrusivePtr<gfx::InputAssembler>      _inputAssembler;
    IntrusivePtr<gfx::DescriptorSet>       _descriptorSet;
    IntrusivePtr<gfx::DescriptorSetLayout> _descriptorSetLayout;
    IntrusivePtr<gfx::PipelineLayout>      _pipelineLayout;

#define MAX_QUAD_COUNT 1024
#define VERTEX_STRIDE  9
#define PARTICLE_COUNT 100
    float    _vbufferArray[MAX_QUAD_COUNT][4][VERTEX_STRIDE];
    uint16_t _ibufferArray[MAX_QUAD_COUNT][6];

    struct ParticleData {
        Vec3  position;
        Vec3  velocity;
        float age;
        float life;
    };

    Mat4         _matrices[3];
    ParticleData _particles[PARTICLE_COUNT];
};

} // namespace cc
