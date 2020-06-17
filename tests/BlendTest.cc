#include "BlendTest.h"

NS_CC_BEGIN

namespace {
enum {
  NO_BLEND = 0x0,
  NORMAL_BLEND,
  ADDITIVE_BLEND,
  SUBSTRACT_BLEND,
  MULTIPLY_BLEND,
  TOTAL_BLEND
};

GFXTexture *createTexture(GFXDevice *device, const GFXTextureInfo &textureInfo,
                          std::string imageFile) {
  auto img = new cocos2d::Image();
  img->autorelease();
  bool valid = img->initWithImageFile(imageFile);
  CCASSERT(valid, "load image failed");
  auto imgData = img->getRenderFormat() == GFXFormat::RGB8
                     ? TestBaseI::RGB2RGBA(img)
                     : img->getData();

  auto texture = device->createTexture(textureInfo);

  GFXBufferTextureCopy textureRegion;
  textureRegion.texExtent.width = img->getWidth();
  textureRegion.texExtent.height = img->getHeight();
  textureRegion.texExtent.depth = 1;

  GFXBufferTextureCopyList regions;
  regions.push_back(std::move(textureRegion));

  GFXDataArray imageBuffers = {{imgData}};
  device->copyBuffersToTexture(imageBuffers, texture, regions);
  delete[] imgData;
  return texture;
}

struct Quad : public Object {
  Quad(GFXDevice *_device, GFXFramebuffer *_fbo) : device(_device), fbo(_fbo) {
    createShader();
    createVertexBuffer();
    createInputAssembler();
    createTexture();
    createPipeline();
  }

  ~Quad() {}

  void destroy() {
    CC_SAFE_DESTROY(shader);
    CC_SAFE_DESTROY(vertexBuffer);
    CC_SAFE_DESTROY(inputAssembler);
    CC_SAFE_DESTROY(indexBuffer);
    CC_SAFE_DESTROY(texture);
    CC_SAFE_DESTROY(sampler);
    for (int i = 0; i < TOTAL_BLEND; i++) {
      CC_SAFE_DESTROY(pipelineState[i]);
      CC_SAFE_DESTROY(pipelineLayout[i]);
      CC_SAFE_DESTROY(bindingLayout[i]);
      CC_SAFE_DESTROY(uniformBuffer[i]);
    }
  }

  void createShader() {
    GFXShaderStageList shaderStageList;
    GFXShaderStage vertexShaderStage;
    vertexShaderStage.type = GFXShaderType::VERTEX;

    //#if (CC_PLATFORM == CC_PLATFORM_MAC_OSX)
    //            vertexShaderStage.source = R"(
    //            #include <metal_stdlib>
    //            #include <simd/simd.h>
    //
    //            using namespace metal;
    //
    //            struct MVP_Matrix
    //            {
    //                float4x4 u_model;
    //                float4x4 u_projection;
    //            };
    //
    //            struct main0_out
    //            {
    //                float2 uv [[user(locn0)]];
    //                float4 gl_Position [[position]];
    //            };
    //
    //            struct main0_in
    //            {
    //                float2 a_position [[attribute(0)]];
    //                float2 a_uv [[attribute(1)]];
    //            };
    //
    //            vertex main0_out main0(main0_in in [[stage_in]], constant
    //            MVP_Matrix& _22 [[buffer(0)]])
    //            {
    //                main0_out out = {};
    //                out.uv = in.a_uv;
    //                out.gl_Position = (_22.u_projection * _22.u_model) *
    //                float4(in.a_position, 0.0, 1.0); return out;
    //            }
    //            )";
    //#else
#if defined(USE_VULKAN) || defined(USE_METAL)
    vertexShaderStage.source = R"(
            #ifdef GL_ES
                        precision highp float;
            #endif
            layout(location = 0) in vec2 a_position;
            layout(location = 1) in vec2 a_uv;
            layout(location = 0) out vec2 uv;
            layout(binding = 0) uniform MVP_Matrix
            {
                mat4 u_model, u_projection;
            };
            void main()
            {
                uv = a_uv;
                gl_Position = u_projection * u_model * vec4(a_position, 0, 1);
            }
            )";
#elif defined(USE_GLES2)
    vertexShaderStage.source = R"(
            attribute vec2 a_position;
            attribute vec2 a_uv;
            varying vec2 uv;
            uniform mat4 u_model, u_projection;
            
            void main()
            {
                uv = a_uv;
                gl_Position = u_projection * u_model * vec4(a_position, 0, 1);
            }
            )";
#else
    vertexShaderStage.source = R"(
            in vec2 a_position;
            in vec2 a_uv;
            out vec2 uv;
            layout(std140) uniform MVP_Matrix
            {
                mat4 u_model, u_projection;
            };
            void main()
            {
                uv = a_uv;
                gl_Position = u_projection * u_model * vec4(a_position, 0, 1);
            }
            )";
#endif // USE_GLES2
       //#endif // (CC_PLATFORM == CC_PLATFORM_MAC_OSX)

    shaderStageList.emplace_back(std::move(vertexShaderStage));

    GFXShaderStage fragmentShaderStage;
    fragmentShaderStage.type = GFXShaderType::FRAGMENT;
    //#if (CC_PLATFORM == CC_PLATFORM_MAC_OSX)
    //            fragmentShaderStage.source = R"(
    //            #include <metal_stdlib>
    //            #include <simd/simd.h>
    //
    //            using namespace metal;
    //
    //            struct main0_out
    //            {
    //                float4 o_color [[color(0)]];
    //            };
    //
    //            struct main0_in
    //            {
    //                float2 uv [[user(locn0)]];
    //            };
    //
    //            fragment main0_out main0(main0_in in [[stage_in]],
    //            texture2d<float> u_texture [[texture(0)]], sampler
    //            u_textureSmplr [[sampler(0)]])
    //            {
    //                main0_out out = {};
    //                out.o_color = u_texture.sample(u_textureSmplr, in.uv);
    //                return out;
    //            }
    //            )";
    //#else
#if defined(USE_VULKAN) || defined(USE_METAL)
    fragmentShaderStage.source = R"(
            #ifdef GL_ES
                        precision highp float;
            #endif
            layout(location = 0) in vec2 uv;
            layout(binding = 1) uniform sampler2D u_texture;
            layout(location = 0) out vec4 o_color;
            void main()
            {
                o_color = texture(u_texture, uv);
            }
            )";
#elif defined(USE_GLES2)
    fragmentShaderStage.source = R"(
#ifdef GL_ES
            precision highp float;
#endif
            varying vec2 uv;
            uniform sampler2D u_texture;
            
            void main()
            {
                gl_FragColor = texture2D(u_texture, uv);
            }
            )";
#else
    fragmentShaderStage.source = R"(
#ifdef GL_ES
            precision highp float;
#endif
            in vec2 uv;
            uniform sampler2D u_texture;
            out vec4 o_color;
            void main()
            {
                o_color = texture(u_texture, uv);
            }
            )";
#endif // USE_GLES2
       //#endif // (CC_PLATFORM == CC_PLATFORM_MAC_OSX)

    shaderStageList.emplace_back(std::move(fragmentShaderStage));

    GFXUniformList mvpMatrix = {
        {"u_model", GFXType::MAT4, 1},
        {"u_projection", GFXType::MAT4, 1},
    };
    GFXUniformBlockList uniformBlockList = {
        {GFXShaderType::VERTEX, 0, "MVP_Matrix", mvpMatrix}};

    GFXUniformSamplerList samplers = {
        {GFXShaderType::FRAGMENT, 1, "u_texture", GFXType::SAMPLER2D, 1}};

    GFXShaderInfo shaderInfo;
    shaderInfo.name = "Blend Test: Quad";
    shaderInfo.stages = std::move(shaderStageList);
    shaderInfo.blocks = std::move(uniformBlockList);
    shaderInfo.samplers = std::move(samplers);
    shader = device->createShader(shaderInfo);
  }

  void createVertexBuffer() {
    float vertexData[] = {-0.5f, -0.5f, 0,   0,   -0.5f, 0.5f,  0,   1.f,
                          0.5f,  0.5f,  1.f, 1.f, 0.5f,  -0.5f, 1.f, 0};

    unsigned short indices[6] = {0, 3, 1, 1, 3, 2};

    // vertex buffer
    GFXBufferInfo vertexBufferInfo = {
        GFXBufferUsage::VERTEX, GFXMemoryUsage::DEVICE, 4 * sizeof(float),
        sizeof(vertexData), GFXBufferFlagBit::NONE};

    vertexBuffer = device->createBuffer(vertexBufferInfo);
    vertexBuffer->update(vertexData, 0, sizeof(vertexData));

    // index buffer
    GFXBufferInfo indexBufferInfo = {
        GFXBufferUsage::INDEX, GFXMemoryUsage::DEVICE, sizeof(unsigned short),
        sizeof(indices), GFXBufferFlagBit::NONE};
    indexBuffer = device->createBuffer(indexBufferInfo);
    indexBuffer->update(indices, 0, sizeof(indices));

    // uniform buffer
    GFXBufferInfo uniformBufferInfo = {
        GFXBufferUsage::UNIFORM, GFXMemoryUsage::DEVICE, sizeof(Mat4),
        2 * sizeof(Mat4), GFXBufferFlagBit::NONE};

    Mat4 projection;
    Mat4::createOrthographicOffCenter(0.0f, (float)device->getWidth(),
                                      (float)device->getHeight(), 0.0f, 0.0f,
                                      1000.f, &projection);
    TestBaseI::modifyProjectionBasedOnDevice(projection);
    for (int i = 0; i < TOTAL_BLEND; i++) {
      uniformBuffer[i] = device->createBuffer(uniformBufferInfo);
      uniformBuffer[i]->update(projection.m, sizeof(Mat4), sizeof(projection));
    }
  }

  void createInputAssembler() {
    GFXAttribute position = {"a_position", GFXFormat::RG32F, false, 0, false};
    GFXAttribute texcoord = {"a_uv", GFXFormat::RG32F, false, 0, false};
    GFXInputAssemblerInfo inputAssemblerInfo;
    inputAssemblerInfo.attributes.emplace_back(std::move(position));
    inputAssemblerInfo.attributes.emplace_back(std::move(texcoord));
    inputAssemblerInfo.vertexBuffers.emplace_back(vertexBuffer);
    inputAssemblerInfo.indexBuffer = indexBuffer;
    inputAssembler = device->createInputAssembler(inputAssemblerInfo);
  }

  void createTexture() {
    GFXTextureInfo textureInfo;
    textureInfo.usage = GFXTextureUsage::SAMPLED;
    textureInfo.format = GFXFormat::RGBA8;
    textureInfo.width = 128;
    textureInfo.height = 128;
    texture = ::cocos2d::createTexture(device, textureInfo, "sprite0.png");

    // create sampler
    GFXSamplerInfo samplerInfo;
    samplerInfo.addressU = GFXAddress::CLAMP;
    samplerInfo.addressV = GFXAddress::CLAMP;
    sampler = device->createSampler(samplerInfo);
  }

  void createPipeline() {
    GFXBindingList bindingList = {
        {GFXShaderType::VERTEX, 0, GFXBindingType::UNIFORM_BUFFER, "MVP_Matrix",
         1},
        {GFXShaderType::FRAGMENT, 1, GFXBindingType::SAMPLER, "u_texture", 1}};
    GFXBindingLayoutInfo bindingLayoutInfo = {bindingList};

    for (int i = 0; i < TOTAL_BLEND; i++) {
      bindingLayout[i] = device->createBindingLayout(bindingLayoutInfo);

      bindingLayout[i]->bindBuffer(0, uniformBuffer[i]);
      bindingLayout[i]->bindSampler(1, sampler);
      bindingLayout[i]->bindTexture(1, texture);
      bindingLayout[i]->update();

      GFXPipelineLayoutInfo pipelineLayoutInfo;
      pipelineLayoutInfo.layouts = {bindingLayout[i]};
      pipelineLayout[i] = device->createPipelineLayout(pipelineLayoutInfo);
    }

    GFXPipelineStateInfo pipelineInfo[TOTAL_BLEND];
    pipelineInfo[NO_BLEND].primitive = GFXPrimitiveMode::TRIANGLE_LIST;
    pipelineInfo[NO_BLEND].shader = shader;
    pipelineInfo[NO_BLEND].inputState = {inputAssembler->getAttributes()};
    pipelineInfo[NO_BLEND].layout = pipelineLayout[NO_BLEND];
    pipelineInfo[NO_BLEND].renderPass = fbo->getRenderPass();
    pipelineInfo[NO_BLEND].rasterizerState.cullMode = GFXCullMode::NONE;
    pipelineInfo[NO_BLEND].depthStencilState.depthWrite = false;
    pipelineInfo[NO_BLEND].blendState.targets[0].blend = true;
    pipelineInfo[NO_BLEND].blendState.targets[0].blendEq = GFXBlendOp::ADD;
    pipelineInfo[NO_BLEND].blendState.targets[0].blendAlphaEq = GFXBlendOp::ADD;
    pipelineInfo[NO_BLEND].blendState.targets[0].blendSrc = GFXBlendFactor::ONE;
    pipelineInfo[NO_BLEND].blendState.targets[0].blendDst =
        GFXBlendFactor::ZERO;
    pipelineInfo[NO_BLEND].blendState.targets[0].blendSrcAlpha =
        GFXBlendFactor::ONE;
    pipelineInfo[NO_BLEND].blendState.targets[0].blendDstAlpha =
        GFXBlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineState[NO_BLEND] =
        device->createPipelineState(pipelineInfo[NO_BLEND]);

    pipelineInfo[NORMAL_BLEND].primitive = GFXPrimitiveMode::TRIANGLE_LIST;
    pipelineInfo[NORMAL_BLEND].shader = shader;
    pipelineInfo[NORMAL_BLEND].inputState = {inputAssembler->getAttributes()};
    pipelineInfo[NORMAL_BLEND].layout = pipelineLayout[NORMAL_BLEND];
    pipelineInfo[NORMAL_BLEND].renderPass = fbo->getRenderPass();
    pipelineInfo[NORMAL_BLEND].rasterizerState.cullMode = GFXCullMode::NONE;
    pipelineInfo[NORMAL_BLEND].depthStencilState.depthWrite = false;
    pipelineInfo[NORMAL_BLEND].blendState.targets[0].blend = true;
    pipelineInfo[NORMAL_BLEND].blendState.targets[0].blendEq = GFXBlendOp::ADD;
    pipelineInfo[NORMAL_BLEND].blendState.targets[0].blendAlphaEq =
        GFXBlendOp::ADD;
    pipelineInfo[NORMAL_BLEND].blendState.targets[0].blendSrc =
        GFXBlendFactor::SRC_ALPHA;
    pipelineInfo[NORMAL_BLEND].blendState.targets[0].blendDst =
        GFXBlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineInfo[NORMAL_BLEND].blendState.targets[0].blendSrcAlpha =
        GFXBlendFactor::ONE;
    pipelineInfo[NORMAL_BLEND].blendState.targets[0].blendDstAlpha =
        GFXBlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineState[NORMAL_BLEND] =
        device->createPipelineState(pipelineInfo[NORMAL_BLEND]);

    pipelineInfo[ADDITIVE_BLEND].primitive = GFXPrimitiveMode::TRIANGLE_LIST;
    pipelineInfo[ADDITIVE_BLEND].shader = shader;
    pipelineInfo[ADDITIVE_BLEND].inputState = {inputAssembler->getAttributes()};
    pipelineInfo[ADDITIVE_BLEND].layout = pipelineLayout[ADDITIVE_BLEND];
    pipelineInfo[ADDITIVE_BLEND].renderPass = fbo->getRenderPass();
    pipelineInfo[ADDITIVE_BLEND].rasterizerState.cullMode = GFXCullMode::NONE;
    pipelineInfo[ADDITIVE_BLEND].depthStencilState.depthWrite = false;
    pipelineInfo[ADDITIVE_BLEND].blendState.targets[0].blend = true;
    pipelineInfo[ADDITIVE_BLEND].blendState.targets[0].blendEq =
        GFXBlendOp::ADD;
    pipelineInfo[ADDITIVE_BLEND].blendState.targets[0].blendAlphaEq =
        GFXBlendOp::ADD;
    pipelineInfo[ADDITIVE_BLEND].blendState.targets[0].blendSrc =
        GFXBlendFactor::SRC_ALPHA;
    pipelineInfo[ADDITIVE_BLEND].blendState.targets[0].blendDst =
        GFXBlendFactor::ONE;
    pipelineInfo[ADDITIVE_BLEND].blendState.targets[0].blendSrcAlpha =
        GFXBlendFactor::ONE;
    pipelineInfo[ADDITIVE_BLEND].blendState.targets[0].blendDstAlpha =
        GFXBlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineState[ADDITIVE_BLEND] =
        device->createPipelineState(pipelineInfo[ADDITIVE_BLEND]);

    pipelineInfo[SUBSTRACT_BLEND].primitive = GFXPrimitiveMode::TRIANGLE_LIST;
    pipelineInfo[SUBSTRACT_BLEND].shader = shader;
    pipelineInfo[SUBSTRACT_BLEND].inputState = {
        inputAssembler->getAttributes()};
    pipelineInfo[SUBSTRACT_BLEND].layout = pipelineLayout[SUBSTRACT_BLEND];
    pipelineInfo[SUBSTRACT_BLEND].renderPass = fbo->getRenderPass();
    pipelineInfo[SUBSTRACT_BLEND].rasterizerState.cullMode = GFXCullMode::NONE;
    pipelineInfo[SUBSTRACT_BLEND].depthStencilState.depthWrite = false;
    pipelineInfo[SUBSTRACT_BLEND].blendState.targets[0].blend = true;
    pipelineInfo[SUBSTRACT_BLEND].blendState.targets[0].blendEq =
        GFXBlendOp::ADD;
    pipelineInfo[SUBSTRACT_BLEND].blendState.targets[0].blendAlphaEq =
        GFXBlendOp::ADD;
    pipelineInfo[SUBSTRACT_BLEND].blendState.targets[0].blendSrc =
        GFXBlendFactor::ZERO;
    pipelineInfo[SUBSTRACT_BLEND].blendState.targets[0].blendDst =
        GFXBlendFactor::ONE_MINUS_SRC_COLOR;
    pipelineInfo[SUBSTRACT_BLEND].blendState.targets[0].blendSrcAlpha =
        GFXBlendFactor::ONE;
    pipelineInfo[SUBSTRACT_BLEND].blendState.targets[0].blendDstAlpha =
        GFXBlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineState[SUBSTRACT_BLEND] =
        device->createPipelineState(pipelineInfo[SUBSTRACT_BLEND]);

    pipelineInfo[MULTIPLY_BLEND].primitive = GFXPrimitiveMode::TRIANGLE_LIST;
    pipelineInfo[MULTIPLY_BLEND].shader = shader;
    pipelineInfo[MULTIPLY_BLEND].inputState = {inputAssembler->getAttributes()};
    pipelineInfo[MULTIPLY_BLEND].layout = pipelineLayout[MULTIPLY_BLEND];
    pipelineInfo[MULTIPLY_BLEND].renderPass = fbo->getRenderPass();
    pipelineInfo[MULTIPLY_BLEND].rasterizerState.cullMode = GFXCullMode::NONE;
    pipelineInfo[MULTIPLY_BLEND].depthStencilState.depthWrite = false;
    pipelineInfo[MULTIPLY_BLEND].blendState.targets[0].blend = true;
    pipelineInfo[MULTIPLY_BLEND].blendState.targets[0].blendEq =
        GFXBlendOp::ADD;
    pipelineInfo[MULTIPLY_BLEND].blendState.targets[0].blendAlphaEq =
        GFXBlendOp::ADD;
    pipelineInfo[MULTIPLY_BLEND].blendState.targets[0].blendSrc =
        GFXBlendFactor::ZERO;
    pipelineInfo[MULTIPLY_BLEND].blendState.targets[0].blendDst =
        GFXBlendFactor::SRC_COLOR;
    pipelineInfo[MULTIPLY_BLEND].blendState.targets[0].blendSrcAlpha =
        GFXBlendFactor::ONE;
    pipelineInfo[MULTIPLY_BLEND].blendState.targets[0].blendDstAlpha =
        GFXBlendFactor::ONE_MINUS_SRC_ALPHA;
    pipelineState[MULTIPLY_BLEND] =
        device->createPipelineState(pipelineInfo[MULTIPLY_BLEND]);
  }

  GFXDevice *device = nullptr;
  GFXFramebuffer *fbo = nullptr;
  GFXShader *shader = nullptr;
  GFXBuffer *vertexBuffer = nullptr;
  GFXBuffer *indexBuffer = nullptr;
  GFXInputAssembler *inputAssembler = nullptr;
  GFXSampler *sampler = nullptr;
  GFXTexture *texture = nullptr;
  GFXBuffer *uniformBuffer[TOTAL_BLEND] = {nullptr};
  GFXBindingLayout *bindingLayout[TOTAL_BLEND] = {nullptr};
  GFXPipelineLayout *pipelineLayout[TOTAL_BLEND] = {nullptr};
  GFXPipelineState *pipelineState[TOTAL_BLEND] = {nullptr};

  Mat4 model;
};

struct BigTriangle : public Object {
  BigTriangle(GFXDevice *_device, GFXFramebuffer* _fbo) : device(_device), fbo(_fbo) {
    createShader();
    createVertexBuffer();
    createInputAssembler();
    createTexture();
    createPipeline();
  }

  void destroy() {
    CC_SAFE_DESTROY(shader);
    CC_SAFE_DESTROY(vertexBuffer);
    CC_SAFE_DESTROY(inputAssembler);
    CC_SAFE_DESTROY(pipelineState);
    CC_SAFE_DESTROY(pipelineLayout);
    CC_SAFE_DESTROY(bindingLayout);
    CC_SAFE_DESTROY(timeBuffer);
    CC_SAFE_DESTROY(texture);
    CC_SAFE_DESTROY(sampler);
  }

  void createShader() {
    GFXShaderStageList shaderStageList;
    GFXShaderStage vertexShaderStage;
    vertexShaderStage.type = GFXShaderType::VERTEX;
    //#if(CC_PLATFORM == CC_PLATFORM_MAC_OSX)
    //            vertexShaderStage.source = R"(
    //            #include <metal_stdlib>
    //            #include <simd/simd.h>
    //
    //            using namespace metal;
    //
    //            struct main0_out
    //            {
    //                float2 uv [[user(locn0)]];
    //                float4 gl_Position [[position]];
    //            };
    //
    //            struct main0_in
    //            {
    //                float2 a_position [[attribute(0)]];
    //            };
    //
    //            vertex main0_out main0(main0_in in [[stage_in]])
    //            {
    //                main0_out out = {};
    //                out.uv = (in.a_position + float2(1.0)) * 0.5;
    //                out.gl_Position = float4(in.a_position, 0.1, 1.0);
    //                return out;
    //            }
    //            )";
    //#else
#if defined(USE_VULKAN) | defined(USE_METAL)
    vertexShaderStage.source = R"(
            layout(location = 0) in vec2 a_position;
            layout(location = 0) out vec2 uv;
            void main()
            {
                uv = (a_position + 1.0) * 0.5;
                gl_Position = vec4(a_position, 0.1, 1);
            }
            )";
#elif defined(USE_GLES2)
    vertexShaderStage.source = R"(
            attribute vec2 a_position;
            varying vec2 uv;
            void main()
            {
                uv = (a_position + 1.0) * 0.5;
                gl_Position = vec4(a_position, 0.1, 1);
            }
            )";
#else
    vertexShaderStage.source = R"(
            in vec2 a_position;
            out vec2 uv;
            void main()
            {
                uv = (a_position + 1.0) * 0.5;
                gl_Position = vec4(a_position, 0.1, 1);
            }
            )";
#endif // USE_GLES2
       //#endif // (CC_PLATFORM == CC_PLATFORM_MAC_OSX

    shaderStageList.emplace_back(std::move(vertexShaderStage));

    GFXShaderStage fragmentShaderStage;
    fragmentShaderStage.type = GFXShaderType::FRAGMENT;

    //#if(CC_PLATFORM == CC_PLATFORM_MAC_OSX)
    //            fragmentShaderStage.source = R"(
    //            #include <metal_stdlib>
    //            #include <simd/simd.h>
    //
    //            using namespace metal;
    //
    //            struct Time
    //            {
    //                float u_time;
    //            };
    //
    //            struct main0_out
    //            {
    //                float4 o_color [[color(0)]];
    //            };
    //
    //            struct main0_in
    //            {
    //                float2 uv [[user(locn0)]];
    //            };
    //
    //            fragment main0_out main0(main0_in in [[stage_in]], constant
    //            Time& _12 [[buffer(0)]], texture2d<float> u_texture
    //            [[texture(0)]], sampler u_textureSmplr [[sampler(0)]])
    //            {
    //                main0_out out = {};
    //                float2 offset = float2(_12.u_time *
    //                (-0.00999999977648258209228515625)); out.o_color =
    //                u_texture.sample(u_textureSmplr, ((in.uv + offset)
    //                * 20.0)); return out;
    //            }
    //            )";
    //#else
#if defined(USE_VULKAN) | defined(USE_METAL)
    fragmentShaderStage.source = R"(
            #ifdef GL_ES
                        precision highp float;
            #endif
            layout(location = 0) in vec2 uv;
            layout(binding = 0) uniform Time
            {
                float u_time;
            };
            layout(binding = 1) uniform sampler2D u_texture;
            layout(location = 0) out vec4 o_color;
            void main()
            {
                vec2 offset = vec2(u_time * -0.01);
                o_color = texture(u_texture, 20.0 * (uv + offset));
            }
            )";
#elif defined(USE_GLES2)
    fragmentShaderStage.source = R"(
#ifdef GL_ES
            precision highp float;
#endif
            varying vec2 uv;
            uniform sampler2D u_texture;
            uniform float u_time;
            
            void main()
            {
                vec2 offset = vec2(u_time * -0.01);
                gl_FragColor = texture2D(u_texture, 20.0 * (uv + offset));
            }
            )";
#else
    fragmentShaderStage.source = R"(
#ifdef GL_ES
            precision highp float;
#endif
            in vec2 uv;
            uniform sampler2D u_texture;
            layout(std140) uniform Time
            {
                float u_time;
            };
            out vec4 o_color;
            void main()
            {
                vec2 offset = vec2(u_time * -0.01);
                o_color = texture(u_texture, 20.0 * (uv + offset));
            }
            )";
#endif // USE_GLES2
       //#endif //(CC_PLATFORM == CC_PLATFORM_MAC_OSX)

    shaderStageList.emplace_back(std::move(fragmentShaderStage));

    GFXUniformList time = {{"u_time", GFXType::FLOAT, 1}};
    GFXUniformBlockList uniformBlockList = {
        {GFXShaderType::FRAGMENT, 0, "Time", time}};

    GFXUniformSamplerList samplers = {
        {GFXShaderType::FRAGMENT, 1, "u_texture", GFXType::SAMPLER2D, 1}};

    GFXShaderInfo shaderInfo;
    shaderInfo.name = "Blend Test: BigTriangle";
    shaderInfo.stages = std::move(shaderStageList);
    shaderInfo.blocks = std::move(uniformBlockList);
    shaderInfo.samplers = std::move(samplers);
    shader = device->createShader(shaderInfo);
  }

  void createVertexBuffer() {
    float ySign = device->getProjectionSignY();
    float vertexData[] = {-1.0f,         4.0f * ySign, -1.0f,
                          -1.0f * ySign, 4.0f,         -1.0f * ySign};

    // vertex buffer
    GFXBufferInfo vertexBufferInfo = {
        GFXBufferUsage::VERTEX, GFXMemoryUsage::HOST, 2 * sizeof(float),
        sizeof(vertexData), GFXBufferFlagBit::NONE};

    vertexBuffer = device->createBuffer(vertexBufferInfo);
    vertexBuffer->update(vertexData, 0, sizeof(vertexData));

    // uniform buffer
    GFXBufferInfo uniformBufferInfo = {
        GFXBufferUsage::UNIFORM, GFXMemoryUsage::HOST | GFXMemoryUsage::DEVICE,
        sizeof(float), sizeof(float), GFXBufferFlagBit::NONE};
    timeBuffer = device->createBuffer(uniformBufferInfo);
  }

  void createInputAssembler() {
    GFXAttribute position = {"a_position", GFXFormat::RG32F, false, 0, false};
    GFXInputAssemblerInfo inputAssemblerInfo;
    inputAssemblerInfo.attributes.emplace_back(std::move(position));
    inputAssemblerInfo.vertexBuffers.emplace_back(vertexBuffer);
    inputAssembler = device->createInputAssembler(inputAssemblerInfo);
  }

  void createTexture() {
    GFXTextureInfo textureInfo;
    textureInfo.usage = GFXTextureUsage::SAMPLED;
    textureInfo.format = GFXFormat::RGBA8;
    textureInfo.width = 128;
    textureInfo.height = 128;
    textureInfo.flags = GFXTextureFlagBit::GEN_MIPMAP;
    textureInfo.mipLevel =
        TestBaseI::getMipmapLevelCounts(textureInfo.width, textureInfo.height);
    texture = ::cocos2d::createTexture(device, textureInfo, "background.png");

    // create sampler
    GFXSamplerInfo samplerInfo;
    samplerInfo.name = "Background Generate mipmap";
    samplerInfo.mipFilter = GFXFilter::LINEAR;
    samplerInfo.addressU = GFXAddress::WRAP;
    samplerInfo.addressV = GFXAddress::WRAP;
    sampler = device->createSampler(samplerInfo);
  }

  void createPipeline() {
    GFXBindingList bindingList = {
        {GFXShaderType::FRAGMENT, 0, GFXBindingType::UNIFORM_BUFFER, "Time", 1},
        {GFXShaderType::FRAGMENT, 1, GFXBindingType::SAMPLER, "u_texture", 1},
    };
    GFXBindingLayoutInfo bindingLayoutInfo = {bindingList};
    bindingLayout = device->createBindingLayout(bindingLayoutInfo);

    bindingLayout->bindBuffer(0, timeBuffer);
    bindingLayout->bindSampler(1, sampler);
    bindingLayout->bindTexture(1, texture);
    bindingLayout->update();

    GFXPipelineLayoutInfo pipelineLayoutInfo;
    pipelineLayoutInfo.layouts = {bindingLayout};
    pipelineLayout = device->createPipelineLayout(pipelineLayoutInfo);

    GFXPipelineStateInfo pipelineInfo;
    pipelineInfo.primitive = GFXPrimitiveMode::TRIANGLE_LIST;
    pipelineInfo.shader = shader;
    pipelineInfo.inputState = {inputAssembler->getAttributes()};
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = fbo->getRenderPass();

    pipelineState = device->createPipelineState(pipelineInfo);
  }

  GFXDevice *device = nullptr;
  GFXFramebuffer* fbo = nullptr;
  GFXShader *shader = nullptr;
  GFXBuffer *vertexBuffer = nullptr;
  GFXBuffer *timeBuffer = nullptr;
  GFXInputAssembler *inputAssembler = nullptr;
  GFXSampler *sampler = nullptr;
  GFXBindingLayout *bindingLayout = nullptr;
  GFXTexture *texture = nullptr;
  GFXPipelineState *pipelineState = nullptr;
  GFXPipelineLayout *pipelineLayout = nullptr;
};

Mat4 createModelTransform(const Vec3 &t, const Vec3 &s) {
  Mat4 model;
  model.translate(t);
  model.scale(s);

  return model;
}

BigTriangle *bigTriangle = nullptr;
Quad *quad = nullptr;
} // namespace

void BlendTest::destroy() {
  CC_SAFE_DESTROY(bigTriangle);
  CC_SAFE_DESTROY(quad);
}

bool BlendTest::initialize() {
  bigTriangle = CC_NEW(BigTriangle(_device, _fbo));
  quad = CC_NEW(Quad(_device, _fbo));
  return true;
}

void BlendTest::tick(float dt) {

  _dt += dt;

  GFXRect render_area = {0, 0, _device->getWidth(), _device->getHeight()};
  GFXColor clear_color = {0.0f, 0, 0, 1.0f};

  _device->acquire();

  for (auto commandBuffer : _commandBuffers) {
    commandBuffer->begin();
    commandBuffer->beginRenderPass(
        _fbo, render_area, GFXClearFlagBit::ALL,
        std::move(std::vector<GFXColor>({clear_color})), 1.0f, 0);

    // draw background
    bigTriangle->timeBuffer->update(&_dt, 0, sizeof(_dt));
    commandBuffer->bindInputAssembler(bigTriangle->inputAssembler);
    commandBuffer->bindPipelineState(bigTriangle->pipelineState);
    commandBuffer->bindBindingLayout(bigTriangle->bindingLayout);
    commandBuffer->draw(bigTriangle->inputAssembler);

    commandBuffer->bindInputAssembler(quad->inputAssembler);

    // draw sprite without blending
    float size = std::min(render_area.width, render_area.height) * 0.15f;
    float halfSize = size * 0.5f;
    float offsetX = 5.f + halfSize;
    float offsetY = 50.f + halfSize;
    quad->model = std::move(
        createModelTransform(Vec3(offsetX, offsetY, 0), Vec3(size, size, 1)));
    quad->uniformBuffer[NO_BLEND]->update(quad->model.m, 0,
                                          sizeof(quad->model));
    commandBuffer->bindPipelineState(quad->pipelineState[NO_BLEND]);
    commandBuffer->bindBindingLayout(quad->bindingLayout[NO_BLEND]);
    commandBuffer->draw(quad->inputAssembler);

    // normal
    offsetY += 5.f + size;
    quad->model = std::move(createModelTransform(
        cocos2d::Vec3(offsetX, offsetY, 0), cocos2d::Vec3(size, size, 1)));
    quad->uniformBuffer[NORMAL_BLEND]->update(quad->model.m, 0,
                                              sizeof(quad->model));
    commandBuffer->bindPipelineState(quad->pipelineState[NORMAL_BLEND]);
    commandBuffer->bindBindingLayout(quad->bindingLayout[NORMAL_BLEND]);
    commandBuffer->draw(quad->inputAssembler);

    // additive
    offsetY += 5.f + size;
    quad->model = std::move(createModelTransform(
        cocos2d::Vec3(offsetX, offsetY, 0), cocos2d::Vec3(size, size, 1)));
    quad->uniformBuffer[ADDITIVE_BLEND]->update(quad->model.m, 0,
                                                sizeof(quad->model));
    commandBuffer->bindPipelineState(quad->pipelineState[ADDITIVE_BLEND]);
    commandBuffer->bindBindingLayout(quad->bindingLayout[ADDITIVE_BLEND]);
    commandBuffer->draw(quad->inputAssembler);

    // substract
    offsetY += 5.f + size;
    quad->model = std::move(createModelTransform(
        cocos2d::Vec3(offsetX, offsetY, 0), cocos2d::Vec3(size, size, 1)));
    quad->uniformBuffer[SUBSTRACT_BLEND]->update(quad->model.m, 0,
                                                 sizeof(quad->model));
    commandBuffer->bindPipelineState(quad->pipelineState[SUBSTRACT_BLEND]);
    commandBuffer->bindBindingLayout(quad->bindingLayout[SUBSTRACT_BLEND]);
    commandBuffer->draw(quad->inputAssembler);

    // multiply
    offsetY += 5.f + size;
    quad->model = std::move(createModelTransform(
        cocos2d::Vec3(offsetX, offsetY, 0), cocos2d::Vec3(size, size, 1)));
    quad->uniformBuffer[MULTIPLY_BLEND]->update(quad->model.m, 0,
                                                sizeof(quad->model));
    commandBuffer->bindPipelineState(quad->pipelineState[MULTIPLY_BLEND]);
    commandBuffer->bindBindingLayout(quad->bindingLayout[MULTIPLY_BLEND]);
    commandBuffer->draw(quad->inputAssembler);

    commandBuffer->endRenderPass();
    commandBuffer->end();
  }
  _device->getQueue()->submit(_commandBuffers);
  _device->present();
}

NS_CC_END