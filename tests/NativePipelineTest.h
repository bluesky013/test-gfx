#pragma once

#include "TestBase.h"
#include "base/Ptr.h"
#include "renderer/pipeline/custom/RenderInterfaceTypes.h"

namespace cc {

class NativePipelineTest : public TestBaseI {
public:
    DEFINE_CREATE_METHOD(NativePipelineTest)
    using TestBaseI::TestBaseI;
    ~NativePipelineTest() override = default;

    bool onInit() override;
    void onTick() override;
    void onDestroy() override;

    void onSpacePressed() override;

protected:
    std::unique_ptr<render::Pipeline> ppl;
};

}
