#include "NativePipelineTest.h"

namespace cc {

bool NativePipelineTest::onInit() {
    ppl.reset(render::Factory::createPipeline());

    auto *layoutGraph = ppl->getLayoutGraphBuilder();
    uint32_t forwardColorId = layoutGraph->addRenderStage("forwardColor");
    uint32_t forwardQueueId = layoutGraph->addRenderPhase("Queue", forwardColorId);

    layoutGraph->compile();
    layoutGraph->print();
    return true;
}

void NativePipelineTest::onTick() {
    ppl->beginSetup();
    ppl->addRenderTarget("testCL", gfx::Format::RGBA8, 128, 128, render::ResourceResidency::MANAGED);

    auto *builder = ppl->addRasterPass(128, 128, "forwardColor");
    builder->addRasterView("testCL", {"color0", render::AccessType::WRITE, render::AttachmentType::RENDER_TARGET, gfx::LoadOp::CLEAR, gfx::StoreOp::STORE, gfx::ClearFlagBit::ALL, {}});
    builder->addQueue(render::QueueHint::RENDER_OPAQUE);

    ppl->endSetup();

    ppl->beginFrame();
    ppl->render({});
    ppl->endFrame();
}

void NativePipelineTest::onDestroy() {
    ppl = nullptr;
}

void NativePipelineTest::onSpacePressed() {

}

}
