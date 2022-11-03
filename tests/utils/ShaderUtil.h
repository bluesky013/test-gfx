#pragma once

#include "gfx-base/GFXShader.h"
#include <string>

namespace cc {

gfx::Shader *createShader(gfx::ShaderInfo &shaderInfo, gfx::Device *device, const gfx::ShaderStageList &shaderList);

}
