/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#include "aom/aom_codec.h"
static const char* const cfg = "cmake ../../ -G \"Unix Makefiles\" -DCMAKE_TOOLCHAIN_FILE=\"../../build/cmake/toolchains/ppc-linux-gcc.cmake\" -DCONFIG_SVT_AV1=0 -DCONFIG_MULTITHREAD=0 -DFORCE_HIGHBITDEPTH_DECODING=0 -DCONFIG_SIZE_LIMIT=1 -DDECODE_HEIGHT_LIMIT=12288 -DDECODE_WIDTH_LIMIT=12288 -DCONFIG_HIGHWAY=0 -DENABLE_DOCS=0";
const char *aom_codec_build_config(void) {return cfg;}
