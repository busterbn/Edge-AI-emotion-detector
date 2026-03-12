#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdint.h>
#include <stddef.h>

#define MODEL_INPUT_SIZE 48

// Crop center square from frame and resize to MODEL_INPUT_SIZE x MODEL_INPUT_SIZE.
// Output is int8 (pixel - 128) for quantized model input.
void preprocess_frame(const uint8_t *frame, int frame_w, int frame_h,
                      int8_t *output);

#endif
