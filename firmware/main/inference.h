#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "esp_err.h"

#define NUM_EMOTIONS 7

#ifdef __cplusplus
extern "C" {
#endif

extern const char *emotion_labels[NUM_EMOTIONS];

esp_err_t inference_init(void);

// Run inference on preprocessed int8 input (48x48x1).
// Returns emotion index (0-6) and fills confidence (0.0-1.0).
int inference_run(const int8_t *input, float *confidence);

#ifdef __cplusplus
}
#endif

#endif
