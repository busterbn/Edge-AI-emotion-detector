#include "preprocess.h"

// Bilinear interpolation resize (integer-only)
// Crops center square from frame, then resizes to MODEL_INPUT_SIZE x MODEL_INPUT_SIZE
void preprocess_frame(const uint8_t *frame, int frame_w, int frame_h,
                      int8_t *output) {
    // Crop center square
    int crop_size = (frame_w < frame_h) ? frame_w : frame_h;
    int x_offset = (frame_w - crop_size) / 2;
    int y_offset = (frame_h - crop_size) / 2;

    // Fixed-point scaling (16.16)
    int scale = (crop_size << 16) / MODEL_INPUT_SIZE;

    for (int out_y = 0; out_y < MODEL_INPUT_SIZE; out_y++) {
        int src_y_fp = out_y * scale;
        int src_y = (src_y_fp >> 16) + y_offset;
        int frac_y = (src_y_fp >> 8) & 0xFF;

        int src_y1 = src_y + 1;
        if (src_y1 >= frame_h) src_y1 = frame_h - 1;

        for (int out_x = 0; out_x < MODEL_INPUT_SIZE; out_x++) {
            int src_x_fp = out_x * scale;
            int src_x = (src_x_fp >> 16) + x_offset;
            int frac_x = (src_x_fp >> 8) & 0xFF;

            int src_x1 = src_x + 1;
            if (src_x1 >= frame_w) src_x1 = frame_w - 1;

            // Bilinear interpolation
            int p00 = frame[src_y  * frame_w + src_x];
            int p10 = frame[src_y  * frame_w + src_x1];
            int p01 = frame[src_y1 * frame_w + src_x];
            int p11 = frame[src_y1 * frame_w + src_x1];

            int top    = p00 + ((frac_x * (p10 - p00)) >> 8);
            int bottom = p01 + ((frac_x * (p11 - p01)) >> 8);
            int pixel  = top  + ((frac_y * (bottom - top)) >> 8);

            // Convert uint8 [0,255] to int8 [-128,127] for quantized model
            output[out_y * MODEL_INPUT_SIZE + out_x] = (int8_t)(pixel - 128);
        }
    }
}
