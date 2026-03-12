#include "inference.h"
#include "model_data.h"
#include "preprocess.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "inference";

const char *emotion_labels[NUM_EMOTIONS] = {
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
};

#define TENSOR_ARENA_SIZE (64 * 1024)

static uint8_t *tensor_arena = NULL;
static tflite::MicroInterpreter *interpreter = NULL;
static TfLiteTensor *input_tensor = NULL;
static TfLiteTensor *output_tensor = NULL;

esp_err_t inference_init(void) {
    // Allocate tensor arena in PSRAM
    tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM");
        return ESP_ERR_NO_MEM;
    }

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: %lu vs %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMaxPool2D();
    resolver.AddMean();  // GlobalAveragePooling2D compiles to Mean
    resolver.AddReshape();
    resolver.AddQuantize();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return ESP_FAIL;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    ESP_LOGI(TAG, "Model loaded. Arena used: %zu / %d bytes",
             interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
    ESP_LOGI(TAG, "Input:  %dx%dx%d (type %d)",
             input_tensor->dims->data[1], input_tensor->dims->data[2],
             input_tensor->dims->data[3], input_tensor->type);

    return ESP_OK;
}

int inference_run(const int8_t *input, float *confidence) {
    // Copy preprocessed data into input tensor
    memcpy(input_tensor->data.int8,
           input,
           MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * sizeof(int8_t));

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return -1;
    }

    // Find argmax and dequantize
    float output_scale = output_tensor->params.scale;
    int output_zero_point = output_tensor->params.zero_point;

    int best_idx = 0;
    float best_score = -1e9f;

    for (int i = 0; i < NUM_EMOTIONS; i++) {
        float score = (output_tensor->data.int8[i] - output_zero_point) * output_scale;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    // Compute softmax confidence for the best class
    float sum_exp = 0.0f;
    for (int i = 0; i < NUM_EMOTIONS; i++) {
        float score = (output_tensor->data.int8[i] - output_zero_point) * output_scale;
        sum_exp += expf(score - best_score);
    }
    *confidence = 1.0f / sum_exp;

    return best_idx;
}
