#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "camera.h"
#include "preprocess.h"
#include "inference.h"

static const char *TAG = "main";

void app_main(void) {
    // Init NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    // Init camera
    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed, halting");
        return;
    }

    // Init TFLite interpreter
    if (inference_init() != ESP_OK) {
        ESP_LOGE(TAG, "Inference init failed, halting");
        return;
    }

    ESP_LOGI(TAG, "System ready. Starting emotion recognition loop...");

    int8_t preprocessed[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];

    while (1) {
        camera_fb_t *fb = camera_capture();
        if (!fb) {
            ESP_LOGW(TAG, "Camera capture failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // Preprocess: crop center + resize to 48x48 + int8
        preprocess_frame(fb->buf, fb->width, fb->height, preprocessed);

        // Run inference
        float confidence;
        int emotion = inference_run(preprocessed, &confidence);

        if (emotion >= 0) {
            printf("Emotion: %-10s  Confidence: %.1f%%\n",
                   emotion_labels[emotion], confidence * 100.0f);
        }

        camera_release(fb);

        // ~10 FPS
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
