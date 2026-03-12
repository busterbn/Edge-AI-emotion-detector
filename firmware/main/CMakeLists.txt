idf_component_register(
    SRCS "main.c" "camera.c" "preprocess.c" "inference.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_timer nvs_flash
    PRIV_REQUIRES esp32-camera esp-tflite-micro
)
