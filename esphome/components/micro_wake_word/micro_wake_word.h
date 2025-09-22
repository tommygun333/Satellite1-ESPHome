#pragma once

#ifdef USE_ESP_IDF

#include "preprocessor_settings.h"
#include "streaming_model.h"

#include "esphome/components/microphone/microphone_source.h"

#include "esphome/core/automation.h"
#include "esphome/core/component.h"
#include "esphome/components/audio/ring_buffer.h"

#include <freertos/event_groups.h>

#include <frontend.h>
#include <frontend_util.h>

namespace esphome {
namespace micro_wake_word {

enum State {
  STARTING,
  DETECTING_WAKE_WORD,
  STOPPING,
  STOPPED,
};

class MicroWakeWord : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override;
  void dump_config() override;

  void start();
  void stop();

  bool is_running() const { return this->state_ != State::STOPPED; }

  void set_features_step_size(uint8_t step_size) { this->features_step_size_ = step_size; }

  void set_microphone_source(microphone::MicrophoneSource *microphone_source) {
    this->microphone_source_ = microphone_source;
  }

  void set_stop_after_detection(bool stop_after_detection) { this->stop_after_detection_ = stop_after_detection; }

  Trigger<std::string> *get_wake_word_detected_trigger() const { return this->wake_word_detected_trigger_; }

  void add_wake_word_model(WakeWordModel *model);

#ifdef USE_MICRO_WAKE_WORD_VAD
  void add_vad_model(const uint8_t *model_start, uint8_t probability_cutoff, size_t sliding_window_size,
                     size_t tensor_arena_size);

  // Intended for the voice assistant component to fetch VAD status
  bool get_vad_state() { return this->vad_state_; }
#endif

  // Intended for the voice assistant component to access which wake words are available
  // Since these are pointers to the WakeWordModel objects, the voice assistant component can enable or disable them
  std::vector<WakeWordModel *> get_wake_words();

  // Runtime-adjustable thresholds (top-level setters callable from YAML lambdas)
  // Wake word thresholds (applied to all wake-word models; only one active model is expected)
  void set_wake_word_probability_cutoff(float cutoff) {
    if (cutoff < 0.0f) cutoff = 0.0f;
    if (cutoff > 1.0f) cutoff = 1.0f;
    this->ww_probability_cutoff_ = cutoff;
  }
  void set_wake_word_sliding_window_size(size_t size) {
    if (size < 1) size = 1;
    if (size > 100) size = 100;  // safety cap
    this->ww_sliding_window_size_ = size;
    // Reset counters so new window takes effect cleanly
    for (auto &v : this->ww_hits_) v = 0;
  }

#ifdef USE_MICRO_WAKE_WORD_VAD
  // VAD thresholds
  void set_vad_probability_cutoff(float cutoff) {
    if (cutoff < 0.0f) cutoff = 0.0f;
    if (cutoff > 1.0f) cutoff = 1.0f;
    this->vad_probability_cutoff_ = cutoff;
  }
  void set_vad_sliding_window_size(size_t size) {
    if (size < 1) size = 1;
    if (size > 100) size = 100;  // safety cap
    this->vad_sliding_window_size_ = size;
    this->vad_hits_ = 0;
  }
#endif

 protected:
  microphone::MicrophoneSource *microphone_source_{nullptr};
  Trigger<std::string> *wake_word_detected_trigger_ = new Trigger<std::string>();
  State state_{State::STOPPED};

  std::weak_ptr<audio::RingBuffer> ring_buffer_;
  std::vector<WakeWordModel *> wake_word_models_;

#ifdef USE_MICRO_WAKE_WORD_VAD
  std::unique_ptr<VADModel> vad_model_;
  bool vad_state_{false};
  // Runtime VAD gating parameters and state
  float vad_probability_cutoff_{0.70f};
  size_t vad_sliding_window_size_{20};
  size_t vad_hits_{0};
#endif

  // Runtime WW gating parameters and state
  float ww_probability_cutoff_{0.75f};
  size_t ww_sliding_window_size_{10};
  // Per-model consecutive-hit counters (aligned with wake_word_models_)
  std::vector<size_t> ww_hits_;

  bool pending_start_{false};
  bool pending_stop_{false};

  bool stop_after_detection_;

  uint8_t features_step_size_;

  // Audio frontend handles generating spectrogram features
  struct FrontendConfig frontend_config_;
  struct FrontendState frontend_state_;

  // Handles managing the stop/state of the inference task
  EventGroupHandle_t event_group_;

  // Used to send messages about the models' states to the main loop
  QueueHandle_t detection_queue_;

  static void inference_task(void *params);
  TaskHandle_t inference_task_handle_{nullptr};

  /// @brief Suspends the inference task
  void suspend_task_();
  /// @brief Resumes the inference task
  void resume_task_();

  void set_state_(State state);

  /// @brief Generates spectrogram features from an input buffer of audio samples
  /// @param audio_buffer (int16_t *) Buffer containing input audio samples
  /// @param samples_available (size_t) Number of samples avaiable in the input buffer
  /// @param features_buffer (int8_t *) Buffer to store generated features
  /// @return (size_t) Number of samples processed from the input buffer
  size_t generate_features_(int16_t *audio_buffer, size_t samples_available,
                            int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE]);

  /// @brief Processes any new probabilities for each model. If any wake word is detected, it will send a DetectionEvent
  /// to the detection_queue_.
  void process_probabilities_();

  /// @brief Deletes each model's TFLite interpreters and frees tensor arena memory.
  void unload_models_();

  /// @brief Runs an inference with each model using the new spectrogram features
  /// @param audio_features (int8_t *) Buffer containing new spectrogram features
  /// @return True if successful, false if any errors were encountered
  bool update_model_probabilities_(const int8_t audio_features[PREPROCESSOR_FEATURE_SIZE]);
};

}  // namespace micro_wake_word
}  // namespace esphome

#endif  // USE_ESP_IDF
