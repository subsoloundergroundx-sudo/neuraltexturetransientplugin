#pragma once

#include "DspModule.h"
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <string>

/**
 * MODULE NeuralTransientTexture
 * 
 * AI Assisted Spectral Transient Processor
 * 
 * Lock-Free Real-Time Architecture:
 * - Atomic Config Struct (Triple-buffering / Atomic Load-Store)
 * - Zero Mutex contention on the real-time audio callback thread
 * - 64-bit precision internal DSD DSP pipeline
 */
class NeuralTransientTextureModule : public DspModule {
public:
    enum TransientAttackType {
        ATTACK_KICK = 0,
        ATTACK_SNARE,
        ATTACK_CLAP,
        ATTACK_HIHAT,
        ATTACK_PERCUSSION,
        ATTACK_VOCAL,
        ATTACK_SYNTH,
        ATTACK_BASS,
        ATTACK_COUNT
    };

    enum TransientProfile {
        PROFILE_HARD = 0,
        PROFILE_SOFT,
        PROFILE_SHARP,
        PROFILE_PUNCHY,
        PROFILE_HARSH,
        PROFILE_SMOOTH,
        PROFILE_DENSE,
        PROFILE_SPARSE,
        PROFILE_COUNT
    };

    enum SaturationModel {
        SAT_TAPE = 0,
        SAT_TUBE,
        SAT_TRANSFORMER,
        SAT_PUNCH_DRIVE,
        SAT_SOFT_CLIP,
        SAT_HARD_CLIP,
        SAT_ANALOG_HYBRID,
        SAT_COUNT
    };

    struct TelemetryData {
        float peakEnergy = 0.0f;
        float attackSpeed = 0.0f;
        float transientDensity = 0.0f;
        float spectralBalance = 0.0f;
        float frequencyFocus = 1000.0f;
        float crestFactor = 1.0f;
        float rmsEnergy = 0.0f;
        float spectralCentroid = 1000.0f;
        float spectralFlux = 0.0f;
        int detectedAttack = ATTACK_SNARE;
        int detectedProfile = PROFILE_PUNCHY;
        float neuralPunch = 0.0f;
        float neuralSoften = 0.0f;
        float neuralTexture = 0.0f;
        float neuralSat = 0.0f;
        float transientActivity = 0.0f;
        float qualityMeter = 0.95f;

        // DEBUG_SYSTEM Telemetry Monitors
        float inputRMS = 0.0f;
        float outputRMS = 0.0f;
        float fftActivity = 0.0f;
        int transientCount = 0;
        float neuralActivity = 0.0f;
        int dspRoutingStatus = 0;
        bool forceTestModeActive = false;
        bool deltaModeActive = false;
        float wetDryMix = 1.0f;
    };

    struct Config {
        bool bypass = false;
        float inputGainDb = 0.0f;
        float outputGainDb = 0.0f;
        float lookaheadMs = 10.0f;
        int fftSize = 1024;
        float sensitivity = 0.5f;
        float targetLowHz = 80.0f;
        float targetHighHz = 16000.0f;
        bool targetEnabled = false;
        bool softEnabled = true;
        float softAmount = 0.4f;
        bool punchEnabled = true;
        float punchAmount = 0.5f;
        bool textureEnabled = true;
        float textureAmount = 0.35f;
        float textureDensity = 0.5f;
        float textureCharacter = 0.5f;
        bool saturateEnabled = true;
        int saturateModel = SAT_ANALOG_HYBRID;
        float saturateDrive = 0.3f;
        float saturateMix = 0.5f;
        bool limiterSafeguard = true;
        bool forceTestMode = false;
        bool deltaMode = false;
        float wetMix = 1.0f;
        int oversampleFactor = 2;
    };

    NeuralTransientTextureModule();
    ~NeuralTransientTextureModule() override = default;

    void process(float* buffer, int numFrames) override;
    void setSampleRate(int sampleRate) override;
    void reset() override;

    // Lock-Free Atomic Configuration Interface
    void setConfig(const Config& config);
    Config getConfig() const;

    // Controls
    void setInputGainDb(float gainDb);
    void setOutputGainDb(float gainDb);
    void setLookaheadMs(float ms);
    void setFftSize(int size);
    void setSensitivity(float sens);
    void setSpectralTarget(float lowHz, float highHz, bool enabled);

    // Modes & Processing amounts
    void setSoftMode(bool enabled, float amount);
    void setPunchMode(bool enabled, float amount);
    void setTextureMode(bool enabled, float amount, float density, float character);
    void setSaturateMode(bool enabled, int model, float drive, float mix);

    void setLimiterSafeguard(bool enabled);
    void setOfflineRenderMode(bool enabled, int oversampleFactor);
    void setOversampling(int factor);
    void setWetMix(float mix);
    float getWetMix() const;
    void setDeltaMode(bool enabled);
    bool isDeltaMode() const;
    void setBypass(bool bypass) override;
    bool isBypass() const;

    // DEBUG_SYSTEM
    void setForceTestMode(bool enabled);
    bool isForceTestMode() const;

    // Telemetry readouts for UI visualization & Debug System
    void getTelemetry(TelemetryData& outData) const;
    void getSpectrum(float* outSpectrum, int numBins) const;

private:
    void initBuffers();
    void updateLookaheadLength();
    void computeFftAnalysis(const double* lookaheadWindow, int windowSize);
    void runTransientDetectorAndClassifier();
    void runNeuralDecisionEngine();
    void processDsdPipeline64(double* inL, double* inR, int frames, const Config& cfg);

    // 64-bit precision Biquad filter helper for spectral targeting
    struct Biquad64 {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;
        void reset() { z1 = 0.0; z2 = 0.0; }
        inline double process(double in) {
            double out = in * b0 + z1;
            z1 = in * b1 - out * a1 + z2;
            z2 = in * b2 - out * a2;
            return out;
        }
    };

    void updateBandpassCoefficients(float lowHz, float highHz);

    // Atomic Configuration Struct (Lock-free thread communication)
    std::atomic<Config> atomicConfig_;

    // Telemetry and routing diagnostics
    std::atomic<int> dspRoutingStatus_{0};
    std::atomic<float> liveInputRms_{0.0f};
    std::atomic<float> liveOutputRms_{0.0f};

    // 64-bit Lookahead Ring Buffer
    std::vector<double> lookaheadRingL_;
    std::vector<double> lookaheadRingR_;
    size_t ringWritePos_ = 0;
    std::atomic<size_t> lookaheadFrames_{480}; // 10ms at 48kHz
    size_t ringCapacity_ = 192000;

    // FFT Analysis Working Buffers (64-bit)
    std::vector<double> fftWindow_;
    std::vector<double> fftReal_;
    std::vector<double> fftImag_;
    std::vector<double> fftMag_;
    std::vector<double> prevFftMag_;

    // Spectral target filters (64-bit)
    Biquad64 bpFilterL1_, bpFilterL2_;
    Biquad64 bpFilterR1_, bpFilterR2_;

    // Feature Extractors
    double peakEnergy_ = 0.0;
    double attackSpeed_ = 0.0;
    double transientDensity_ = 0.0;
    double spectralBalance_ = 0.0;
    double frequencyFocus_ = 1000.0;
    double crestFactor_ = 1.0;
    double rmsEnergy_ = 0.0;
    double spectralCentroid_ = 1000.0;
    double spectralFlux_ = 0.0;

    // Transient Detection State
    double envelopeFast_ = 0.0;
    double envelopeSlow_ = 0.0;
    double transientPulse_ = 0.0;
    int transientCounter_ = 0;
    int detectedAttackType_ = ATTACK_SNARE;
    int detectedProfile_ = PROFILE_PUNCHY;

    // Neural Decision Outputs (Target Control Signals)
    double neuralPunchTarget_ = 0.5;
    double neuralSoftenTarget_ = 0.4;
    double neuralTextureTarget_ = 0.35;
    double neuralSatTarget_ = 0.3;
    double neuralWidthTarget_ = 1.0;
    double neuralAdaptiveDrive_ = 0.3;

    // Smoothed Modulation Values
    double punchMod_ = 0.5;
    double softenMod_ = 0.4;
    double textureMod_ = 0.35;
    double satMod_ = 0.3;
    double widthMod_ = 1.0;

    // Texture Generator state
    double texturePhase_ = 0.0;

    // Zero-allocation real-time scratch buffers
    std::vector<double> scratchInL_;
    std::vector<double> scratchInR_;

    // Limiter Safeguard state (64-bit)
    double limiterEnvelope_ = 0.0;

    // Telemetry & Synchronization
    mutable std::mutex telemetryMutex_;
    TelemetryData currentTelemetry_;
    std::array<std::atomic<float>, 64> spectrumBins_;
    int fftProcessCounter_ = 0;
};
