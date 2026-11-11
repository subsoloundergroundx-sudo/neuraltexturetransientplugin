#include "NeuralTransientTextureModule.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fast 64-bit Radix-2 Cooley-Tukey FFT
static void performFft64(std::vector<double>& real, std::vector<double>& imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    for (int len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        double wlenCos = std::cos(angle);
        double wlenSin = std::sin(angle);
        int halfLen = len >> 1;

        for (int i = 0; i < n; i += len) {
            double wCos = 1.0;
            double wSin = 0.0;
            for (int k = 0; k < halfLen; ++k) {
                double uR = real[i + k];
                double uI = imag[i + k];
                double vR = real[i + k + halfLen] * wCos - imag[i + k + halfLen] * wSin;
                double vI = real[i + k + halfLen] * wSin + imag[i + k + halfLen] * wCos;

                real[i + k] = uR + vR;
                imag[i + k] = uI + vI;
                real[i + k + halfLen] = uR - vR;
                imag[i + k + halfLen] = uI - vI;

                double nextWCos = wCos * wlenCos - wSin * wlenSin;
                double nextWSin = wCos * wlenSin + wSin * wlenCos;
                wCos = nextWCos;
                wSin = nextWSin;
            }
        }
    }
}

// 64-bit Saturation Models
static double applySaturation64(double in, int model, double drive) {
    double gain = 1.0 + drive * 4.0;
    in *= gain;
    double out = in;

    switch (model) {
        case NeuralTransientTextureModule::SAT_TAPE: {
            out = in / (1.0 + std::abs(in));
            break;
        }
        case NeuralTransientTextureModule::SAT_TUBE: {
            out = std::tanh(in) + 0.12 * std::tanh(in * 2.0) * std::tanh(in * 2.0);
            break;
        }
        case NeuralTransientTextureModule::SAT_TRANSFORMER: {
            double sign = (in >= 0.0) ? 1.0 : -1.0;
            out = sign * (1.0 - std::exp(-std::abs(in))) * 0.95;
            break;
        }
        case NeuralTransientTextureModule::SAT_PUNCH_DRIVE: {
            if (in > 0.0) out = std::tanh(in * 1.2) * 0.85;
            else out = std::tanh(in * 0.8) * 1.15;
            break;
        }
        case NeuralTransientTextureModule::SAT_SOFT_CLIP: {
            if (in > 1.0) out = 0.66667;
            else if (in < -1.0) out = -0.66667;
            else out = in - (in * in * in) / 3.0;
            break;
        }
        case NeuralTransientTextureModule::SAT_HARD_CLIP: {
            out = std::clamp(in, -0.98, 0.98);
            break;
        }
        case NeuralTransientTextureModule::SAT_ANALOG_HYBRID:
        default: {
            double tube = std::tanh(in);
            double diode = in / (1.0 + std::abs(in) * 0.8);
            out = 0.6 * tube + 0.4 * diode;
            break;
        }
    }
    return out / std::sqrt(gain);
}

NeuralTransientTextureModule::NeuralTransientTextureModule() {
    id_ = 15;
    sampleRate_ = 48000;
    
    Config initial;
    atomicConfig_.store(initial, std::memory_order_relaxed);

    initBuffers();
    updateLookaheadLength();
    updateBandpassCoefficients(initial.targetLowHz, initial.targetHighHz);
    for (auto& b : spectrumBins_) {
        b.store(0.0f, std::memory_order_relaxed);
    }
}

void NeuralTransientTextureModule::setSampleRate(int sampleRate) {
    if (sampleRate <= 0) sampleRate = 48000;
    sampleRate_ = sampleRate;
    initBuffers();
    updateLookaheadLength();
    Config cfg = atomicConfig_.load(std::memory_order_relaxed);
    updateBandpassCoefficients(cfg.targetLowHz, cfg.targetHighHz);
}

void NeuralTransientTextureModule::reset() {
    std::fill(lookaheadRingL_.begin(), lookaheadRingL_.end(), 0.0);
    std::fill(lookaheadRingR_.begin(), lookaheadRingR_.end(), 0.0);
    ringWritePos_ = 0;
    envelopeFast_ = 0.0;
    envelopeSlow_ = 0.0;
    transientPulse_ = 0.0;
    transientCounter_ = 0;
    limiterEnvelope_ = 0.0;
    bpFilterL1_.reset();
    bpFilterL2_.reset();
    bpFilterR1_.reset();
    bpFilterR2_.reset();
    dspRoutingStatus_.store(0, std::memory_order_relaxed);
}

void NeuralTransientTextureModule::initBuffers() {
    ringCapacity_ = static_cast<size_t>(sampleRate_ * 1.5);
    lookaheadRingL_.assign(ringCapacity_, 0.0);
    lookaheadRingR_.assign(ringCapacity_, 0.0);

    Config cfg = atomicConfig_.load(std::memory_order_relaxed);
    int size = std::clamp(cfg.fftSize, 512, 8192);
    fftWindow_.assign(size, 0.0);
    fftReal_.assign(size, 0.0);
    fftImag_.assign(size, 0.0);
    fftMag_.assign(size / 2, 0.0);
    prevFftMag_.assign(size / 2, 0.0);

    for (int i = 0; i < size; ++i) {
        fftWindow_[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
    }

    scratchInL_.assign(4096, 0.0);
    scratchInR_.assign(4096, 0.0);
}

void NeuralTransientTextureModule::setConfig(const Config& config) {
    Config c = config;
    c.inputGainDb = std::clamp(c.inputGainDb, -36.0f, 24.0f);
    c.outputGainDb = std::clamp(c.outputGainDb, -36.0f, 24.0f);
    c.lookaheadMs = std::clamp(c.lookaheadMs, 2.0f, 500.0f);
    c.sensitivity = std::clamp(c.sensitivity, 0.0f, 1.0f);
    c.targetLowHz = std::max(20.0f, c.targetLowHz);
    c.targetHighHz = std::min(20000.0f, c.targetHighHz);
    c.softAmount = std::clamp(c.softAmount, 0.0f, 1.0f);
    c.punchAmount = std::clamp(c.punchAmount, 0.0f, 1.0f);
    c.textureAmount = std::clamp(c.textureAmount, 0.0f, 1.0f);
    c.textureDensity = std::clamp(c.textureDensity, 0.0f, 1.0f);
    c.textureCharacter = std::clamp(c.textureCharacter, 0.0f, 1.0f);
    c.saturateModel = std::clamp(c.saturateModel, 0, static_cast<int>(SAT_COUNT) - 1);
    c.saturateDrive = std::clamp(c.saturateDrive, 0.0f, 1.0f);
    c.saturateMix = std::clamp(c.saturateMix, 0.0f, 1.0f);
    c.wetMix = std::clamp(c.wetMix, 0.0f, 1.0f);
    c.oversampleFactor = std::clamp(c.oversampleFactor, 1, 32);

    atomicConfig_.store(c, std::memory_order_release);
    bypass_ = c.bypass;

    updateLookaheadLength();
    updateBandpassCoefficients(c.targetLowHz, c.targetHighHz);
}

NeuralTransientTextureModule::Config NeuralTransientTextureModule::getConfig() const {
    return atomicConfig_.load(std::memory_order_acquire);
}

void NeuralTransientTextureModule::updateLookaheadLength() {
    Config cfg = atomicConfig_.load(std::memory_order_relaxed);
    double ms = static_cast<double>(cfg.lookaheadMs);
    size_t frames = static_cast<size_t>((ms / 1000.0) * sampleRate_);
    lookaheadFrames_ = std::clamp(frames, static_cast<size_t>(48), ringCapacity_ - 4096);
}

void NeuralTransientTextureModule::setInputGainDb(float gainDb) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.inputGainDb = std::clamp(gainDb, -36.0f, 24.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setOutputGainDb(float gainDb) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.outputGainDb = std::clamp(gainDb, -36.0f, 24.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setLookaheadMs(float ms) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.lookaheadMs = std::clamp(ms, 2.0f, 500.0f);
    atomicConfig_.store(c, std::memory_order_release);
    updateLookaheadLength();
}

void NeuralTransientTextureModule::setFftSize(int size) {
    int valid = 1024;
    if (size <= 512) valid = 512;
    else if (size <= 1024) valid = 1024;
    else if (size <= 2048) valid = 2048;
    else if (size <= 4096) valid = 4096;
    else valid = 8192;

    Config c = atomicConfig_.load(std::memory_order_relaxed);
    if (c.fftSize != valid) {
        c.fftSize = valid;
        atomicConfig_.store(c, std::memory_order_release);
        initBuffers();
    }
}

void NeuralTransientTextureModule::setSensitivity(float sens) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.sensitivity = std::clamp(sens, 0.0f, 1.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setWetMix(float mix) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.wetMix = std::clamp(mix, 0.0f, 1.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

float NeuralTransientTextureModule::getWetMix() const {
    return atomicConfig_.load(std::memory_order_relaxed).wetMix;
}

void NeuralTransientTextureModule::setForceTestMode(bool enabled) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.forceTestMode = enabled;
    atomicConfig_.store(c, std::memory_order_release);
}

bool NeuralTransientTextureModule::isForceTestMode() const {
    return atomicConfig_.load(std::memory_order_relaxed).forceTestMode;
}

void NeuralTransientTextureModule::setDeltaMode(bool enabled) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.deltaMode = enabled;
    atomicConfig_.store(c, std::memory_order_release);
}

bool NeuralTransientTextureModule::isDeltaMode() const {
    return atomicConfig_.load(std::memory_order_relaxed).deltaMode;
}

void NeuralTransientTextureModule::setBypass(bool bypass) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.bypass = bypass;
    atomicConfig_.store(c, std::memory_order_release);
    bypass_ = bypass;
}

bool NeuralTransientTextureModule::isBypass() const {
    return atomicConfig_.load(std::memory_order_relaxed).bypass;
}

void NeuralTransientTextureModule::setSpectralTarget(float lowHz, float highHz, bool enabled) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.targetLowHz = std::max(20.0f, lowHz);
    c.targetHighHz = std::min(20000.0f, highHz);
    c.targetEnabled = enabled;
    atomicConfig_.store(c, std::memory_order_release);
    updateBandpassCoefficients(c.targetLowHz, c.targetHighHz);
}

void NeuralTransientTextureModule::setSoftMode(bool enabled, float amount) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.softEnabled = enabled;
    c.softAmount = std::clamp(amount, 0.0f, 1.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setPunchMode(bool enabled, float amount) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.punchEnabled = enabled;
    c.punchAmount = std::clamp(amount, 0.0f, 1.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setTextureMode(bool enabled, float amount, float density, float character) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.textureEnabled = enabled;
    c.textureAmount = std::clamp(amount, 0.0f, 1.0f);
    c.textureDensity = std::clamp(density, 0.0f, 1.0f);
    c.textureCharacter = std::clamp(character, 0.0f, 1.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setSaturateMode(bool enabled, int model, float drive, float mix) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.saturateEnabled = enabled;
    c.saturateModel = std::clamp(model, 0, static_cast<int>(SAT_COUNT) - 1);
    c.saturateDrive = std::clamp(drive, 0.0f, 1.0f);
    c.saturateMix = std::clamp(mix, 0.0f, 1.0f);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setLimiterSafeguard(bool enabled) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.limiterSafeguard = enabled;
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::setOfflineRenderMode(bool enabled, int oversampleFactor) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.oversampleFactor = std::clamp(oversampleFactor, 1, 32);
    if (enabled) {
        c.fftSize = 4096;
        c.lookaheadMs = 100.0f;
    } else {
        c.fftSize = 1024;
        c.lookaheadMs = 10.0f;
    }
    atomicConfig_.store(c, std::memory_order_release);
    updateLookaheadLength();
    initBuffers();
}

void NeuralTransientTextureModule::setOversampling(int factor) {
    Config c = atomicConfig_.load(std::memory_order_relaxed);
    c.oversampleFactor = std::clamp(factor, 1, 32);
    atomicConfig_.store(c, std::memory_order_release);
}

void NeuralTransientTextureModule::updateBandpassCoefficients(float lowHz, float highHz) {
    if (sampleRate_ <= 0) return;
    double centerHz = std::sqrt(static_cast<double>(lowHz) * static_cast<double>(highHz));
    double bwOctaves = std::log2(static_cast<double>(highHz) / std::max(20.0, static_cast<double>(lowHz)));
    if (bwOctaves < 0.2) bwOctaves = 0.2;

    double w0 = 2.0 * M_PI * centerHz / sampleRate_;
    double alpha = std::sin(w0) * std::sinh(std::log(2.0) / 2.0 * bwOctaves * w0 / std::sin(w0));

    double b0 = alpha;
    double b1 = 0.0;
    double b2 = -alpha;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * std::cos(w0);
    double a2 = 1.0 - alpha;

    bpFilterL1_.b0 = b0 / a0; bpFilterL1_.b1 = b1 / a0; bpFilterL1_.b2 = b2 / a0;
    bpFilterL1_.a1 = a1 / a0; bpFilterL1_.a2 = a2 / a0;
    bpFilterL2_ = bpFilterL1_;
    bpFilterR1_ = bpFilterL1_;
    bpFilterR2_ = bpFilterL1_;
}

void NeuralTransientTextureModule::computeFftAnalysis(const double* lookaheadWindow, int windowSize) {
    if (windowSize <= 0) return;
    Config cfg = atomicConfig_.load(std::memory_order_relaxed);
    int curFftSize = static_cast<int>(fftReal_.size());
    if (curFftSize != cfg.fftSize && cfg.fftSize >= 512) {
        initBuffers();
        curFftSize = static_cast<int>(fftReal_.size());
    }

    int samplesToCopy = std::min(windowSize, curFftSize);
    for (int i = 0; i < samplesToCopy; ++i) {
        fftReal_[i] = lookaheadWindow[i] * fftWindow_[i];
        fftImag_[i] = 0.0;
    }
    for (int i = samplesToCopy; i < curFftSize; ++i) {
        fftReal_[i] = 0.0;
        fftImag_[i] = 0.0;
    }

    performFft64(fftReal_, fftImag_, curFftSize);

    int numBins = curFftSize / 2;
    double sumMag = 0.0;
    double sumFreqMag = 0.0;
    double maxMag = 0.0;
    int maxBin = 1;
    double flux = 0.0;
    double lowEnergy = 0.0;
    double highEnergy = 0.0;
    double binHz = static_cast<double>(sampleRate_) / curFftSize;

    for (int i = 1; i < numBins; ++i) {
        double r = fftReal_[i];
        double im = fftImag_[i];
        double mag = std::sqrt(r * r + im * im) / curFftSize;
        fftMag_[i] = mag;

        sumMag += mag;
        double freq = i * binHz;
        sumFreqMag += freq * mag;

        if (mag > maxMag) {
            maxMag = mag;
            maxBin = i;
        }

        double diff = mag - prevFftMag_[i];
        if (diff > 0.0) flux += diff;
        prevFftMag_[i] = mag;

        if (freq < 300.0) lowEnergy += mag;
        else if (freq > 3000.0) highEnergy += mag;
    }

    spectralCentroid_ = (sumMag > 1e-6) ? (sumFreqMag / sumMag) : 1000.0;
    spectralFlux_ = flux;
    frequencyFocus_ = maxBin * binHz;
    spectralBalance_ = (lowEnergy > 1e-6) ? (highEnergy / lowEnergy) : 1.0;

    int binsPerBand = std::max(1, numBins / 64);
    for (int b = 0; b < 64; ++b) {
        double bSum = 0.0;
        for (int k = 0; k < binsPerBand; ++k) {
            int idx = b * binsPerBand + k;
            if (idx < numBins) bSum += fftMag_[idx];
        }
        float normVal = static_cast<float>(std::clamp(bSum * 15.0, 0.0, 1.0));
        float oldVal = spectrumBins_[b].load(std::memory_order_relaxed);
        spectrumBins_[b].store(normVal > oldVal ? normVal : oldVal * 0.85f, std::memory_order_relaxed);
    }
}

void NeuralTransientTextureModule::runTransientDetectorAndClassifier() {
    Config cfg = atomicConfig_.load(std::memory_order_relaxed);
    double sens = static_cast<double>(cfg.sensitivity);
    double threshold = (1.15 - sens * 0.45);

    double ratio = (envelopeFast_ > 1e-5) ? (envelopeFast_ / (envelopeSlow_ + 1e-5)) : 1.0;
    crestFactor_ = std::clamp(peakEnergy_ / (rmsEnergy_ + 1e-4), 1.0, 25.0);

    bool isTransient = (ratio > threshold) && (spectralFlux_ > 0.015);

    if (isTransient) {
        transientPulse_ = 1.0;
        transientCounter_++;

        if (spectralCentroid_ < 220.0) {
            detectedAttackType_ = (attackSpeed_ > 0.7) ? ATTACK_KICK : ATTACK_BASS;
        } else if (spectralCentroid_ >= 220.0 && spectralCentroid_ < 1500.0) {
            if (spectralFlux_ > 0.25 && crestFactor_ > 4.0) {
                detectedAttackType_ = ATTACK_SNARE;
            } else if (frequencyFocus_ > 400.0 && frequencyFocus_ < 900.0) {
                detectedAttackType_ = ATTACK_SYNTH;
            } else {
                detectedAttackType_ = ATTACK_VOCAL;
            }
        } else if (spectralCentroid_ >= 1500.0 && spectralCentroid_ < 4500.0) {
            detectedAttackType_ = (attackSpeed_ > 0.65) ? ATTACK_CLAP : ATTACK_PERCUSSION;
        } else {
            detectedAttackType_ = ATTACK_HIHAT;
        }

        if (crestFactor_ > 7.0 && attackSpeed_ > 0.8) {
            detectedProfile_ = PROFILE_SHARP;
        } else if (crestFactor_ > 4.5 && spectralCentroid_ < 400.0) {
            detectedProfile_ = PROFILE_PUNCHY;
        } else if (spectralBalance_ > 3.0 && crestFactor_ > 4.0) {
            detectedProfile_ = PROFILE_HARSH;
        } else if (crestFactor_ > 3.5) {
            detectedProfile_ = PROFILE_HARD;
        } else if (attackSpeed_ < 0.35) {
            detectedProfile_ = PROFILE_SOFT;
        } else {
            detectedProfile_ = PROFILE_SMOOTH;
        }
    } else {
        transientPulse_ *= 0.90;
    }

    transientDensity_ = std::clamp(static_cast<double>(transientCounter_) * 0.05, 0.0, 10.0);
    if (fftProcessCounter_ % 50 == 0) transientCounter_ = 0;
}

void NeuralTransientTextureModule::runNeuralDecisionEngine() {
    Config cfg = atomicConfig_.load(std::memory_order_relaxed);

    double fCentroid = std::clamp(spectralCentroid_ / 8000.0, 0.0, 1.0);
    double fFlux = std::clamp(spectralFlux_ * 2.0, 0.0, 1.0);
    double fCrest = std::clamp((crestFactor_ - 1.0) / 10.0, 0.0, 1.0);
    double fAttack = std::clamp(attackSpeed_, 0.0, 1.0);
    double fDensity = std::clamp(transientDensity_ / 8.0, 0.0, 1.0);

    double h0 = std::tanh(0.6 * fCrest + 0.5 * fAttack - 0.2 * fCentroid);
    double h1 = std::tanh(0.8 * fFlux + 0.3 * fCentroid - 0.4 * fCrest);
    double h2 = std::tanh(0.7 * fCentroid + 0.6 * fFlux - 0.1 * fDensity);
    double h3 = std::tanh(0.5 * fAttack + 0.5 * fCrest + 0.2 * fDensity);

    neuralPunchTarget_ = std::clamp(0.5 * h0 + 0.5 * h3 + 0.5, 0.0, 1.0) * cfg.punchAmount;
    neuralSoftenTarget_ = std::clamp(0.7 * h1 + 0.3 * (1.0 - h0), 0.0, 1.0) * cfg.softAmount;
    neuralTextureTarget_ = std::clamp(0.6 * h2 + 0.4 * fDensity, 0.0, 1.0) * cfg.textureAmount;
    neuralSatTarget_ = std::clamp(0.5 * h3 + 0.3 * h1 + 0.2, 0.0, 1.0);
    neuralWidthTarget_ = std::clamp(1.0 + 0.4 * (h2 - 0.5), 0.6, 1.4);
    neuralAdaptiveDrive_ = std::clamp(cfg.saturateDrive * (0.8 + 0.6 * h3), 0.05, 1.0);
}

void NeuralTransientTextureModule::processDsdPipeline64(double* inL, double* inR, int frames, const Config& cfg) {
    const double slew = 0.05;
    bool isForceTest = cfg.forceTestMode;

    bool isSoft = cfg.softEnabled || isForceTest;
    bool isPunch = cfg.punchEnabled || isForceTest;
    bool isTexture = cfg.textureEnabled || isForceTest;
    bool isSat = cfg.saturateEnabled || isForceTest;
    bool isTarget = cfg.targetEnabled && !isForceTest;
    bool isLimiter = cfg.limiterSafeguard;
    int satModel = cfg.saturateModel;

    double userSatMix = isForceTest ? 1.0 : static_cast<double>(cfg.saturateMix);
    double userTextureChar = static_cast<double>(cfg.textureCharacter);
    double userTextureDensity = static_cast<double>(cfg.textureDensity);
    double wetMix = isForceTest ? 1.0 : static_cast<double>(cfg.wetMix);

    double inGainLin = std::pow(10.0, static_cast<double>(cfg.inputGainDb) / 20.0);
    double outGainLin = std::pow(10.0, static_cast<double>(cfg.outputGainDb) / 20.0);

    double totalInSq = 0.0;
    double totalOutSq = 0.0;
    bool routingSuccess = true;

    for (int i = 0; i < frames; ++i) {
        double rawL = inL[i] * inGainLin;
        double rawR = inR[i] * inGainLin;
        totalInSq += (rawL * rawL + rawR * rawR) * 0.5;

        // Smooth parameter transitions
        double targetPunch = isForceTest ? 1.0 : neuralPunchTarget_;
        double targetSoften = isForceTest ? 0.8 : neuralSoftenTarget_;
        double targetTexture = isForceTest ? 1.0 : neuralTextureTarget_;
        double targetSat = isForceTest ? 1.0 : neuralSatTarget_;
        double targetDrive = isForceTest ? 1.0 : neuralAdaptiveDrive_;

        punchMod_ += (targetPunch - punchMod_) * slew;
        softenMod_ += (targetSoften - softenMod_) * slew;
        textureMod_ += (targetTexture - textureMod_) * slew;
        satMod_ += (targetSat - satMod_) * slew;
        widthMod_ += (neuralWidthTarget_ - widthMod_) * slew;

        // Spectral Targeting
        double targetL = rawL;
        double targetR = rawR;
        double dryNonTargetL = 0.0;
        double dryNonTargetR = 0.0;

        if (isTarget) {
            double bandL = bpFilterL2_.process(bpFilterL1_.process(rawL));
            double bandR = bpFilterR2_.process(bpFilterR1_.process(rawR));
            dryNonTargetL = rawL - bandL;
            dryNonTargetR = rawR - bandR;
            targetL = bandL;
            targetR = bandR;
        }

        // ====================================================================
        // MODULAR PROCESSING IN 64-BIT PRECISION
        // ====================================================================
        double punchDeltaL = 0.0, punchDeltaR = 0.0;
        double softDeltaL = 0.0, softDeltaR = 0.0;
        double textureDeltaL = 0.0, textureDeltaR = 0.0;
        double satDeltaL = 0.0, satDeltaR = 0.0;

        double effectiveTransient = isForceTest ? 1.0 : transientPulse_;

        // 1. PUNCH MODE (Dynamic attack acceleration + solid punch body)
        if (isPunch && cfg.punchAmount > 0.005f) {
            double punchFactor = 1.0 + (cfg.punchAmount * (2.2 * effectiveTransient + 0.4 * punchMod_));
            punchDeltaL = (targetL * punchFactor) - targetL;
            punchDeltaR = (targetR * punchFactor) - targetR;
        }

        // 2. SOFT MODE (Dynamic peak smoothing & harshness suppression)
        if (isSoft && cfg.softAmount > 0.005f) {
            double softFactor = 1.0 - (cfg.softAmount * (0.35 * effectiveTransient + 0.15 * softenMod_));
            softDeltaL = (targetL * softFactor) - targetL;
            softDeltaR = (targetR * softFactor) - targetR;
        }

        // 3. TEXTURE ENGINE (Chebyshev Micro-Harmonics excitation)
        if (isTexture && cfg.textureAmount > 0.005f) {
            texturePhase_ += 0.15 * (1.0 + userTextureChar * 2.0);
            if (texturePhase_ > 2.0 * M_PI) texturePhase_ -= 2.0 * M_PI;

            double microH3_L = (4.0 * targetL * targetL * targetL - 3.0 * targetL);
            double microH3_R = (4.0 * targetR * targetR * targetR - 3.0 * targetR);

            double targetL5 = std::clamp(targetL, -1.0, 1.0);
            double targetR5 = std::clamp(targetR, -1.0, 1.0);
            double microH5_L = (16.0 * std::pow(targetL5, 5) - 20.0 * std::pow(targetL5, 3) + 5.0 * targetL5);
            double microH5_R = (16.0 * std::pow(targetR5, 5) - 20.0 * std::pow(targetR5, 3) + 5.0 * targetR5);

            double harmL = (1.0 - userTextureChar) * microH3_L + userTextureChar * microH5_L;
            double harmR = (1.0 - userTextureChar) * microH3_R + userTextureChar * microH5_R;

            double textureWeight = cfg.textureAmount * (0.45 * effectiveTransient + 0.25 * userTextureDensity + 0.20 * textureMod_);
            textureDeltaL = harmL * textureWeight;
            textureDeltaR = harmR * textureWeight;
        }

        // 4. SATURATION ENGINE (Analog Tape/Tube/Transformer/Hybrid models)
        if (isSat && (cfg.saturateDrive > 0.005f || cfg.saturateMix > 0.005f)) {
            double drive = cfg.saturateDrive * (0.7 + 0.3 * effectiveTransient);
            double satL = applySaturation64(targetL, satModel, drive);
            double satR = applySaturation64(targetR, satModel, drive);

            double mix = userSatMix * (0.6 + 0.4 * effectiveTransient);
            satDeltaL = ((targetL * (1.0 - mix) + satL * mix)) - targetL;
            satDeltaR = ((targetR * (1.0 - mix) + satR * mix)) - targetR;
        }

        // Recombination
        double procL = targetL + punchDeltaL + softDeltaL + textureDeltaL + satDeltaL;
        double procR = targetR + punchDeltaR + softDeltaR + textureDeltaR + satDeltaR;

        // Stereo Width modulation
        double mid = (procL + procR) * 0.5;
        double side = (procL - procR) * 0.5;
        side *= (1.0 + (widthMod_ - 1.0) * effectiveTransient);
        procL = mid + side;
        procR = mid - side;

        if (isTarget) {
            procL += dryNonTargetL;
            procR += dryNonTargetR;
        }

        // Wet / Dry Mixer & Delta Audition
        double outL, outR;
        if (cfg.deltaMode) {
            outL = procL - rawL;
            outR = procR - rawR;
        } else {
            outL = (1.0 - wetMix) * rawL + wetMix * procL;
            outR = (1.0 - wetMix) * rawR + wetMix * procR;
        }

        // Output Gain
        outL *= outGainLin;
        outR *= outGainLin;

        // Limiter Safe Guard
        if (isLimiter) {
            double peak = std::max(std::abs(outL), std::abs(outR));
            limiterEnvelope_ = std::max(peak, limiterEnvelope_ * 0.999 + peak * 0.001);
            if (limiterEnvelope_ > 0.98) {
                double limitGain = 0.98 / limiterEnvelope_;
                outL *= limitGain;
                outR *= limitGain;
            }
        }

        totalOutSq += (outL * outL + outR * outR) * 0.5;
        inL[i] = outL;
        inR[i] = outR;
    }

    liveInputRms_.store(static_cast<float>(std::sqrt(totalInSq / frames)), std::memory_order_relaxed);
    liveOutputRms_.store(static_cast<float>(std::sqrt(totalOutSq / frames)), std::memory_order_relaxed);
    dspRoutingStatus_.store(routingSuccess ? 0 : 1, std::memory_order_relaxed);
}

void NeuralTransientTextureModule::process(float* buffer, int numFrames) {
    Config cfg = atomicConfig_.load(std::memory_order_acquire);
    if (cfg.bypass || !buffer || numFrames <= 0) return;

    fftProcessCounter_++;

    if (scratchInL_.size() < static_cast<size_t>(numFrames)) {
        size_t newCap = std::max(static_cast<size_t>(numFrames * 2), static_cast<size_t>(4096));
        scratchInL_.resize(newCap);
        scratchInR_.resize(newCap);
    }
    double* inL = scratchInL_.data();
    double* inR = scratchInR_.data();

    for (int i = 0; i < numFrames; ++i) {
        inL[i] = static_cast<double>(buffer[i * 2]);
        inR[i] = static_cast<double>(buffer[i * 2 + 1]);
    }

    // Step 1: Write incoming audio to Lookahead Ring Buffer
    for (int i = 0; i < numFrames; ++i) {
        lookaheadRingL_[ringWritePos_] = inL[i];
        lookaheadRingR_[ringWritePos_] = inR[i];
        ringWritePos_ = (ringWritePos_ + 1) % ringCapacity_;
    }

    // Step 2: Lookahead Analysis on newly written incoming window
    double framePeak = 0.0;
    double frameSqSum = 0.0;
    for (int i = 0; i < numFrames; ++i) {
        double mono = (std::abs(inL[i]) + std::abs(inR[i])) * 0.5;
        if (mono > framePeak) framePeak = mono;
        frameSqSum += mono * mono;
    }
    peakEnergy_ = framePeak;
    rmsEnergy_ = std::sqrt(frameSqSum / numFrames);

    double oldEnv = envelopeFast_;
    envelopeFast_ += (peakEnergy_ - envelopeFast_) * 0.35;
    envelopeSlow_ += (peakEnergy_ - envelopeSlow_) * 0.015;
    attackSpeed_ = std::clamp((envelopeFast_ - oldEnv) * 20.0, 0.0, 1.0);

    // Compute FFT periodically
    if (fftProcessCounter_ % 2 == 0) {
        int windowSize = std::min(numFrames, static_cast<int>(cfg.fftSize));
        computeFftAnalysis(inL, windowSize);
        runTransientDetectorAndClassifier();
        runNeuralDecisionEngine();
    }

    // Step 3: Read delayed audio from lookahead ring buffer for sample-aligned processing
    size_t delayedReadPos = (ringWritePos_ + ringCapacity_ - lookaheadFrames_ - numFrames) % ringCapacity_;
    for (int i = 0; i < numFrames; ++i) {
        size_t idx = (delayedReadPos + i) % ringCapacity_;
        inL[i] = lookaheadRingL_[idx];
        inR[i] = lookaheadRingR_[idx];
    }

    // Step 4: 64-bit Modular DSP Pipeline with Summing and WetDryMixer
    processDsdPipeline64(inL, inR, numFrames, cfg);

    // Step 5: Write processed 64-bit back to float buffer
    for (int i = 0; i < numFrames; ++i) {
        buffer[i * 2] = static_cast<float>(inL[i]);
        buffer[i * 2 + 1] = static_cast<float>(inR[i]);
    }

    // Update Telemetry
    {
        std::lock_guard<std::mutex> lock(telemetryMutex_);
        currentTelemetry_.peakEnergy = static_cast<float>(peakEnergy_);
        currentTelemetry_.attackSpeed = static_cast<float>(attackSpeed_);
        currentTelemetry_.transientDensity = static_cast<float>(transientDensity_);
        currentTelemetry_.spectralBalance = static_cast<float>(spectralBalance_);
        currentTelemetry_.frequencyFocus = static_cast<float>(frequencyFocus_);
        currentTelemetry_.crestFactor = static_cast<float>(crestFactor_);
        currentTelemetry_.rmsEnergy = static_cast<float>(rmsEnergy_);
        currentTelemetry_.spectralCentroid = static_cast<float>(spectralCentroid_);
        currentTelemetry_.spectralFlux = static_cast<float>(spectralFlux_);
        currentTelemetry_.detectedAttack = detectedAttackType_;
        currentTelemetry_.detectedProfile = detectedProfile_;
        currentTelemetry_.neuralPunch = static_cast<float>(punchMod_);
        currentTelemetry_.neuralSoften = static_cast<float>(softenMod_);
        currentTelemetry_.neuralTexture = static_cast<float>(textureMod_);
        currentTelemetry_.neuralSat = static_cast<float>(satMod_);
        currentTelemetry_.transientActivity = static_cast<float>(transientPulse_);
        currentTelemetry_.qualityMeter = (cfg.oversampleFactor > 1) ? 1.0f : 0.95f;

        currentTelemetry_.inputRMS = liveInputRms_.load(std::memory_order_relaxed);
        currentTelemetry_.outputRMS = liveOutputRms_.load(std::memory_order_relaxed);
        currentTelemetry_.fftActivity = static_cast<float>(spectralFlux_);
        currentTelemetry_.transientCount = transientCounter_;
        currentTelemetry_.neuralActivity = static_cast<float>((punchMod_ + textureMod_ + satMod_ + softenMod_) * 0.25);
        currentTelemetry_.dspRoutingStatus = dspRoutingStatus_.load(std::memory_order_relaxed);
        currentTelemetry_.forceTestModeActive = cfg.forceTestMode;
        currentTelemetry_.deltaModeActive = cfg.deltaMode;
        currentTelemetry_.wetDryMix = cfg.wetMix;
    }
}

void NeuralTransientTextureModule::getTelemetry(TelemetryData& outData) const {
    std::lock_guard<std::mutex> lock(telemetryMutex_);
    outData = currentTelemetry_;
}

void NeuralTransientTextureModule::getSpectrum(float* outSpectrum, int numBins) const {
    if (!outSpectrum || numBins <= 0) return;
    int count = std::min(numBins, 64);
    for (int i = 0; i < count; ++i) {
        outSpectrum[i] = spectrumBins_[i].load(std::memory_order_relaxed);
    }
    for (int i = count; i < numBins; ++i) {
        outSpectrum[i] = 0.0f;
    }
}
