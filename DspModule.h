#pragma once

class DspModule {
public:
    virtual ~DspModule() = default;
    
    // Process a stereo buffer
    // Interleaved format: L, R, L, R...
    virtual void process(float* buffer, int numFrames) = 0;
    virtual void setSampleRate(int sampleRate) = 0;
    virtual void reset() {}
    virtual void setBypass(bool bypass) { bypass_ = bypass; }
    bool isBypassed() const { return bypass_; }

    void setId(int id) { id_ = id; }
    int getId() const { return id_; }

protected:
    bool bypass_ = false;
    int sampleRate_ = 48000;
    int id_ = 0;
};
