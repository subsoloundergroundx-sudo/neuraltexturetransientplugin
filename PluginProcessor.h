#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <vector>
#include "NeuralTransientTextureModule.h"

class NeuralTransientProcessor : public juce::AudioProcessor
{
public:
    NeuralTransientProcessor();
    ~NeuralTransientProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Neural Transient"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.1; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void pushParamsToModule();

    juce::AudioProcessorValueTreeState apvts;
    NeuralTransientTextureModule module;
    std::vector<float> interleaved;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralTransientProcessor)
};
