#include "PluginProcessor.h"

namespace ParamIDs
{
    static constexpr auto bypass  = "bypass";
    static constexpr auto sens    = "sensitivity";
    static constexpr auto look    = "lookahead";
    static constexpr auto wet     = "wetmix";
    static constexpr auto punch   = "punch";
    static constexpr auto soften  = "soften";
    static constexpr auto texture = "texture";
    static constexpr auto sat     = "saturation";
}

juce::AudioProcessorValueTreeState::ParameterLayout NeuralTransientProcessor::createLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterBool>(ParamIDs::bypass, "Bypass", false));
    layout.add(std::make_unique<juce::AudioParameterFloat>(ParamIDs::sens, "Sensitivity", 0.0f, 1.0f, 0.5f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(ParamIDs::look, "Lookahead (ms)", 2.0f, 500.0f, 10.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(ParamIDs::wet, "Wet Mix", 0.0f, 1.0f, 1.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(ParamIDs::punch, "Punch", 0.0f, 1.0f, 0.5f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(ParamIDs::soften, "Soften", 0.0f, 1.0f, 0.4f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(ParamIDs::texture, "Texture", 0.0f, 1.0f, 0.35f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(ParamIDs::sat, "Saturation", 0.0f, 1.0f, 0.3f));
    return layout;
}

NeuralTransientProcessor::NeuralTransientProcessor()
    : AudioProcessor(BusesProperties()
                        .withInput("Input", juce::AudioChannelSet::stereo(), true)
                        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createLayout())
{
}

void NeuralTransientProcessor::prepareToPlay(double sampleRate, int)
{
    module.setSampleRate(static_cast<int>(sampleRate));
    module.reset();
}

void NeuralTransientProcessor::pushParamsToModule()
{
    auto c = module.getConfig();
    c.bypass          = apvts.getRawParameterValue(ParamIDs::bypass)->load() > 0.5f;
    c.sensitivity     = apvts.getRawParameterValue(ParamIDs::sens)->load();
    c.lookaheadMs     = apvts.getRawParameterValue(ParamIDs::look)->load();
    c.wetMix          = apvts.getRawParameterValue(ParamIDs::wet)->load();
    c.punchEnabled    = true;
    c.punchAmount     = apvts.getRawParameterValue(ParamIDs::punch)->load();
    c.softEnabled     = true;
    c.softAmount      = apvts.getRawParameterValue(ParamIDs::soften)->load();
    c.textureEnabled  = true;
    c.textureAmount   = apvts.getRawParameterValue(ParamIDs::texture)->load();
    c.saturateEnabled = true;
    c.saturateDrive   = apvts.getRawParameterValue(ParamIDs::sat)->load();
    module.setConfig(c);
}

void NeuralTransientProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    if (buffer.getNumChannels() < 2) return;

    pushParamsToModule();

    const int n = buffer.getNumSamples();
    interleaved.resize(static_cast<size_t>(n) * 2);

    const auto* inL = buffer.getReadPointer(0);
    const auto* inR = buffer.getReadPointer(1);
    for (int i = 0; i < n; ++i)
    {
        interleaved[static_cast<size_t>(i) * 2]     = inL[i];
        interleaved[static_cast<size_t>(i) * 2 + 1] = inR[i];
    }

    module.process(interleaved.data(), n);

    auto* outL = buffer.getWritePointer(0);
    auto* outR = buffer.getWritePointer(1);
    for (int i = 0; i < n; ++i)
    {
        outL[i] = interleaved[static_cast<size_t>(i) * 2];
        outR[i] = interleaved[static_cast<size_t>(i) * 2 + 1];
    }
}

juce::AudioProcessorEditor* NeuralTransientProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void NeuralTransientProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void NeuralTransientProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralTransientProcessor();
}
