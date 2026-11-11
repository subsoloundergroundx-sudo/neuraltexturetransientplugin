cmake_minimum_required(VERSION 3.22)
project(NeuralTransient VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(NeuralTransient
    COMPANY_NAME "Penizudo"
    PLUGIN_MANUFACTURER_CODE Pnzd
    PLUGIN_CODE Ntrn
    FORMATS VST3
    PRODUCT_NAME "Neural Transient")

target_sources(NeuralTransient PRIVATE
    Source/PluginProcessor.cpp
    Source/NeuralTransientTextureModule.cpp)

target_include_directories(NeuralTransient PRIVATE Source)

target_compile_definitions(NeuralTransient PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(NeuralTransient PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_basics)
