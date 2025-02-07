#ifndef CONFIGURATION_YOLOX_X_HPP
#define CONFIGURATION_YOLOX_X_HPP

#include "Types.hpp"
using namespace mllm;

struct YOLOXNameConfig {
    string backbone_stem_name = "backbone.backbone.stem.";
    string backbone_dark_name = "backbone.backbone.dark";
    string backbone_lateral_name = "backbone.lateral_conv0.";
    string backbone_c3p4_name = "backbone.C3_p4.";
    string backbone_reduce_name = "backbone.reduce_conv1.";
    string backbone_c3p3_name = "backbone.C3_p3.";
    string backbone_bu_name = "backbone.bu_conv";
    string backbone_c3n3_name = "backbone.C3_n3.";
    string backbone_c3n4_name = "backbone.C3_n4.";
    string head_name = "head.";

    void init() {
        // Add any initialization if needed
    }
};

class YOLOXConfig {
public:
    // Model architecture parameters
    int num_classes = 80; // COCO dataset classes
    int input_height = 640;
    int input_width = 640;
    int channels = 3;

    // Backbone parameters
    int stem_out_channels = 32;
    int dark2_out_channels = 64;
    int dark3_out_channels = 128;
    int dark4_out_channels = 256;
    int dark5_out_channels = 512;

    // Head parameters
    vector<int> strides{8, 16, 32};
    vector<int> in_channels{128, 256, 512};
    vector<int> act_channels{256, 512, 1024};

    YOLOXNameConfig names_config;

    explicit YOLOXConfig() {
        names_config.init();
    }
};

#endif // CONFIGURATION_YOLOX_X_HPP
