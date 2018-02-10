// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Argos Vision module
//! This module provides vision processing to Argos for FRC 2018. List of parameters and their deatils can found
//  in the respective config file.
//
//  @author Argos (Teja Maddala)
//
//  @videomapping YUYV 640 480 28.5 YUYV 640 480 28.5 SampleVendor ArgosVision
//  @email
//  @address Peoria, IL
//  @copyright Copyright (C) 2017 by Argos
//  @mainurl http://teamargos.org
//  @license GPL v3
//  @distribution Unrestricted
//  @restrictions None
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef ARGOSVISION_H
#define ARGOSVISION_H


#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>

#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Util/Coordinates.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


enum detectionMode
{
  VISION_TARGETS = 1,
  PLATFORM = 2,
  POWER_CUBE = 3
};

static jevois::ParameterCategory const ParamCateg("ArgosVision Options");

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(detect_mode, unsigned char, "Use to switch detection mode, 1 - VISION_TARGET, 2 - PLATFORM, 3 - POWER_CUBE", 
                                       1, ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(hrange_target, jevois::Range<unsigned char>, "Target Hue Range",
				                jevois::Range<unsigned char>(45, 60), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(srange_target, jevois::Range<unsigned char>, "Target Hue Range",
				                jevois::Range<unsigned char>(50, 255), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(vrange_target, jevois::Range<unsigned char>, "Target Hue Range",
				                jevois::Range<unsigned char>(50, 255), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(hrange_red_platform, jevois::Range<unsigned char>, "Red Platform Hue Range",
				                jevois::Range<unsigned char>(1, 11), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(hrange_blue_platform, jevois::Range<unsigned char>, "Blue Platform Hue Range",
				                jevois::Range<unsigned char>(1, 11), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(srange_platform, jevois::Range<unsigned char>, "Platform Saturation Range",
				                jevois::Range<unsigned char>(100, 255), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(vrange_platform, jevois::Range<unsigned char>, "Platform Value Range",
				                jevois::Range<unsigned char>(50, 255), ParamCateg);
//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(hrange_powercube, jevois::Range<unsigned char>, "powercube Hue Range",
				                jevois::Range<unsigned char>(20, 30), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(srange_powercube, jevois::Range<unsigned char>, "powercube Hue Range",
				                jevois::Range<unsigned char>(100, 255), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(vrange_powercube, jevois::Range<unsigned char>, "powercube Hue Range",
				                jevois::Range<unsigned char>(50, 255), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(maxnumobj, size_t, "Max number of objects to declare a clean image. If more blobs are "
			 "detected in a frame, we skip that frame before we even try to analyze shapes of the blobs",
                         20, ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(erodesize, size_t, "Erosion structuring element size (pixels), or 0 for no erosion",
                         2, ParamCateg);
                         
//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(target_area, jevois::Range<unsigned int>, "Range of acceptable area of the detected vision targets",
                         jevois::Range<unsigned int>(50, 5000), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(dilatesize, size_t, "Dilation structuring element size (pixels), or 0 for no dilation",
                         4, ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(debug, bool, "Show contours of all object candidates if true",
                         false, ParamCateg);

class ArgosVision : public jevois::StdModule,
                    public jevois::Parameter<detect_mode, hrange_target, srange_target, vrange_target, hrange_red_platform, hrange_blue_platform, srange_platform,
                                             vrange_platform, hrange_powercube, srange_powercube, vrange_powercube, maxnumobj, target_area, erodesize, 
                                             dilatesize, debug>
{

  private:
    //width of the input frame
    unsigned int m_w;
    //height of the input frame
    unsigned int m_h;
    
  public:

    //! Default base class constructor 
    using jevois::StdModule::StdModule;
    
    ArgosVision();
    //! Virtual destructor for safe inheritance
    virtual ~ArgosVision();
    
    ////////////////////////////////////////////////////////////////////////////////
    //@brief: Called by Jevois whenever there is a new frame avalibale to process
    //@param: input frame and output frame
    //@return: void
    ////////////////////////////////////////////////////////////////////////////////
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override;

    ////////////////////////////////////////////////////////////////////////////////
    //@brief: Filtering of VisionTargets, groups double targets as a single target
    //        and sends out serial message of left and right targets with labels
    //        LEFT_T and RIGHT_T respectively
    //@param: vector of vector contours and vector of hierarchy of contours
    //@return: void
    ////////////////////////////////////////////////////////////////////////////////
    void detectAndGroupTargets(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg);

    ////////////////////////////////////////////////////////////////////////////////
    //@brief: Filtering of Platform, groups largest detected red and blue platform and
    //        sends serial messages with iabels RED_P and BLUE_P respectively
    //@param: vector of vector red and blue contours and vector of hierarchy of 
    //        re and blue contours
    //@return: void
    ////////////////////////////////////////////////////////////////////////////////
    void detectPlatform(const std::vector<std::vector<cv::Point>> &contours, const std::vector<std::vector<cv::Point>> &contours2,
                        const std::vector<cv::Vec4i> &hierarchy, const std::vector<cv::Vec4i> &hierarchy2, jevois::RawImage &outimg);

    ////////////////////////////////////////////////////////////////////////////////
    //@brief: Filtering and detection PowerCubes
    //@param: vector of vector contours and vector of hierarchy of contours
    //@return: void
    ////////////////////////////////////////////////////////////////////////////////
    void detectPowerCubes(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg);

};

#endif
