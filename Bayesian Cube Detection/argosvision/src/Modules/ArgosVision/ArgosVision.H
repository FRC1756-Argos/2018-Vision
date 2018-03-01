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
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#define SNAP_PATH "~/data/snaps"
#define YML_FILE "training_data.yml"


static jevois::ParameterCategory const ParamCateg("ArgosVision Options");

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(maxnumobj, size_t, "Max number of objects to declare a clean image. If more blobs are "
			 "detected in a frame, we skip that frame before we even try to analyze shapes of the blobs",
                         20, ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(erodesize, size_t, "Erosion structuring element size (pixels), or 0 for no erosion",
                         2, ParamCateg);
                         
//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(target_area_range, jevois::Range<unsigned int>, "Range of acceptable area of the detected vision targets",
                         jevois::Range<unsigned int>(50, 55000), ParamCateg);

                         //! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(target_area, jevois::Range<unsigned int>, "Range of acceptable area of the detected vision targets",
                         jevois::Range<unsigned int>(200, 80000), ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(dilatesize, size_t, "Dilation structuring element size (pixels), or 0 for no dilation",
                         4, ParamCateg);

//! Parameter \relates ArgosVision
JEVOIS_DECLARE_PARAMETER(debug, bool, "Show contours of all object candidates if true",
                         false, ParamCateg);



#endif
