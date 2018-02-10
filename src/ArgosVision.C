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

#include "ArgosVision.H"


////////////////////////////////////////////////////////////////////////////////
//@brief: Called by Jevois whenever there is a new frame avalibale to process
//@param: input frame and output frame
//@return: void
////////////////////////////////////////////////////////////////////////////////
void ArgosVision::process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe)
{
  static jevois::Timer timer("processing");

  // get the detection mode: 1 - VISION_TARGET, 2 - PLATFORM, 3 - POWER_CUBE
  uint8_t detectionMode = detect_mode::get();

  // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
  jevois::RawImage inimg = inframe.get();
  
  //update width and height for further calculations and also as Jevois serial structure needs them  
  m_w = inimg.width;
  m_h = inimg.height;
  inimg.require("input", m_w, m_h, V4L2_PIX_FMT_YUYV);

  timer.start();

  // While we process it, start a thread to wait for output frame and paste the input image into it:
  jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
  auto image_pasting_thread = std::async(std::launch::async, [&]() {
      outimg = outframe.get();
      outimg.require("output", m_w, m_h + 14, inimg.fmt);
      jevois::rawimage::paste(inimg, outimg, 0, 0);
      jevois::rawimage::writeText(outimg, "ARGOS VISION", 3, 3, jevois::yuyv::White);
      jevois::rawimage::drawFilledRect(outimg, 0, m_h, m_w, outimg.height-m_h, 0x8000);
    });

  // Convert input image to BGR24, then to HSV:
  cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
  cv::Mat imghsv, imghsv2;
  cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

  // Threshold the HSV image to only keep pixels within the desired HSV range:
  cv::Mat imgth, imgth2;

  switch(detectionMode)
  {
    case VISION_TARGETS:
        cv::inRange(imghsv, cv::Scalar(hrange_target::get().min(), srange_target::get().min(), vrange_target::get().min()),
                    cv::Scalar(hrange_target::get().max(), srange_target::get().max(), vrange_target::get().max()), imgth);
        break;
    case PLATFORM:
        cv::inRange(imghsv, cv::Scalar(hrange_red_platform::get().min(), srange_platform::get().min(), vrange_platform::get().min()),
                    cv::Scalar(hrange_red_platform::get().max(), srange_platform::get().max(), vrange_platform::get().max()), imgth);
        imghsv2 = imghsv;
        cv::inRange(imghsv, cv::Scalar(hrange_blue_platform::get().min(), srange_platform::get().min(), vrange_platform::get().min()),
                    cv::Scalar(hrange_blue_platform::get().max(), srange_platform::get().max(), vrange_platform::get().max()), imgth2); 
        break;           
    case POWER_CUBE:
        cv::inRange(imghsv, cv::Scalar(hrange_powercube::get().min(), srange_powercube::get().min(), vrange_powercube::get().min()),
                    cv::Scalar(hrange_powercube::get().max(), srange_powercube::get().max(), vrange_powercube::get().max()), imgth);
        break;
    default:
        //Vision Target
        cv::inRange(imghsv, cv::Scalar(hrange_target::get().min(), srange_target::get().min(), vrange_target::get().min()),
                    cv::Scalar(hrange_target::get().max(), srange_target::get().max(), vrange_target::get().max()), imgth);
  }

  // Wait for paste to finish up:
  image_pasting_thread.get();

  // Let camera know we are done processing the input image:
  inframe.done();
  
  // Apply morphological operations to cleanup the image noise:
  cv::Mat erodeElement = getStructuringElement(cv::MORPH_RECT, cv::Size(erodesize::get(), erodesize::get()));
  cv::erode(imgth, imgth, erodeElement);

  cv::Mat dilateElement = getStructuringElement(cv::MORPH_RECT, cv::Size(dilatesize::get(), dilatesize::get()));
  cv::dilate(imgth, imgth, dilateElement);

  // Detect objects by finding contours:
  std::vector<std::vector<cv::Point>> contours, contours2; 
  std::vector<cv::Vec4i> hierarchy, hierarchy2;
  cv::findContours(imgth, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  if (detectionMode == PLATFORM)
    cv::findContours(imgth2, contours2, hierarchy2, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

  // If desired, draw all contours in a thread:
  std::future<void> draw_fut;
  if (debug::get())
  {
    draw_fut = std::async(std::launch::async, [&]() {
        // We reinterpret the top portion of our YUYV output image as an opencv 8UC2 image:
        cv::Mat outuc2(imgth.rows, imgth.cols, CV_8UC2, outimg.pixelsw<unsigned char>()); // pixel data shared
        for (size_t i = 0; i < contours.size(); ++i)
          cv::drawContours(outuc2, contours, i, jevois::yuyv::LightPink, 2, 8, hierarchy);
      });
  }

  //detect targets appropriately send Serial Messages to the serial port
  switch(detectionMode)
  {
    case VISION_TARGETS:
        detectAndGroupTargets(contours, hierarchy, outimg);
        break;
    case PLATFORM:
        detectPlatform(contours, contours2, hierarchy, hierarchy2, outimg);
        break;           
    case POWER_CUBE:
        detectPowerCubes(contours, hierarchy, outimg);
        break;
    default:
        //Vision Target
        detectAndGroupTargets(contours, hierarchy, outimg);
  }

  // Show processing fps:
  std::string const & fpscpu = timer.stop();
  jevois::rawimage::writeText(outimg, fpscpu, 3, m_h - 13, jevois::yuyv::White);

  // Possibly wait until all contours are drawn, if they had been requested:
  if (draw_fut.valid()) draw_fut.get();
  
  // Send the output image with our processing results to the host over USB:
  outframe.send();
}


////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering of VisionTargets, groups double targets as a single target
//        and sends out serial message of left and right targets with labels
//        LEFT_T and RIGHT_T respectively
//@param: vector of vector contours and vector of hierarchy of contours
//@return: void
////////////////////////////////////////////////////////////////////////////////
void ArgosVision::detectAndGroupTargets(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg)
{
  int targets = 0;
  if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
  {
    double refArea = 0.0; 
    float x1 = 0.0, x2 = 0.0;
    float y1 = 0.0, y2 = 0.0; 
    int refIdx_1, refIdx_2 = 0;
 
    for (int i = 0; i >= 0; i = hierarchy[i][0])
    {
      //skip for last target
      if (i-1 == hierarchy.size())
      {
        continue;
      }
      cv::Moments moment_1 = cv::moments((cv::Mat)contours[i]);
      cv::Moments moment_2 = cv::moments((cv::Mat)contours[i+1]);
      double area_1 = moment_1.m00;
      double area_2 = moment_2.m00;
      if (target_area::get().contains(int(area_1 + 0.4999)) && area_1 > refArea)
      { 
        x1 = moment_1.m10 / area_1 + 0.4999; 
        y1 = moment_1.m01 / area_1 + 0.4999;  
        refIdx_1 = i;
        if (target_area::get().contains(int(area_2 + 0.4999)) && area_2 > refArea)
        {
          x2 = moment_1.m10 / area_2 + 0.4999; 
          y2 = moment_1.m01 / area_2 + 0.4999; 
          refIdx_2 = i+1;
        }
      }
    }

    float x = (x1+x2)/2;
    float y = (y1+y2)/2;
    float W, H;

    //TODO: Add contours and find grouped bounding rectangle, refArea should belong to that one, and classify left and right based on y direction

    if (refArea > 0.0)
    {
      ++targets;
      jevois::rawimage::drawCircle(outimg, x, y, 20, 1, jevois::yuyv::LightGreen);

      // Send coords to serial port
      sendSerialImg2D(m_w, m_h, x, y, W, H, "LEFT_T");
    }
  }
  // Show number of detected targets:
  jevois::rawimage::writeText(outimg, "Detected " + std::to_string(targets) + " Targets.",
                          3, m_h + 2, jevois::yuyv::White);
}


////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering of Platform, groups largest detected red and blue platform and
//        sends serial messages with iabels RED_P and BLUE_P respectively
//@param: vector of vector red and blue contours and vector of hierarchy of 
//        re and blue contours
//@return: void
////////////////////////////////////////////////////////////////////////////////
void ArgosVision::detectPlatform(const std::vector<std::vector<cv::Point>> &contours, const std::vector<std::vector<cv::Point>> &contours2,
                    const std::vector<cv::Vec4i> &hierarchy, const std::vector<cv::Vec4i> &hierarchy2, jevois::RawImage &outimg)
{
  int platforms = 0;
  if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
  {
    double refArea = 0.0; 
    int x = 0, y = 0; 
    int refIdx = 0;

    for (int index = 0; index >= 0; index = hierarchy[index][0])
    {
      cv::Moments moment = cv::moments((cv::Mat)contours[index]);
      double area = moment.m00;
      if (target_area::get().contains(int(area + 0.4999)) && area > refArea)
      { 
        x = moment.m10 / area + 0.4999; 
        y = moment.m01 / area + 0.4999; 
        refArea = area; 
        refIdx = index; 
      }
    }
    
    //TODO: process countours2 and hierarchy2 similarly
    
    if (refArea > 0.0)
    {
      ++platforms;
      jevois::rawimage::drawCircle(outimg, x, y, 20, 1, jevois::yuyv::LightGreen);

      // Send coords to serial port
      sendSerialContour2D(m_w, m_h, contours[refIdx], "RED_P");
    }
  }
  // Show detected platform
  jevois::rawimage::writeText(outimg, "Detected " + std::to_string(platforms) + " Platform.",
                              3, m_h + 2, jevois::yuyv::White);
}

////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering and detection PowerCubes
//@param: vector of vector contours and vector of hierarchy of contours
//@return: void
////////////////////////////////////////////////////////////////////////////////
void ArgosVision::detectPowerCubes(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg)
{
  int powerCubes = 0;
  if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
  {
    double refArea = 0.0; 
    int x = 0, y = 0; 
    int refIdx = 0;

    for (int index = 0; index >= 0; index = hierarchy[index][0])
    {
      cv::Moments moment = cv::moments((cv::Mat)contours[index]);
      double area = moment.m00;
      if (target_area::get().contains(int(area + 0.4999)) && area > refArea)
      { 
        x = moment.m10 / area + 0.4999; 
        y = moment.m01 / area + 0.4999; 
        refArea = area; 
        refIdx = index; 
      }
    }
    
    if (refArea > 0.0)
    {
      ++powerCubes;
      jevois::rawimage::drawCircle(outimg, x, y, 20, 1, jevois::yuyv::LightGreen);

      // Send coords to serial port
      sendSerialContour2D(m_w, m_h, contours[refIdx], "POWER_CUBE");
    }
  }
  // Show number of detected powercubes
  jevois::rawimage::writeText(outimg, "Detected " + std::to_string(powerCubes) + " Power Cube.",
                              3, m_h + 2, jevois::yuyv::White);
}

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ArgosVision);
