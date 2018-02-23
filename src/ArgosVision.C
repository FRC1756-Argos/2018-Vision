// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Argos Vision module
//! This module provides vision processing to Argos for FRC 2018. List of parameters and their deatils can found
//  in the respective config file.
//
//  @author Argos1756
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

    
class ArgosVision : public jevois::StdModule,
                    public jevois::Parameter<detect_mode, hrange_target, srange_target, vrange_target, hrange_red_platform, hrange_blue_platform, srange_platform,
                                             vrange_platform, hrange_powercube, srange_powercube, vrange_powercube, maxnumobj, target_area, erodesize, 
                                             dilatesize, debug, tc1>
{

  private:
    //width of the input frame
    unsigned int m_w;
    //height of the input frame
    unsigned int m_h;
    //focal length
    double m_focallength;
    //minimum distance to group targets
    static constexpr double MINIMUM_TARGET_GROUPING_DISTANCE = 75.0; //pixels

    VisionTarget Target;
  public:
    //! Default base class constructor ok
    using jevois::StdModule::StdModule;

    //! Virtual destructor for safe inheritance
    virtual ~ArgosVision() { }
    
    //to detect targets
    void detectTargets(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg);
    //to detect platforms
    void detectPlatform(const std::vector<std::vector<cv::Point>> &contours, const std::vector<std::vector<cv::Point>> &contours2,
                        const std::vector<cv::Vec4i> &hierarchy, const std::vector<cv::Vec4i> &hierarchy2, jevois::RawImage &outimg);
    //to detect power cubes
    void detectPowerCubes(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg);
    //to pair vision targets
    std::vector<VisionTarget> groupTargets(const std::vector<std::vector<cv::Point>> &targets);
    //to find visiontarget pairs based on the distance
    bool isClose(const std::vector<cv::Point> &target1, const std::vector<cv::Point> &target2);
    //to find distance between points
    double getDistance(double X1, double Y1, double X2, double Y2);

    //! Processing function
    void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) 
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
      cv::erode(imgth, imgth, erodeElement);

      cv::Mat dilateElement = getStructuringElement(cv::MORPH_RECT, cv::Size(dilatesize::get(), dilatesize::get()));
      cv::dilate(imgth, imgth, dilateElement);

      // Detect objects by finding contours:
      std::vector<std::vector<cv::Point>> contours, contours2; 
      std::vector<cv::Vec4i> hierarchy, hierarchy2;
      cv::findContours(imgth, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
      if (detectionMode == PLATFORM)
      {
        cv::erode(imgth, imgth, erodeElement);
        cv::dilate(imgth, imgth, dilateElement);
        cv::findContours(imgth2, contours2, hierarchy2, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
      }

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

      switch(detectionMode)
      {
        case VISION_TARGETS:
            detectTargets(contours, hierarchy, outimg);
            break;
        case PLATFORM:
            detectPlatform(contours, contours2, hierarchy, hierarchy2, outimg);
            break;           
        case POWER_CUBE:
            detectPowerCubes(contours, hierarchy, outimg);
            break;
        default:
            //Vision Target
            detectTargets(contours, hierarchy, outimg);
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, m_h - 13, jevois::yuyv::White);

      // Possibly wait until all contours are drawn, if they had been requested:
      if (draw_fut.valid()) draw_fut.get();
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }
};

////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering of VisionTargets, groups double targets as a single target
//        and sends out serial message of left and right targets with labels
//        LEFT_T and RIGHT_T respectively
//@param: vector of vector contours and vector of hierarchy of contours
//@return: void
////////////////////////////////////////////////////////////////////////////////
void ArgosVision::detectTargets(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg)
{
  int numOfTargets;
  std::vector<VisionTarget> Targets;

  if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
  {
    float x = 0.0;
    float y = 0.0;
    int refIdx = 0;
	  bool targetsFound = false;
    std::vector<std::vector<cv::Point>> targetContours;
 
    for (int index = 0; index >= 0; index = hierarchy[index][0])
    {
	    cv::Moments moment = moments((cv::Mat)contours[index]);
	    double area = moment.m00;

      if (target_area::get().contains(int(area + 0.4999)))
      {
	      VisionTarget target;
        x = moment.m10 / area + 0.4999;
        y = moment.m01 / area + 0.4999;
        refIdx = index;

		    target.setXpos(x);
		    target.setYpos(y);

        targetContours.push_back(contours[refIdx]);
		    Targets.push_back(target);

		    targetsFound = true;
      }
    }

    if(targetsFound)
    {
      TargetInfo targetInfo;
      std::vector<VisionTarget> groupedTargets;
      groupedTargets = groupTargets(targetContours);
      //Draw Targets
      for (const auto& targetContour: targetContours)
      {
        for (auto& target: Targets )
        {
          cv::Rect brect = cv::boundingRect( cv::Mat(targetContour) );
          targetInfo.width = brect.width;
          targetInfo.height = brect.height;
          jevois::rawimage::drawRect(outimg, (target.getXpos() - 0.5*brect.width), (target.getYpos() - brect.height*0.5), brect.width, brect.height, jevois::yuyv::LightGreen);
        }
      }
      //send labeled serial messages
      for (auto& groupedTarget: groupedTargets)
      {
        targetInfo.Xc = groupedTarget.getXpos();
        targetInfo.Yc = groupedTarget.getYpos();
        //draw crosshairs
        jevois::rawimage::drawLine(outimg, targetInfo.Xc-10, targetInfo.Yc, targetInfo.Xc+10, targetInfo.Yc, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, targetInfo.Xc, targetInfo.Yc-10, targetInfo.Xc, targetInfo.Yc+10, 1, jevois::yuyv::LightGreen);

        switch(groupedTargets.size())
        {
          case 1: 
            sendSerialStd3D(targetInfo.Xc, targetInfo.Yc, targetInfo.distance, targetInfo.width, targetInfo.height, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, "SINGLE_T", "TARGET");
            break;
          case 2:
            if (x < 0)
            {
              sendSerialStd3D(targetInfo.Xc, targetInfo.Yc, targetInfo.distance, targetInfo.width, targetInfo.height, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, "LEFT_T", "TARGET");
            }
            else
            {
              sendSerialStd3D(targetInfo.Xc, targetInfo.Yc, targetInfo.distance, targetInfo.width, targetInfo.height, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, "RIGHT_T", "TARGET");
            }
            break;
          default:
            sendSerialStd3D(targetInfo.Xc, targetInfo.Yc, targetInfo.distance, targetInfo.width, targetInfo.height, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, "MULTI_T", "TARGET");
            break;      
        }
      }
    }
  }
  // Show number of detected targets:
  jevois::rawimage::writeText(outimg, "Detected " + std::to_string(numOfTargets) + " Targets.",
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
    
    if (refArea > 0.0)
    {
      ++platforms;
      jevois::rawimage::drawCircle(outimg, x, y, 20, 1, jevois::yuyv::LightGreen);
      sendSerialContour2D(m_w, m_h, contours[refIdx], "RED_P");
    }
  }

  if (hierarchy2.size() > 0 && hierarchy2.size() <= maxnumobj::get())
  {
    double refArea = 0.0; 
    int x = 0, y = 0; 
    int refIdx = 0;

    for (int index = 0; index >= 0; index = hierarchy[index][0])
    {
      cv::Moments moment = cv::moments((cv::Mat)contours2[index]);
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
      ++platforms;
      jevois::rawimage::drawCircle(outimg, x, y, 20, 1, jevois::yuyv::LightGreen);
      sendSerialContour2D(m_w, m_h, contours2[refIdx], "BLUE_P");
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
  double cubeArea = 0.0;
  double aspectRatio= 0.0;
 
  if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
  {
    double cubeArea = 0.0; 
    int x = 0, y = 0; 
    int refIdx = 0;
    float d = 0.0;

    for (int index = 0; index >= 0; index = hierarchy[index][0])
    {
      cv::Moments moment = cv::moments((cv::Mat)contours[index]);
      double area = moment.m00;
      if (target_area::get().contains(int(area + 0.4999)) && area > cubeArea )
      { 
        x = moment.m10 / area + 0.4999;
        y = moment.m01 / area + 0.4999;
        cubeArea = area;
        refIdx = index;
      }
    }
    
    if (cubeArea > 0.0)
    {
      ++powerCubes;
      cv::Rect brect = cv::boundingRect( cv::Mat(contours[refIdx]) );
      //jevois::rawimage::drawCircle(outimg, x, y, (brect.width*0.5), 1, jevois::yuyv::LightGreen);
      cubeArea = brect.width * brect.height;
      jevois::rawimage::drawRect(outimg, (x - 0.5*brect.width), (y - brect.height*0.5), brect.width, brect.height, jevois::yuyv::LightGreen);
	    jevois::rawimage::drawLine(outimg, x-10, y, x+10, y, 1, jevois::yuyv::LightGreen);
      jevois::rawimage::drawLine(outimg, x, y-10, x, y+10, 1, jevois::yuyv::LightGreen);
	    //jevois::rawimage::drawLine(outimg, x, y-25, x, y-25, 2, jevois::yuyv::LightGreen);

      
      aspectRatio = (brect.height/brect.width); 
      m_focallength = ((brect.width * 2540.0) / 127.0); //in mm
      //d = (5*m_focallength)/brect.width;
      // Send coords to serial port
      sendSerialStd3D(x, y, d, brect.width, brect.height, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, "POWER_CUBE", "POWER_CUBE");
    }
  }
  // Show number of detected powercubes
  jevois::rawimage::writeText(outimg, "Detected " + std::to_string(powerCubes) +  " PowerCube of area:" + std::to_string(cubeArea) + "AR: " + std::to_string(aspectRatio) + 
                              "f: " + std::to_string(m_focallength), 3, m_h + 2, jevois::yuyv::White);
}

////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering of VisionTargets, groups double targets as a single target
//@param: vector of targets
//@return: vector of grouped targets
////////////////////////////////////////////////////////////////////////////////
std::vector<VisionTarget> ArgosVision::groupTargets(const std::vector<std::vector<cv::Point>> &targets)
{
  double x, y, x1, x2, y1, y2;
  std::vector<VisionTarget> groupedTargets;
  
  for (uint8_t i = 0; i < targets.size(); i++)
  {
    for (uint8_t j = i+1; j < targets.size(); j++)
    {
      if(isClose(targets[i], targets[j]))
      {
        VisionTarget groupTarget;

        cv::Moments moment1 = cv::moments((cv::Mat)targets[i]);
        cv::Moments moment2 = cv::moments((cv::Mat)targets[j]);

	      double area1 = moment1.m00;
	      double area2 = moment2.m00;

        x1 = moment1.m10 / area1 + 0.4999;
        y1 = moment1.m01 / area1 + 0.4999;

        x2 = moment2.m10 / area2 + 0.4999;
        y2 = moment2.m01 / area2 + 0.4999;

        x = (x1 + x2) / 2.0;
        y = (y1 + y2) / 2.0;

		    groupTarget.setXpos(x);
		    groupTarget.setYpos(y);

		    groupedTargets.push_back(groupTarget);
      }
    }
  }
  return groupedTargets;
}

////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering of VisionTargets, groups double targets as a single target
//@param: vector of targets
//@return: vector of grouped targets
////////////////////////////////////////////////////////////////////////////////
bool ArgosVision::isClose(const std::vector<cv::Point> &T1, const std::vector<cv::Point> &T2)
{
  double x1, y1, x2, y2;
  cv::Moments moment1 = cv::moments((cv::Mat)T1);
  cv::Moments moment2 = cv::moments((cv::Mat)T2);

  double area1 = moment1.m00;
  double area2 = moment2.m00;

  x1 = moment1.m10 / area1 + 0.4999;
  y1 = moment1.m01 / area1 + 0.4999;

  x2 = moment2.m10 / area2 + 0.4999;
  y2 = moment2.m01 / area2 + 0.4999;

  if (getDistance(x1, y1, x2, y2) > MINIMUM_TARGET_GROUPING_DISTANCE)
  {
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
//@brief: compute distance between two points
//@param: cordinate points
//@return: distance in double
////////////////////////////////////////////////////////////////////////////////
double getDistance(double X1, double Y1, double X2, double Y2)
{
    return sqrt((X2 - X1)*(X2 - X1) + (Y2 - Y1)*(Y2 - Y1));
}

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ArgosVision);
