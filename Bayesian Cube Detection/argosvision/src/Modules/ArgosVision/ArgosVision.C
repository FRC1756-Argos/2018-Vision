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
                    public jevois::Parameter< maxnumobj, cube_area_range, erodesize, dilatesize, debug>
{
  public:
    //!constructor
    using jevois::StdModule::StdModule;
   /* ArgosVision(std::string const & instance) : jevois::StdModule(instance),
                                                m_focalLength(5),
                                                snapRequested(false)*/

    //! Virtual destructor for safe inheritance
    virtual ~ArgosVision() { }
    
    // get the trained model from yml file
    cv::Mat getProbabilityDistribution(std::string data, std::string ymlfilename);

    // track detected powercube and send serial message
    void trackPowerCubes(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg);

    //to detect power cubes
    cv::Mat detectPowerCubes(const cv::Mat &imghsv);

    //to record video 
    void takeSnap(const cv::Mat &rgbimg, const int& snapNumber);

    // Receive a string from a serial port which contains a user command

    void parseSerial(std::string const & str, std::shared_ptr<jevois::UserInterface> s) override
    {
      if (str == "snap")
      {
        snapRequested.store(true);
      }
    }

    //! Processing function
    void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");
      static int snapNumber = 0;
      snapRequested.store(false);
      m_focalLength = 5;

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

      if(snapRequested.load())
      {
        snapNumber++;
        takeSnap(imgbgr, snapNumber);
        snapRequested.store(false);
      }
      
      // Wait for paste to finish up:
      image_pasting_thread.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      cv::Mat imghsv;
      cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

      //get probability map
      cv::Mat imgth = detectPowerCubes(imghsv);
      
      // Apply morphological operations to cleanup the image noise:
      cv::Mat erodeElement = getStructuringElement(cv::MORPH_RECT, cv::Size(erodesize::get(), erodesize::get()));
      cv::erode(imgth, imgth, erodeElement);

      cv::Mat dilateElement = getStructuringElement(cv::MORPH_RECT, cv::Size(dilatesize::get(), dilatesize::get()));
      cv::dilate(imgth, imgth, dilateElement);

      /*// Floodfill from point (0, 0)
      cv::Mat im_floodfill = imgth.clone();
      cv::floodFill(im_floodfill, cv::Point(0,0), cv::Scalar(255));
      
      // Invert floodfilled image
      cv::Mat im_floodfill_inv;
      bitwise_not(im_floodfill, im_floodfill_inv);
      
      // Combine the two images to get the foreground.
      imgth = (imgth | im_floodfill_inv);*/

      // Detect cube by finding contours:
      std::vector<std::vector<cv::Point>> contours; 
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(imgth, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

      trackPowerCubes(contours, hierarchy, outimg);

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

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, m_h - 13, jevois::yuyv::White);

      // Possibly wait until all contours are drawn, if they had been requested:
      if (draw_fut.valid()) 
      {
        draw_fut.get();
      }
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    private:
    //width of the input frame
    unsigned int m_w;
    //height of the input frame
    unsigned int m_h;
    //focal length
    double m_focalLength;
    //snap indicator
    std::atomic<bool> snapRequested;
};

////////////////////////////////////////////////////////////////////////////////
//@brief: get data from the yml file
//@param: data to read and ymlfilename
//@return: matrix of data
////////////////////////////////////////////////////////////////////////////////
cv::Mat ArgosVision::getProbabilityDistribution(std::string data, std::string ymlfilename)
{
  //cv::Mat temp = cv::Mat(256, 256, CV_64F, double(0));;
  cv::Mat temp;
  cv::FileStorage fs(ymlfilename, cv::FileStorage::READ);
  fs[data] >> temp;
  fs.release();
  return temp;
}

////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering and detection PowerCubes
//@param: vector of vector contours and vector of hierarchy of contours
//@return: thresholded posterior probability map
////////////////////////////////////////////////////////////////////////////////
cv::Mat ArgosVision::detectPowerCubes(const cv::Mat &imghsv)
{
  uchar Hue, Sat;
  cv::Mat Posterior_probability_map;

  double pr_pixel_equals_0 = 0.9653;
  double pr_pixel_equals_1 = 1 - pr_pixel_equals_0;

  cv::Mat Pr_pixel_equals_1_given_cube = cv::Mat(imghsv.rows, imghsv.cols, CV_8UC1, float(0));
  //cv::Mat Pr_cube_given_pixel_equals_1 = cv::Mat(imghsv.rows, imghsv.cols, CV_64F, double(0));
  //cv::Mat Pr_cube_given_pixel_equals_0 = cv::Mat(imghsv.rows, imghsv.cols, CV_64F, double(0));
  cv::Mat Pr_cube_given_pixel_equals_1 = getProbabilityDistribution("Pr_cube_given_pixel_equals_1", std::string(YML_FILE));
  cv::Mat Pr_cube_given_pixel_equals_0 = getProbabilityDistribution("Pr_cube_given_pixel_equals_0", std::string(YML_FILE));

  for (int i = 0; i < imghsv.rows; ++i)
  {
      const cv::Vec3b* pixel = imghsv.ptr<cv::Vec3b>(i);
      for (int j = 0; j < imghsv.cols; ++j)
      {
          Hue = pixel[j][0];
          Sat = pixel[j][1];
          
          double pr_cube = (Pr_cube_given_pixel_equals_1.at<float>(Hue, Sat)*pr_pixel_equals_1)
                          + (Pr_cube_given_pixel_equals_0.at<float>(Hue, Sat)*pr_pixel_equals_0);
                          
          pr_cube = pr_cube + 0.00001; //avoid divided by zero
          
          Pr_pixel_equals_1_given_cube.at<float>(i, j) = (Pr_cube_given_pixel_equals_1.at<float>(Hue, Sat)*pr_pixel_equals_1)/pr_cube;
      }
  }
  
  cv::threshold(Pr_pixel_equals_1_given_cube, Posterior_probability_map, 0.3, 255, cv::THRESH_BINARY);

  return Pr_pixel_equals_1_given_cube;

}

////////////////////////////////////////////////////////////////////////////////
//@brief: Filtering and detection PowerCubes
//@param: vector of vector contours and vector of hierarchy of contours
//@return: void
////////////////////////////////////////////////////////////////////////////////
void ArgosVision::takeSnap(const cv::Mat &rgbimg, const int& snapNumber)
{
  std::string const cmd = "/bin/mkdir -p " + std::string(SNAP_PATH);
  if (std::system(cmd.c_str()) == 0)
  {
    std::string imageName = std::string(SNAP_PATH) + "snap_" + std::to_string(snapNumber) + ".png";
    cv::imwrite(imageName, rgbimg);
  }
}
  

////////////////////////////////////////////////////////////////////////////////
//@brief: tracking PowerCubes
//@param: vector of vector contours and vector of hierarchy of contours
//@return: void
////////////////////////////////////////////////////////////////////////////////
void ArgosVision::trackPowerCubes(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, jevois::RawImage &outimg)
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
      m_focalLength = ((brect.width * 2540.0) / 127.0); //in mm
      //d = (5*m_focallength)/brect.width;
      // Send coords to serial port
      sendSerialStd3D(x, y, d, brect.width, brect.height, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, "POWER_CUBE", "POWER_CUBE");
    }
  }
  // Show number of detected powercubes
  jevois::rawimage::writeText(outimg, "Detected " + std::to_string(powerCubes) +  " PowerCube of area:" + std::to_string(cubeArea) + "AR: " + std::to_string(aspectRatio) + 
                              "f: " + std::to_string(m_focalLength), 3, m_h + 2, jevois::yuyv::White);
}

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ArgosVision);
