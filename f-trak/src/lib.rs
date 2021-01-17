use opencv::prelude::*;
use std::path::Path;
use std::sync::mpsc;

pub fn test_func() {
    println!("Hello, world!");
}

type FaceBBox = ((i32, i32), (i32, i32));
pub struct FaceCapture {
   m_bbox : FaceBBox,
   m_bbox_transmitter: mpsc::Sender<FaceBBox>,
   m_terminate_transmitter: mpsc::Sender<bool>,
   m_protopath_str: String,
   m_modelpath_str: String,
   m_min_confidence: f32
}

impl FaceCapture {
   pub fn new(bbox_transmitter: mpsc::Sender<FaceBBox>, 
              termination_transmitter : mpsc::Sender<bool>,
              protopath_str: String,
              modelpath_str: String,
              min_confidence: f32) -> Self {
      
      let bbox : FaceBBox = ((0, 0), (0, 0));
      Self {
         m_bbox: bbox,
         m_bbox_transmitter: bbox_transmitter,
         m_terminate_transmitter: termination_transmitter,
         m_protopath_str: protopath_str,
         m_modelpath_str: modelpath_str,
         m_min_confidence: min_confidence
      }
   }

   fn resize(frame: Mat, width: i32) -> Result<Mat, &'static str> {

      if width < 0 {
         return Err("F")
      }
      
      let frame_size = frame.size().unwrap();
      let h = frame_size.height as f64;
      let w = frame_size.width as f64;

      let r = width as f64 / w;
      let dim = opencv::core::Size::new(width, (h * r) as i32);
      
      let mut resized_frame = Mat::default().unwrap();

      let scale_factor = 1.0;
      let resize_result = opencv::imgproc::resize(&frame, 
                                                  &mut resized_frame, 
                                                  dim, 
                                                  scale_factor, 
                                                  scale_factor, 
                                                  3); //opencv::imgproc::InterpolationFlags::INTER_AREA);

      match resize_result {
         Ok(_) => return Ok(frame),
         Err(e) => {
            println!("Some error occured: {:?}", e);
            return Err("Failed to resize image");
         },
      }
   }

   pub fn begin_capture(&mut self) {
      let protopath = Path::new(&self.m_protopath_str);
      let modelpath = Path::new(&self.m_modelpath_str);

      let mut exists = protopath.exists();

      match exists {
         true => {},
         false => println!("Failed to find proto file"),
      }

      exists = modelpath.exists();

      match exists {
         true => {},
         false => println!("Failed to find model file"),
      }

      let prototxt = protopath.to_str().unwrap();
      let model = modelpath.to_str().unwrap();

      println!("[INFO] loading model...");
      let net_res = opencv::dnn::read_net_from_caffe(prototxt, model);
      let mut net = opencv::dnn::Net::default().unwrap();
      
      match net_res {
         Ok(v) => net = v,
         Err(e) => println!("Some error occured: {:?}", e),
      }

      //Todo: allow user to select camera
      let mut camera = opencv::videoio::VideoCapture::new(0, 0 /*opencv::videoio::VideoCaptureAPIs::CAP_ANY*/).unwrap();
   
      let window_name = "f-trak";

      let win_created = opencv::highgui::named_window(window_name, 1 /*opencv::highgui::WindowFlags::WINDOW_AUTOSIZE*/);

      match win_created {
         Ok(v) => println!("Successfully set up camera!{:?}", v),
         Err(e) => println!("Some error occured: {:?}", e),
      }

      loop {

         let mut frame = Mat::default().unwrap();

         let success = camera.read(&mut frame);

         if success.unwrap() {
            
            let mean = opencv::core::Scalar::new(104.0, 177.0, 123.0, 1.0);
            let scale_factor = 1.0;

            frame = FaceCapture::resize(frame, 400).unwrap();

            //Needed later to draw boxes
            let frame_size = frame.size().unwrap();
            let h = frame_size.height;
            let w = frame_size.width;

            let mut resized_frame = Mat::default().unwrap();
            let resize_result = opencv::imgproc::resize(&frame, 
                                                        &mut resized_frame, 
                                                        opencv::core::Size::new(300, 300), 
                                                        scale_factor, 
                                                        scale_factor, 
                                                        1); //opencv::imgproc::InterpolationFlags::INTER_LINEAR

           match resize_result {
               Ok(_) => {},
               Err(e) => println!("Failed to resize image {:?}", e),
            }
            
            let blob = opencv::dnn::blob_from_image(&mut resized_frame, 
                                                    scale_factor, 
                                                    opencv::core::Size::new(300, 300), 
                                                    mean, 
                                                    false, 
                                                    false, 
                                                    opencv::core::CV_32F).unwrap();

            let set_input_result = net.set_input(&blob, "", scale_factor, opencv::core::Scalar::default());

            match set_input_result {
               Ok(_) => {},
               Err(e) => println!("Failed to set_input {:?}", e),
            }

            //expected detection shape (1, 1, *some val*, 7)
            let mut detections = opencv::types::VectorOfMat::new();

            let output_name = String::from("detection_out");
            let mut blob_names = opencv::types::VectorOfString::new();
            blob_names.push(&output_name);
            
            let forward_result = net.forward(&mut detections, &blob_names);
            
            match  forward_result {
               Ok(_) => {},
               Err(e) => println!("Failed to  forward_result {:?}", e),
            }

            for detection in detections {
               let confidence_index: [i32; 4] = [0, 0, 0, 2];
               let confidence = detection.at_nd::<f32>(&confidence_index);
               
               if *confidence.unwrap() < self.m_min_confidence {
                  continue;
               }

               let start_x_index: [i32; 4] = [0, 0, 0, 3];
               let start_y_index: [i32; 4] = [0, 0, 0, 4];
               let end_x_index: [i32; 4] = [0, 0, 0, 5];
               let end_y_index: [i32; 4] = [0, 0, 0, 6];
               
               let raw_start_x = detection.at_nd::<f32>(&start_x_index).unwrap();
               let raw_start_y = detection.at_nd::<f32>(&start_y_index).unwrap();
               let raw_end_x = detection.at_nd::<f32>(&end_x_index).unwrap();
               let raw_end_y = detection.at_nd::<f32>(&end_y_index).unwrap();

               let mut start_x = raw_start_x.clone();
               let mut start_y = raw_start_y.clone();
               let mut end_x = raw_end_x.clone();
               let mut end_y = raw_end_y.clone();

               //println!("Before mult ({:?}, {:?}), ({:?}, {:?})", start_x, start_y, end_x, end_y);

               start_x *= w as f32;
               end_x *= w as f32;
               start_y *= h as f32;
               end_y *= h as f32;

               //println!("After mult ({:?}, {:?}), ({:?}, {:?})", start_x, start_y, end_x, end_y);

               let y = |starting_y: f32| -> f32 {
                  if starting_y - 10.0 > 10.0 {
                     return starting_y - 10.0;
                  }
                  else {
                     return starting_y + 10.0;
                  }
               }(start_y);


               let start_pt = opencv::core::Point2i::new(start_x as i32, y as i32);
               let end_pt = opencv::core::Point2i::new(end_x as i32, end_y as i32);
               self.m_bbox = ((start_pt.x, start_pt.y), (end_pt.x, end_pt.y));
               self.m_bbox_transmitter.send(self.m_bbox.clone()).unwrap();

               let rect = opencv::core::Rect::from_points(start_pt, end_pt);

               let draw_rect_result = opencv::imgproc::rectangle(&mut frame, 
                                                                 rect, 
                                                                 opencv::core::Scalar::new(255.0, 0.0, 0.0, 1.0),
                                                                 2,
                                                                 1, //opencv::imgproc::LineTypes::LINE_4, 
                                                                 0);

               match draw_rect_result {
                  Ok(_) => {},
                  Err(e) => println!("Some error occured: {:?}", e),
               }
            }
               //extract confidence and compare to min confidence
               //continue if min confidence not met

               //else extract box coords and draw on frame

            let imshow_res = opencv::highgui::imshow(window_name, &frame);
            match imshow_res {
               Ok(_) => {},
               Err(e) => println!("Failed to imshow {:?}", e),
            }
         }

         let key = opencv::highgui::wait_key(1).unwrap();

         if key%256 == 27 {
            println!("Escape hit, closing...");
            self.m_terminate_transmitter.send(true).unwrap();
            break;
         }
         else
         {
            self.m_terminate_transmitter.send(false).unwrap();
         }
      }

      ()
   }
}