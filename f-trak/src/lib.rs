use opencv::prelude::*;

pub fn test_func() {
    println!("Hello, world!");
}

pub struct FaceCapture {
   //bbox : ((i32, i32), (i32, i32)),
}

impl FaceCapture {
   pub fn new() -> Self {
      Self {
         //bbox: ((0,0), (0,0))
      }
   }

   pub fn begin_capture() {
      //Todo: allow users to configure their own model architecture and weights
      //let prototxt = "../f-trak/static/deploy.prototxt.txt";
      //let model = "../f-trak/static/model.caffemodel";
      //let min_confidence = 0.9;

      //load model from disk
      //println!("[INFO] loading model...");
      //let net = opencv::dnn::read_net_from_caffe(prototxt, model);

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
            
            let imshow_res = opencv::highgui::imshow(window_name, &frame);
            match imshow_res {
               Ok(_) => {},
               Err(e) => println!("Failed to imshow {:?}", e),
            }
         }

         let key = opencv::highgui::wait_key(1).unwrap();

         if key%256 == 27 {
            println!("Escape hit, closing...");
            break;
         }
      }

      ()
   }
}