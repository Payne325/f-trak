extern crate f_trak;
use std::thread;
use std::sync::mpsc;

fn main() {
   let protopath = "D:/Portfolio/f-trak/f-trak/static/deploy.prototxt.txt".to_string();
   let modelpath = "D:/Portfolio/f-trak/f-trak/static/model.caffemodel".to_string();
   let min_confidence = 0.9;

   type Boundingbox = ((i32, i32), (i32, i32));

   let (bbox_transmitter, bbox_receiver) = mpsc::channel::<Boundingbox>();
   let (terminate_transmitter, terminate_receiver) = mpsc::channel::<bool>();

   thread::spawn(move || {
      println!("DEBUG: Spawned the face capture thread!");
      let mut face_cap = f_trak::FaceCapture::new(bbox_transmitter, 
                                                  terminate_transmitter,
                                                  protopath,
                                                  modelpath,
                                                  min_confidence);
      face_cap.begin_capture();
   });

   loop {
      let val = bbox_receiver.try_recv();

      match val {
         Ok(t) => println!("BBOX: (({}, {}), ({}, {}))", t.0.0, t.0.1, t.1.0, t.1.1),
         Err(_) => { /*println!("ERROR: {}", e);*/ },
      }

      let cancel = terminate_receiver.try_recv();

      match cancel {
         Ok(terminate_flag) => if terminate_flag {break;} ,
         Err(_) => { /*println!("ERROR: {}", e);*/ },
      }
   }
   ()
}
