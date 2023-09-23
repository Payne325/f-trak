extern crate f_trak;
use std::sync::mpsc;
use std::thread;

fn main() {
    let protopath = "ADD-YOUR-FILE-PATH-TO/static/deploy.prototxt.txt".to_string();
    let modelpath = "ADD-YOUR-FILE-PATH-TO/static/model.caffemodel".to_string();

    let min_confidence = 0.9;

    type Boundingbox = ((i32, i32), (i32, i32));

    let (bbox_transmitter, bbox_receiver) = mpsc::channel::<Boundingbox>();
    let (terminate_transmitter, terminate_receiver) = mpsc::channel::<bool>();

    thread::spawn(move || {
        println!("DEBUG: Spawned the face capture thread!");
        let mut face_cap = f_trak::FaceCapture::new(
            bbox_transmitter,
            terminate_transmitter,
            protopath,
            modelpath,
            min_confidence,
        );
        face_cap.begin_capture();
    });

    loop {
        let bbox_res = bbox_receiver.try_recv();

        if let Ok(val) = bbox_res {
            println!(
                "BBOX: (({}, {}), ({}, {}))",
                val.0 .0, val.0 .1, val.1 .0, val.1 .1
            );
        }

        let terminate_res = terminate_receiver.try_recv();

        if let Ok(terminate) = terminate_res {
            if terminate {
                break;
            }
        }
    }
}
