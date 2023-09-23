use opencv::prelude::*;
use std::sync::mpsc;

type FaceBBox = ((i32, i32), (i32, i32));
const WINDOW_NAME: &str = "f-trak";

pub struct FaceCapture {
    bbox: FaceBBox,
    bbox_transmitter: mpsc::Sender<FaceBBox>,
    termination_transmitter: mpsc::Sender<bool>,
    protopath: String,
    modelpath: String,
    min_confidence: f32,
}

impl FaceCapture {
    pub fn new(
        bbox_transmitter: mpsc::Sender<FaceBBox>,
        termination_transmitter: mpsc::Sender<bool>,
        protopath: String,
        modelpath: String,
        min_confidence: f32,
    ) -> Self {
        let bbox: FaceBBox = ((0, 0), (0, 0));
        Self {
            bbox,
            bbox_transmitter,
            termination_transmitter,
            protopath,
            modelpath,
            min_confidence,
        }
    }

    pub fn begin_capture(&mut self) {
        let net_res =
            opencv::dnn::read_net_from_caffe(self.protopath.as_str(), self.modelpath.as_str());

        let mut net = match net_res {
            Ok(net) => net,
            Err(e) => panic!("Failed to construct neural network: {:?}", e),
        };

        //Todo: allow user to select camera
        let mut camera = opencv::videoio::VideoCapture::new(
            0, 0, //opencv::videoio::VideoCaptureAPIs::CAP_ANY
        )
        .expect("Unable to select camera 0.");

        //flags = opencv::highgui::WindowFlags::WINDOW_AUTOSIZE
        match opencv::highgui::named_window(WINDOW_NAME, 1) {
            Ok(v) => println!("DEBUG: Successfully set up camera!{:?}", v),
            Err(e) => println!("Failed to create window: {:?}", e),
        }

        loop {
            let mut frame = Mat::default();
            let success = camera
                .read(&mut frame)
                .expect("Failed to read frame from camera.");

            if success {
                let scale_factor = 1.0;
                let frame_size = frame.size().unwrap(); //Needed later to draw debug boxes
                let mut resized_frame = Mat::default();
                let nn_processing_size = opencv::core::Size::new(300, 300);
                let resize_result = opencv::imgproc::resize(
                    &frame,
                    &mut resized_frame,
                    nn_processing_size,
                    scale_factor,
                    scale_factor,
                    1, //opencv::imgproc::InterpolationFlags::INTER_LINEAR
                );

                match resize_result {
                    Ok(_) => {}
                    Err(e) => println!("Failed to resize camera captured image {:?}", e),
                }

                let mean = opencv::core::Scalar::new(104.0, 177.0, 123.0, 1.0);
                let blob = opencv::dnn::blob_from_image(
                    &resized_frame,
                    scale_factor,
                    nn_processing_size,
                    mean,
                    false,
                    false,
                    opencv::core::CV_32F,
                )
                .unwrap();

                if let Err(e) =
                    net.set_input(&blob, "", scale_factor, opencv::core::Scalar::default())
                {
                    println!("Failed to set neural network input data {:?}", e);
                }

                let mut nn_output = opencv::types::VectorOfMat::new();
                let mut blob_names = opencv::types::VectorOfString::new();
                blob_names.push(&String::from("detection_out"));

                if let Err(e) = net.forward(&mut nn_output, &blob_names) {
                    println!("Failed to forward neural network {:?}", e);
                }

                let detections = match nn_output.get(0) {
                    Ok(d) => d,
                    Err(e) => panic!(
                        "Failed to extract detections from neural network output data: {:?}",
                        e
                    ),
                };

                let num_detections = detections.mat_size()[2];

                for detection_index in 0..num_detections {
                    let confidence_index: [i32; 4] = [0, 0, detection_index, 2];
                    let confidence = detections
                        .at_nd::<f32>(&confidence_index)
                        .expect("Should have some confidence value.");

                    if *confidence < self.min_confidence {
                        continue;
                    }

                    let start_x_index: [i32; 4] = [0, 0, detection_index, 3];
                    let start_y_index: [i32; 4] = [0, 0, detection_index, 4];
                    let end_x_index: [i32; 4] = [0, 0, detection_index, 5];
                    let end_y_index: [i32; 4] = [0, 0, detection_index, 6];

                    // values are held as f32s, trying to get as any other more useful type will cause a panic.
                    let raw_start_x = detections.at_nd::<f32>(&start_x_index).unwrap();
                    let raw_start_y = detections.at_nd::<f32>(&start_y_index).unwrap();
                    let raw_end_x = detections.at_nd::<f32>(&end_x_index).unwrap();
                    let raw_end_y = detections.at_nd::<f32>(&end_y_index).unwrap();

                    let start_x = raw_start_x * frame_size.width as f32;
                    let start_y = raw_start_y * frame_size.height as f32;
                    let end_x = raw_end_x * frame_size.width as f32;
                    let end_y = raw_end_y * frame_size.height as f32;

                    let start_pt = opencv::core::Point2i::new(start_x as i32, start_y as i32);
                    let end_pt = opencv::core::Point2i::new(end_x as i32, end_y as i32);
                    self.bbox = ((start_pt.x, start_pt.y), (end_pt.x, end_pt.y));
                    self.bbox_transmitter.send(self.bbox).unwrap();

                    let rect = opencv::core::Rect::from_points(start_pt, end_pt);

                    let colour_blue = opencv::core::Scalar::new(255.0, 0.0, 0.0, 1.0); //BGR colour
                    if let Err(e) = opencv::imgproc::rectangle(
                        &mut frame,
                        rect,
                        colour_blue,
                        2, //line thickness
                        1, //opencv::imgproc::LineTypes::LINE_4,
                        0,
                    ) {
                        println!("Failed to draw rectangle: {:?}", e);
                    }
                }

                if let Err(e) = opencv::highgui::imshow(WINDOW_NAME, &frame) {
                    println!("Failed to show frame: {:?}", e);
                }
            }

            let key = opencv::highgui::wait_key(1).unwrap();

            if key % 256 == 27 {
                println!("DEBUG: Escape hit, closing...");
                self.termination_transmitter.send(true).unwrap();

                if let Err(e) = camera.release() {
                    println!("Failed to release video device: {:?}", e);
                }

                break;
            } else {
                self.termination_transmitter.send(false).unwrap();
            }
        }
    }
}
