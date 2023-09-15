use opencv::prelude::*;
use std::path::Path;
use std::sync::mpsc;

type FaceBBox = ((i32, i32), (i32, i32));
pub struct FaceCapture {
    m_bbox: FaceBBox,
    m_bbox_transmitter: mpsc::Sender<FaceBBox>,
    m_terminate_transmitter: mpsc::Sender<bool>,
    m_protopath_str: String,
    m_modelpath_str: String,
    m_min_confidence: f32,
}

impl FaceCapture {
    pub fn new(
        bbox_transmitter: mpsc::Sender<FaceBBox>,
        termination_transmitter: mpsc::Sender<bool>,
        protopath_str: String,
        modelpath_str: String,
        min_confidence: f32,
    ) -> Self {
        let bbox: FaceBBox = ((0, 0), (0, 0));
        Self {
            m_bbox: bbox,
            m_bbox_transmitter: bbox_transmitter,
            m_terminate_transmitter: termination_transmitter,
            m_protopath_str: protopath_str,
            m_modelpath_str: modelpath_str,
            m_min_confidence: min_confidence,
        }
    }

    pub fn begin_capture(&mut self) {
        let protopath = Path::new(&self.m_protopath_str);
        let modelpath = Path::new(&self.m_modelpath_str);

        let mut file_exists = protopath.exists();

        match file_exists {
            true => {}
            false => println!("Failed to find proto file"),
        }

        file_exists = modelpath.exists();

        match file_exists {
            true => {}
            false => println!("Failed to find model file"),
        }

        let prototxt = protopath.to_str().unwrap();
        let model = modelpath.to_str().unwrap();

        let net_res = opencv::dnn::read_net_from_caffe(prototxt, model);
        let mut net = opencv::dnn::Net::default().unwrap();

        match net_res {
            Ok(v) => net = v,
            Err(e) => println!("Failed to construct neural network: {:?}", e),
        }

        //Todo: allow user to select camera
        let mut camera = opencv::videoio::VideoCapture::new(
            0, 0, //opencv::videoio::VideoCaptureAPIs::CAP_ANY
        )
        .unwrap();

        let window_name = "f-trak";

        let win_created = opencv::highgui::named_window(
            window_name,
            1, //opencv::highgui::WindowFlags::WINDOW_AUTOSIZE
        );

        match win_created {
            Ok(v) => println!("DEBUG: Successfully set up camera!{:?}", v),
            Err(e) => println!("Failed to create window: {:?}", e),
        }

        loop {
            let mut frame = Mat::default();
            let success = camera.read(&mut frame);

            if success.unwrap() {
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

                let set_input_result =
                    net.set_input(&blob, "", scale_factor, opencv::core::Scalar::default());

                match set_input_result {
                    Ok(_) => {}
                    Err(e) => println!("Failed to set neural network input data {:?}", e),
                }

                let mut nn_output = opencv::types::VectorOfMat::new();
                let mut blob_names = opencv::types::VectorOfString::new();
                blob_names.push(&String::from("detection_out"));

                let forward_success = net.forward(&mut nn_output, &blob_names);

                match forward_success {
                    Ok(_) => {}
                    Err(e) => println!("Failed to forward neural network {:?}", e),
                }

                let detection_res = nn_output.get(0);

                match detection_res {
                    Ok(_) => {}
                    Err(ref e) => println!(
                        "Failed to extract detections from neural network output data: {:?}",
                        e
                    ),
                }

                let detections = detection_res.unwrap();
                let num_detections = detections.mat_size()[2];

                for i in 0..num_detections {
                    let confidence_index: [i32; 4] = [0, 0, i, 2];
                    let confidence = detections.at_nd::<f32>(&confidence_index);

                    if *confidence.unwrap() < self.m_min_confidence {
                        continue;
                    }

                    let start_x_index: [i32; 4] = [0, 0, i, 3];
                    let start_y_index: [i32; 4] = [0, 0, i, 4];
                    let end_x_index: [i32; 4] = [0, 0, i, 5];
                    let end_y_index: [i32; 4] = [0, 0, i, 6];

                    let raw_start_x = detections.at_nd::<f32>(&start_x_index).unwrap();
                    let raw_start_y = detections.at_nd::<f32>(&start_y_index).unwrap();
                    let raw_end_x = detections.at_nd::<f32>(&end_x_index).unwrap();
                    let raw_end_y = detections.at_nd::<f32>(&end_y_index).unwrap();

                    let start_x = raw_start_x * frame_size.width as f32;
                    let start_y = raw_start_y * frame_size.height as f32;
                    let end_x = raw_end_x * frame_size.width as f32;
                    let end_y = raw_end_y * frame_size.height as f32;

                    let start_y = if start_y - 10.0 > 10.0 {
                        start_y - 10.0
                    } else {
                        start_y + 10.0
                    };

                    let start_pt = opencv::core::Point2i::new(start_x as i32, start_y as i32);
                    let end_pt = opencv::core::Point2i::new(end_x as i32, end_y as i32);
                    self.m_bbox = ((start_pt.x, start_pt.y), (end_pt.x, end_pt.y));
                    self.m_bbox_transmitter.send(self.m_bbox).unwrap();

                    let rect = opencv::core::Rect::from_points(start_pt, end_pt);

                    let colour_blue = opencv::core::Scalar::new(255.0, 0.0, 0.0, 1.0); //BGR colour
                    let draw_rect_result = opencv::imgproc::rectangle(
                        &mut frame,
                        rect,
                        colour_blue,
                        2, //line thickness
                        1, //opencv::imgproc::LineTypes::LINE_4,
                        0,
                    ); //shitf

                    match draw_rect_result {
                        Ok(_) => {}
                        Err(e) => println!("Failed to draw rectangle: {:?}", e),
                    }
                }

                let imshow_res = opencv::highgui::imshow(window_name, &frame);
                match imshow_res {
                    Ok(_) => {}
                    Err(e) => println!("Failed to show frame: {:?}", e),
                }
            }

            let key = opencv::highgui::wait_key(1).unwrap();

            if key % 256 == 27 {
                println!("DEBUG: Escape hit, closing...");
                self.m_terminate_transmitter.send(true).unwrap();
                match camera.release() {
                    Ok(_) => {}
                    Err(e) => println!("Failed to release video device: {:?}", e),
                }
                break;
            } else {
                self.m_terminate_transmitter.send(false).unwrap();
            }
        }
    }
}
