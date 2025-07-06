#![recursion_limit = "256"]

mod constants;
mod data;
mod model;
mod training;
mod utils;
use crate::{model::ModelConfig, training::TrainingConfig};

use burn::{
    backend::{Autodiff, Cuda},
    data::dataset::Dataset,
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::cuda::CudaDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );
    // let im =
    //     utils::get_image_with_bounding_box("i_6_jpg.rf.656e7db6074c5e9a96d62604fdc7e0cf", false);
    // im.unwrap().save("test.png").unwrap();
    // let items = data::get_items(false);
    // println!("Number of items: {}", items.len());
    // for item in items.iter().take(2) {
    //     utils::draw_label(item).map(|img| img.save(format!("{}.png", item.label[0])).unwrap());
    // }
}
