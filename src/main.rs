#![recursion_limit = "256"]

mod constants;
mod data;
mod inference;
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
    inference::infer::<MyBackend>(
        artifact_dir,
        device,
        data::EyeDataset::val().get(3).unwrap(),
    );
}
