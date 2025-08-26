#![recursion_limit = "256"]
mod model;
mod training;
mod data;
mod inference;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig {
            model: ModelConfig::new(10, 512),
            optimizer: AdamConfig::new(),
            num_epochs: 10,
            batch_size: 64,
            num_workers: 4,
            seed: 42,
            learning_rate: 1.0e-4,
        },

        device.clone(),
    );

    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}