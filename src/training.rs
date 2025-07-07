use crate::{
    data::{EyeBatch, EyeBatcher, EyeDataset},
    model::{Model, ModelConfig},
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::{MseLoss, Reduction::Mean},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

impl<B: Backend> Model<B> {
    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let output = self.forward(images);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<EyeBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: EyeBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<EyeBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: EyeBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,

    #[config(default = 40)]
    pub num_epochs: usize,

    #[config(default = 8)]
    pub batch_size: usize,

    #[config(default = 10)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = EyeBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(EyeDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(EyeDataset::test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
