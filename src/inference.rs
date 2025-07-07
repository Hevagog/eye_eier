use crate::{
    data::{EyeBatcher, EyeItem},
    training::TrainingConfig,
    utils::draw_predicted_label,
};
use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: EyeItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.clone().label;
    let batcher = EyeBatcher::default();
    let batch = batcher.batch(vec![item.clone()], &device);
    let output = model.forward(batch.images);
    println!("Predicted {} Expected {:?}", output, label);
    let predicted_bbox = output
        .into_data()
        .to_vec()
        .expect("Failed to convert output to Vec")
        .try_into()
        .expect("Output should be a single bounding box");
    let image = draw_predicted_label(&item, predicted_bbox);
    image.unwrap().save("inference_result.png").unwrap();
}
