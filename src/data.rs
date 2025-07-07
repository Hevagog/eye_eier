use crate::constants::{HEIGHT, WIDTH};
use crate::utils;
use burn::data::dataset::Dataset;
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use image::imageops::FilterType;
use image::{DynamicImage, Rgb32FImage};

pub struct EyeDataset {
    items: Vec<EyeItem>,
}

impl EyeDataset {
    pub fn train() -> Self {
        let items = get_items("train");
        Self { items }
    }

    pub fn test() -> Self {
        let items = get_items("test");
        Self { items }
    }
    pub fn val() -> Self {
        let items = get_items("val");
        Self { items }
    }
}

impl Dataset<EyeItem> for EyeDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<EyeItem> {
        self.items.get(index).cloned()
    }
}

#[derive(Clone, Default)]
pub struct EyeBatcher {}

#[derive(Clone, Debug)]
pub struct EyeBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2>,
}

#[derive(Debug, Clone)]
pub struct EyeItem {
    pub image: Vec<u8>,
    // x_center, y_center, width, height
    pub label: [f32; 4],
}

impl<B: Backend> Batcher<B, EyeItem, EyeBatch<B>> for EyeBatcher {
    fn batch(&self, items: Vec<EyeItem>, device: &B::Device) -> EyeBatch<B> {
        let mut image_tensors = Vec::with_capacity(items.len());
        let mut target_tensors = Vec::with_capacity(items.len());
        for item in items {
            // Normalize pixel data on CPU and build TensorData of shape [3, H, W]
            let data_f32: Vec<f32> = item
                .image
                .chunks(3)
                .map(|px| {
                    let r = px[0] as f32 / 255.0;
                    let g = px[1] as f32 / 255.0;
                    let b = px[2] as f32 / 255.0;
                    // Normalize channels
                    [
                        (r - 0.1307) / 0.3081,
                        (g - 0.1307) / 0.3081,
                        (b - 0.1307) / 0.3081,
                    ]
                })
                .flatten()
                .collect();
            let tensor_img = Tensor::<B, 3>::from_data(
                TensorData::new(data_f32, [3, HEIGHT, WIDTH]).convert::<B::FloatElem>(),
                device,
            )
            .reshape([1, 3, HEIGHT, WIDTH]);
            image_tensors.push(tensor_img);

            let label_f32: Vec<f32> = item.label.iter().cloned().collect();
            let tensor_label = Tensor::<B, 2>::from_data(
                TensorData::new(label_f32, [1, 4]).convert::<B::FloatElem>(),
                device,
            );
            target_tensors.push(tensor_label);
        }
        let images = Tensor::cat(image_tensors, 0);
        let targets = Tensor::cat(target_tensors, 0);
        EyeBatch { images, targets }
    }
}

pub fn get_items(data_type: &str) -> Vec<EyeItem> {
    let label_path = format!("Dataset/{}/labels", data_type);
    let image_path = format!("Dataset/{}/images", data_type);

    let mut items = Vec::new();

    for entry in std::fs::read_dir(&label_path).unwrap() {
        let entry = entry.unwrap();
        let label_file_name = entry.file_name().to_string_lossy().to_string();
        if label_file_name.ends_with(".txt") {
            let image_file_name = label_file_name.replace(".txt", ".jpg");
            let full_label_path = format!("{}/{}", &label_path, label_file_name);
            let full_image_path = format!("{}/{}", &image_path, image_file_name);

            if let Some((x_center, y_center, width, height)) =
                utils::read_rect_from_label_path(&full_label_path)
            {
                if let Some(img) = utils::read_image_from_path(&full_image_path) {
                    let item = dyn_to_eyeitem(&img, [x_center, y_center, width, height]);
                    items.push(item);
                }
            }
        }
    }
    items
}

fn dyn_to_eyeitem(dyn_img: &DynamicImage, label: [f32; 4]) -> EyeItem {
    let im = dyn_img.resize_to_fill(WIDTH as u32, HEIGHT as u32, FilterType::Triangle);
    let rgb8 = im.to_rgb8();
    let (w, h) = (rgb8.width() as usize, rgb8.height() as usize);

    assert_eq!(w, WIDTH);
    assert_eq!(h, HEIGHT);

    let image = rgb8.into_raw();
    EyeItem { image, label }
}
