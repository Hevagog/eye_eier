use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use imageproc::{drawing::draw_hollow_rect, rect::Rect};

use std::fs;

use crate::constants::{HEIGHT, WIDTH};
use crate::data::EyeItem;

pub fn read_rect_from_label_path(label_path: &str) -> Option<(f32, f32, f32, f32)> {
    let contents = fs::read_to_string(label_path).ok()?;
    let values: Vec<&str> = contents.trim().split_whitespace().collect();

    if values.len() == 5 {
        let x_center = values[1].parse::<f32>().ok()?;
        let y_center = values[2].parse::<f32>().ok()?;
        let width = values[3].parse::<f32>().ok()?;
        let height = values[4].parse::<f32>().ok()?;
        Some((x_center, y_center, width, height))
    } else {
        None
    }
}

pub fn read_image_from_path(image_path: &str) -> Option<image::DynamicImage> {
    Some(image::open(image_path).unwrap())
}

pub fn get_image_with_bounding_box(
    name: &str,
    is_train: bool,
) -> Option<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let type_path = if is_train {
        "Dataset/train"
    } else {
        "Dataset/test"
    };

    let label_path = format!("{}/labels/{}.txt", type_path, name);
    let image_path = format!("{}/images/{}.jpg", type_path, name);

    if let Some((mut x_center, mut y_center, mut width, mut height)) =
        read_rect_from_label_path(&label_path)
    {
        if let Some(img) = read_image_from_path(&image_path) {
            let x_scale = img.width() as f32;
            let y_scale = img.height() as f32;
            x_center = x_center * x_scale;
            y_center = y_center * y_scale;
            width = width * x_scale;
            height = height * y_scale;
            let x = (x_center - width / 2.0) as u32;
            let y = (y_center - height / 2.0) as u32;
            let rect = Rect::at(x as i32, y as i32).of_size(width as u32, height as u32);
            let img2 = draw_hollow_rect(&img, rect, image::Rgba([255, 0, 0, 255]));
            return Some(img2);
        }
    }
    None
}

pub fn eye_item_to_image(item: &EyeItem) -> Option<image::DynamicImage> {
    // Reconstruct image from flat RGB Vec<u8>
    let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
            let xi = x as usize;
            let yi = y as usize;
            let idx = (yi * WIDTH + xi) * 3;
            let r = item.image[idx];
            let g = item.image[idx + 1];
            let b = item.image[idx + 2];
            Rgb([r, g, b])
        });
    Some(DynamicImage::ImageRgb8(buf))
}

pub fn draw_label(item: &EyeItem) -> Option<DynamicImage> {
    let dyn_img = eye_item_to_image(item)?;
    let mut img = dyn_img.to_rgba8();

    // YOLO label: center x,y and width,height in normalized [0,1]
    let [xc, yc, w, h] = item.label;
    let img_w = img.width() as f32;
    let img_h = img.height() as f32;
    let x = ((xc - w / 2.0) * img_w).round() as i32;
    let y = ((yc - h / 2.0) * img_h).round() as i32;
    let rw: u32 = (w * img_w).round() as u32;
    let rh = (h * img_h).round() as u32;

    let rect = Rect::at(x, y).of_size(rw, rh);
    let im2: ImageBuffer<Rgba<u8>, Vec<u8>> =
        draw_hollow_rect(&mut img, rect, Rgba([255, 0, 0, 255]));

    Some(DynamicImage::ImageRgba8(im2))
}
