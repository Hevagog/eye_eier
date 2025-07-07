use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    bn3: BatchNorm<B, 2>,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([3, 16], [3, 3]).init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3]).init(device),
            conv3: Conv2dConfig::new([32, 64], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(), // Global average pooling
            activation: Relu::new(),
            linear1: LinearConfig::new(64, 128).init(device),
            linear2: LinearConfig::new(128, 4).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            bn1: BatchNormConfig::new(16).init(device),
            bn2: BatchNormConfig::new(32).init(device),
            bn3: BatchNormConfig::new(64).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, channel, height, width]
    ///   - Output [batch_size, 4] (x_center, y_center, width, height)
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _, _, _] = images.dims();

        let mut x = images;

        x = self.conv1.forward(x);
        x = self.bn1.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);

        x = self.conv2.forward(x);
        x = self.bn2.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);

        x = self.conv3.forward(x);
        x = self.bn3.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);

        x = self.pool.forward(x); // [batch_size, 64, 1, 1]
        let x = x.reshape([batch_size, 64]); // [batch_size, 64]
        let x = self.linear1.forward(x); // [batch_size, 128]
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.linear2.forward(x) // [batch_size, 4]
    }
}
