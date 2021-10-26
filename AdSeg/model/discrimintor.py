import paddle.nn as nn
# from paddle.vision.transforms import functional as F


class FCDiscriminator(nn.Layer):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2D(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2D(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2D(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2D(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2D(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x


if __name__ == "__main__":
   r=FCDiscriminator(num_classes=1000)
   print(r)