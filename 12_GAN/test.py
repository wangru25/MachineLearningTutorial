from PIL import Image
import numpy as np

slice56 = np.random.random((226, 226))
print(slice56.shape)

# convert values to 0 - 255 int8 format
formatted = (slice56 * 255 / np.max(slice56)).astype('uint8')
print(formatted.shape)
img = Image.fromarray(formatted)
img.show()
