"""Raw ROS Image RGB preview without cv_bridge or ROS imports."""
import numpy as np


def rgb_array(message):
    count = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(message.encoding)
    if (count is None or message.step < message.width * count or
            len(message.data) != message.height * message.step):
        raise ValueError('不支持的原始图像编码／布局：' + message.encoding)
    pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    pixels = pixels[:, :message.width * count].reshape(message.height, message.width, count)
    if count == 1:
        return np.repeat(pixels, 3, axis=2)
    pixels = pixels[:, :, :3]
    return pixels[:, :, ::-1].copy() if message.encoding.startswith('bgr') else pixels.copy()
