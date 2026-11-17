"""Alternative crop geometry for bounded contact recovery."""
import numpy as np
from PIL import Image


class Variant:
    def __init__(self, pose, mode):
        self.pose, self.mode = pose, mode

    def predict_many(self, frames):
        outputs = []
        for image, bounds in frames:
            boxes = np.asarray(bounds,dtype=float)
            if self.mode == 'expanded':
                pad = (boxes[:,2:]-boxes[:,:2])*.15
                boxes[:,:2] = np.maximum(0,boxes[:,:2]-pad)
                boxes[:,2:] = np.minimum(image.size,boxes[:,2:]+pad)
                outputs.append(self.pose.predict_many([(image,boxes.tolist())])[0])
                continue
            with image.transpose(Image.Transpose.ROTATE_90) as rotated:
                transformed = np.stack((boxes[:,1],image.width-boxes[:,2],
                                        boxes[:,3],image.width-boxes[:,0]),axis=1)
                xy,score,prob = self.pose.predict_many([(rotated,transformed.tolist())])[0]
            xy = np.stack((image.width-1-xy[:,:,1],xy[:,:,0]),axis=-1)
            prob = {k:v.copy() for k,v in prob.items()}
            # Alternative peak coordinates belong to the transformed frame.
            if 'alt_strength' in prob:
                prob['alt_strength'][:] = 0
            outputs.append((xy,score,prob))
        return outputs
