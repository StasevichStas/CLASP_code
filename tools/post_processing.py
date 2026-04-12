import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def dense_crf(img, probs, sxy, compat):
    """
    img: исходное изображение (H, W, 3) тип uint8
    probs: вероятности классов от модели (C, H, W) тип float32

    """

    img = np.ascontiguousarray(img)
    probs = np.ascontiguousarray(probs)

    c, h, w = probs.shape

    d = dcrf.DenseCRF2D(w, h, c)
    unary = unary_from_softmax(probs.astype(np.float32))
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=(sxy, sxy), compat=compat)

    d.addPairwiseBilateral(
        sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=compat*3
    )

    q = d.inference(10)

    return np.array(q).reshape((c, h, w))
