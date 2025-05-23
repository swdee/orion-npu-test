# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import math
import numpy as np


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[
        shave_border : height - shave_border, shave_border : width - shave_border
    ]
    gt = gt[shave_border : height - shave_border, shave_border : width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
