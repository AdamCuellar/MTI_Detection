# import jax
import numpy as np
# import jax.numpy as jnp

""" 
Reed-Xiaoli Anomaly Detection Algorithms
Each one of the algorithms below can be found in the following paper:
https://www.umbc.edu/rssipl/pdf/TGRS/tgrs.anomaly/40tgrs06-chang-proof.pdf

Each function expects a numpy array of shape (height, width, channels).
This can be multiple frames from a video to detect movers or a hyperspectral cube.

Any missing algorithms will be added at a later date (utd, arx-utd, etc)
"""

# def arxGPU(input):
#     assert False, "ARX GPU NOT WORKING"
#     h, w, ch = input.shape
#     I = input.reshape((h * w, ch))
#     I = I - jnp.mean(I)
#     covMatrix = jnp.cov(I.T)
#
#     try:
#         inv = jnp.linalg.inv(covMatrix)
#     except np.linalg.LinAlgError:
#         inv = jnp.linalg.pinv(covMatrix)
#
#     M = jnp.matmul(I, inv)
#     rxScore = M.reshape(h, w, 1, ch) @ I.reshape(h, w, ch, 1)
#     rxScore = jnp.squeeze(rxScore)
#     return rxScore

def arx(input, usePinv=False):
    h, w, ch = input.shape
    I = input.reshape((h*w, ch))
    I = I - np.mean(I)
    covMatrix = np.cov(I.T)

    if not usePinv:
        try:
            inv = np.linalg.inv(covMatrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covMatrix)
    else:
        inv = np.linalg.pinv(covMatrix)

    M = np.matmul(I, inv)
    rxScore = M.reshape(h, w, 1, ch) @ I.reshape(h, w, ch, 1)
    rxScore = np.squeeze(rxScore)
    return rxScore

def arxCorr(input, usePinv=False):
    h, w, ch = input.shape
    I = input.reshape((h*w, ch))
    covMatrix = np.corrcoef(I.T)

    if not usePinv:
        try:
            inv = np.linalg.inv(covMatrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covMatrix)
    else:
        inv = np.linalg.pinv(covMatrix)

    M = np.matmul(I, inv)
    rxScore = M.reshape(h, w, 1, ch) @ I.reshape(h, w, ch, 1)
    rxScore = np.squeeze(rxScore)
    return rxScore

def normArx(input, usePinv=False):
    h, w, ch = input.shape
    I = input.reshape((h*w, ch))
    I = I - np.mean(I)
    I = I /  np.linalg.norm(I)
    covMatrix = np.cov(I.T)

    if not usePinv:
        try:
            inv = np.linalg.inv(covMatrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covMatrix)
    else:
        inv = np.linalg.pinv(covMatrix)

    M = np.matmul(I, inv)
    rxScore = M.reshape(h, w, 1, ch) @ I.reshape(h, w, ch, 1)
    rxScore = np.squeeze(rxScore)
    return rxScore

def normArxCorr(input, usePinv=False):
    h, w, ch = input.shape
    I = input.reshape((h*w, ch))
    I = I / np.linalg.norm(I)
    covMatrix = np.corrcoef(I.T)

    if not usePinv:
        try:
            inv = np.linalg.inv(covMatrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covMatrix)
    else:
        inv = np.linalg.pinv(covMatrix)

    M = np.matmul(I, inv)
    rxScore = M.reshape(h, w, 1, ch) @ I.reshape(h, w, ch, 1)
    rxScore = np.squeeze(rxScore)
    return rxScore

def modifiedArx(input, usePinv=False):
    h, w, ch = input.shape
    I = input.reshape((h*w, ch))
    I = I - np.mean(I)
    covMatrix = np.cov(I.T)

    if not usePinv:
        try:
            inv = np.linalg.inv(covMatrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covMatrix)
    else:
        inv = np.linalg.pinv(covMatrix)

    M = np.matmul(I, inv)
    rxScore = M.reshape(h, w, 1, ch) @ (I.reshape(h, w, ch, 1) / np.linalg.norm(I))
    rxScore = np.squeeze(rxScore)
    return rxScore

def modifiedArxCorr(input, usePinv=False):
    h, w, ch = input.shape
    I = input.reshape((h*w, ch))
    covMatrix = np.corrcoef(I.T)

    if not usePinv:
        try:
            inv = np.linalg.inv(covMatrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covMatrix)
    else:
        inv = np.linalg.pinv(covMatrix)

    M = np.matmul(I, inv)
    rxScore = M.reshape(h, w, 1, ch) @ (I.reshape(h, w, ch, 1) / np.linalg.norm(I))
    rxScore = np.squeeze(rxScore)
    return rxScore

if __name__ == "__main__":
    fakeData = np.random.random((640,640,5))
    rx = arx(fakeData)

    # gpu version fails at the moment
    # rxGPU = arxGPU(fakeData)

    # print(" and GPU match: ", np.all(np.isclose(rx, rxGPU)))