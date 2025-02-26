import numpy as np
from tslearn.metrics import dtw_path


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def dtw_tdi(pred, true):
    loss_dtw = 0
    loss_tdi = 0
    for i in range(pred.shape[0]):
        path, sim = dtw_path(pred[i], true[i])
        loss_dtw += sim

        Dist = 0
        for i, j in path:
            Dist += (i - j) ** 2
        loss_tdi += Dist
    loss_dtw /= pred.shape[0]
    loss_tdi /= pred.shape[0] * (pred.shape[1] ** 2)
    return loss_dtw, loss_tdi


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def metric_gw(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    dtw, tdi = dtw_tdi(pred, true)

    return mae, mse, rmse, mape, mspe, dtw, tdi


def metric_gca(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)

    return mae, mse, rmse
