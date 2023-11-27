import torch

from globals import *

logger = Logging().get(__name__, args.loglevel)

from network import *


class BloodPressureLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BloodPressureLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions_derivative = post_process_signal_first_derivative(predictions)
        targets_derivative = post_process_signal_first_derivative(targets)
        targets_second_derivative = post_process_signal_first_derivative(targets_derivative)

        squared_error = (predictions - targets) ** 2
        weights = 1 / (torch.abs(targets_second_derivative) + self.smooth)

        weighted_squared_error = weights * squared_error[:, 2:]
        mwse = torch.mean(weighted_squared_error)
        return mwse


class MaximalAbsoluteLoss(nn.Module):
    def __init__(self):
        super(MaximalAbsoluteLoss, self).__init__()

    def forward(self, predictions, targets):
        absolute_loss = torch.abs(predictions - targets)
        maximal_absolute_loss = torch.max(absolute_loss)
        return maximal_absolute_loss


def post_process_signal_first_derivative(sig):
    return sig[:, 1:] - sig[:, :-1]


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self.netcomp = {}

        self.tnet = Tnet()
        self.class_token = None
        self.fps = args.vidfps
        self.batch_size = args.batch_size
        self.criterion0 = nn.MSELoss()
        self.criterion1 = BloodPressureLoss()
        self.criterion2 = nn.MSELoss()
        self.criterion3 = nn.MSELoss()

        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        self.criterion_mal = MaximalAbsoluteLoss()

    def bp_loss(self, x_frames, bp_signal, bpmean, bpmax, hr_signal, optimizer, scheduler, debug=False):
        # assert x_frames.shape[:2] == hr_signal.shape
        # assert bp_signal.shape == hr_signal.shape

        target = bp_signal  # * bpmax[:, None] + bpmean[:, None]
        self.feats = {}
        pred_bp, max_bp, mean_bp, tnet_debug_dict = self.tnet(x_frames, target)
        out = pred_bp
        self.feats['tnet'] = out

        if optimizer is not None:
            optimizer.zero_grad()

        # normalized_bp_derivative = post_process_signal_first_derivative(pred_bp)
        # bp_signal_derivative = post_process_signal_first_derivative(bp_signal)

        ### Loss
        # loss_signal_derivative = self.criterion0(normalized_bp_derivative, bp_signal_derivative)
        # loss_signal_weighted_mse = self.criterion1(pred_bp, bp_signal)
        # loss_signal_mean = self.criterion2(mean_bp[:, 0], bpmean)
        # loss_signal_max = self.criterion3(max_bp[:, 0], bpmax)
        # loss_signal = (
        #     loss_signal_derivative
        #     # + 100 * loss_signal_weighted_mse
        #     # + 0.01 * loss_signal_mean
        #     # + 0.01 * loss_signal_max
        # )

        if debug:
            from matplotlib import pyplot as plt
            plt.plot(pred_bp.reshape(-1)[:200].cpu().detach().numpy(), label='Prediction')
            plt.plot(bp_signal.reshape(-1)[:200].cpu().detach().numpy(), label='Ground Truth')
            plt.legend()

            plt.show()

        loss_mse = self.criterion_mse(pred_bp, bp_signal)
        loss_mae = self.criterion_mae(pred_bp, bp_signal)

        loss = loss_mse

        if optimizer is not None and torch.isnan(out).sum() == 0:
            try:
                loss.backward()
            except:
                bp()

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return loss, tnet_debug_dict

    def forward_eval(self, x):
        assert not self.training
        raise NotImplementedError()

    def forward(self, x, bp_signal):
        pass


'''
Credits: https://github.com/ToyotaResearchInstitute/RemotePPG
'''
tr = torch


class NegativeMaxCrossCov(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegativeMaxCrossCov, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds, labels):
        # Normalize
        preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)

        # Zero-pad signals to prevent circular cross-correlation
        # Also allows for signals of different length
        # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        min_N = min(preds.shape[-1], labels.shape[-1])
        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
        preds_pad = F.pad(preds_norm, (0, padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        # FFT
        # preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        # labels_fft = torch.fft.rfft(labels_pad, dim=-1)
        N = 4 * preds_pad.shape[-1] if PHYS_TYPE == 'HR' else 8 * preds_pad.shape[-1]
        preds_fft = torch.fft.rfft(preds_pad, dim=-1, n=N)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1, n=N)
        freqs = torch.fft.rfftfreq(n=N) * self.Fs

        # Cross-correlation in frequency space
        X = preds_fft * torch.conj(labels_fft)
        X_real = tr.view_as_real(X)

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        # freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs <= self.high_pass / 60, freqs >= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = tr.sum(tr.linalg.norm(X_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = tr.sum(tr.linalg.norm(X_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = tr.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = use_energy[ii] / denom[ii]

        # Zero out irrelevant freqs
        X[:, zero_freqs] = 0.

        # Inverse FFT and normalization
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

        # Max of cross correlation, adjusted for relevant energy
        max_cc = torch.max(cc, dim=-1)[0] / energy_ratio

        return -max_cc


class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, args):
        super(NegativeMaxCrossCorr, self).__init__()
        Fs = args.vidfps
        high_pass = HIGH_HR_FREQ * 60
        low_pass = LOW_HR_FREQ * 60
        self.cross_cov = NegativeMaxCrossCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels):
        denom = torch.std(preds, dim=-1) * torch.std(labels, dim=-1)
        cov = self.cross_cov(preds, labels)
        output = torch.zeros_like(cov)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                output[ii] = cov[ii] / denom[ii]
        # return output
        return output.mean()
