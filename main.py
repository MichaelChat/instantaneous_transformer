import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

from dataloader import DynamicV4VDataset, V4V_Dataset
from model import Trainer
from utils import *

logger = Logging().get(__name__, args.loglevel)


def main():
    if os.path.exists(summaries_dir):
        logger.warn(f'Overwriting the exp dir {summaries_dir}')
        import time
        time.sleep(1)
    else:
        os.mkdir(summaries_dir)
        os.mkdir(osj(summaries_dir, 'logs'))

    summary_writer = SummaryLogger(summaries_dir)
    model = Trainer()

    # Move model to appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    if args.test:
        train_loader = None
        test_loader = DataLoader(DynamicV4VDataset(split='testing'), batch_size=args.batch_size,
                                 num_workers=6, shuffle=False)
    else:
        train_loader = DataLoader(DynamicV4VDataset(split='training'), batch_size=args.batch_size,
                                  num_workers=6, shuffle=False, pin_memory=True)
        test_loader = None

        val_loader = DataLoader(DynamicV4VDataset(split='validation'), batch_size=args.batch_size,
                                num_workers=6,
                                shuffle=False)

    best_err = 99999
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
            args.start_epoch = checkpoint['epoch']
            # best_acc = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.test and args.best_model:
        if os.path.isfile(args.best_model):
            print("=> loading checkpoint '{}'".format(args.best_model))
            checkpoint = torch.load(args.best_model, map_location=f'cuda:{args.gpu}')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    if args.test:
        best_model = torch.load(args.best_model, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(best_model['state_dict'])
        err = test(test_loader, model, summary_writer, args.start_epoch, 'test')
        summary_dict = {'validation/metrics/err': err}
        summary_writer.log_errors(summary_dict, args.start_epoch)
        exit()

    parameters = model.parameters()
    optimizer = optim.Adadelta(parameters, lr=args.lr)
    scheduler = None

    logger.info(f'Args: {args}')

    # err = test(val_loader, model, summary_writer, args.start_epoch, 'valid')
    # summary_dict = {'validation/metrics/err': err}
    # summary_writer.log_errors(summary_dict, 0)

    for epoch in range(args.start_epoch, args.epochs + 1):
        logger.info(f'Epoch {epoch}')

        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, summary_writer)
        # evaluate on validation set
        if epoch % args.valfreq == 0 and epoch > args.start_epoch:
            err = test(val_loader, model, summary_writer, epoch, 'valid')

            # remember best err and save checkpoint
            is_best = err <= best_err
            best_err = min(err, best_err)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_err,
            }, is_best)

            summary_dict = {'loss/validation/bp_loss': err}
            summary_writer.log_errors(summary_dict, epoch)

    checkpoint = torch.load(osj(weights_dir, 'checkpoint.pth.tar'))  # 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    if args.include_test:
        test_err = test(test_loader, model, summary_writer, checkpoint['epoch'], 'test')
        summary_dict = {'validation/metrics/err': test_err}
        summary_writer.log_errors(summary_dict, checkpoint['epoch'])


def train(train_loader, model: Trainer, optimizer, scheduler, epoch, summary_writer):
    model.train()
    # idx_to_plot = torch.randint(0, len(train_loader), (1,))

    for idx, (btch, _) in tqdm(enumerate(train_loader)):
        dataX, bpsignal, hrsignal, bpmean, bpmax = btch['X'], btch['y_bp'], btch['y_hr'], btch['y_phys_mean'], btch[
            'y_phys_max']  # 'y_phys_mean': y_phys_mean, 'y_phys_max'
        dataX, bpsignal, hrsignal, bpmean, bpmax = [_.cuda().float() for _ in
                                                    [dataX, bpsignal, hrsignal, bpmean, bpmax]]

        summary_dict = {}

        summary_dict['loss/training/bp_loss'], _ = model.bp_loss(dataX, bpsignal, bpmean, bpmax, hrsignal, optimizer,
                                                                 scheduler)
        summary_writer.log_errors(summary_dict, epoch * len(train_loader) + idx)
        # pred = model.feats['tnet']
        #
        # ### Plot ###
        # if idx == idx_to_plot and epoch % 10 == 0:
        #     pred_list = pred.reshape(-1)
        #     bp_y_list = bpsignal.reshape(-1)
        #     plt.plot(pred_list[:200].cpu().detach().numpy(), label='Prediction')
        #     plt.plot(bp_y_list[:200].cpu().detach().numpy(), label='Ground Truth')
        #
        #     # plt.plot(pred.cpu(), 'o', label='Prediction')
        #     # plt.plot(bpmax.cpu(), 'o', label='Ground Truth')
        #     plt.legend()
        #     plt.show()
        # ############


def test(test_loader, model: Trainer, summary_writer, epoch, phase):
    model.eval()
    idx_to_plot = torch.randint(0, len(test_loader), (1,))
    with torch.no_grad():
        preds, bvps, gts = [], [], []
        avg_loss = []

        for i, (btch, _) in tqdm(enumerate(test_loader)):
            dataX, bp_y, hr_y, bpmean, bpmax = btch['X'], btch['y_bp'], btch['y_hr'], btch['y_phys_mean'], btch[
                'y_phys_max']
            dataX, bp_y, hr_y, bpmean, bpmax = [_.cuda().float() for _ in [dataX, bp_y, hr_y, bpmean, bpmax]]

            print(dataX.shape)
            loss, _ = model.bp_loss(dataX, bp_y, bpmean, bpmax, hr_y, None, None, False)
            pred = model.feats['tnet']
            print(pred.shape)
            print(bp_y.shape)
            ### Plot ###
            # if i == idx_to_plot[0]:

            # image_tensor = dataX[0, 0, :, :, 3:]
            #
            # plt.imshow(image_tensor.cpu())
            # plt.show()
            bp = bp_y * bpmax[:, None] + bpmean[:, None]
            pred_list = pred.reshape(-1)
            bp_y_list = bp_y.reshape(-1)

            plt.plot(pred_list[:200].cpu(), label='Prediction')
            plt.plot(bp_y_list[:200].cpu(), label='Ground Truth')

            # plt.plot(pred.cpu(), 'o', label='Prediction')
            # plt.plot(bpmax.cpu(), 'o', label='Ground Truth')
            plt.title(f'Loss: {loss}')
            plt.legend()
            plt.show()
            ############

            preds.append(det_cpu_npy(pred).astype(np.float32))
            gts.extend(det_cpu_npy(hr_y))
            bvps.extend(det_cpu_npy(bp_y))

            avg_loss.append(loss)

        avg = torch.mean(torch.stack(avg_loss))
        # summary_dict = {'loss/validation/bp_loss': avg}
        # summary_writer.log_errors(summary_dict, epoch)

        return avg


if __name__ == '__main__':
    main()
