#start#
import os, glob, sys, time, torch
from torch.optim import lr_scheduler
from utils.loss import get_loss  # use the original get_loss (e.g., returns FocalLoss, DiceLoss, etc.)
from oal import RAdamW, get_scheduler, orientation_anisotropic_loss
torch.set_printoptions(precision=3)

class GradUtil(object):
    def __init__(self, model, loss='ce', lr=0.01, wd=2e-4, root='.'):
        self.path_checkpoint = os.path.join(root, 'super_params.tar')
        if not os.path.exists(root):
            os.makedirs(root)

        self.lossName = loss
        self.criterion = get_loss(loss)
        params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = RAdamW(params=params, lr=lr, weight_decay=wd)
        self.scheduler = get_scheduler(self.optimizer)

    def isLrLowest(self, thresh=1e-5):
        return self.optimizer.param_groups[0]['lr'] < thresh

    coff_ds = 0.5

    def calcGradient(self, criterion, outs, true, fov=None):
        lossSum = 0
        if isinstance(outs, (list, tuple)):
            for i in range(len(outs) - 1, 0, -1):
                loss = criterion(outs[i], true)
                lossSum += loss * self.coff_ds
            outs = outs[0]
        lossSum += criterion(outs, true)
        return lossSum

    def backward_seg(self, pred, true, fov=None, model=None, requires_grad=True, losInit=[], kernels=None):
        self.optimizer.zero_grad()

        costList = []
        los = self.calcGradient(self.criterion, pred, true, fov)
        costList.append(los)
        self.total_loss += los.item()
        del pred, true, los

        if isinstance(losInit, list) and len(losInit) > 0:
            costList.extend(losInit)

        # Integrate Orientation Anisotropic Loss (OAL) if kernels are provided.
        # λ_oal is set to 0.1 as specified in the paper.
        if kernels is not None:
            oal_loss = orientation_anisotropic_loss(kernels)
            costList.append(oal_loss * 0.1)
            print(f"OAL Loss: {oal_loss.item():.4f}")

        losSum = sum(costList)
        losStr = ','.join(['{:.4f}'.format(l.item()) for l in costList])
        if requires_grad:
            losSum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
        return losSum.item(), losStr

    total_loss = 0

    def update_scheduler(self, i=0):
        logStr = '\\r{:03}# '.format(i)
        logStr += '{}={:.4f},'.format(self.lossName, self.total_loss)
        print(logStr, end='')
        self.scheduler.step(self.total_loss)
        self.total_loss = 0
#end#
