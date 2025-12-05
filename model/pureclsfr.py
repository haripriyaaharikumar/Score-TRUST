import torch
import torch.nn as nn

from my_utils.pytorch1_utils.models import LiteBaseModel
from os.path import join
import torchvision.utils as vutils

class pureclsfr(LiteBaseModel):
    def __init__(self, args, classifier, num_classes,return_features,
                 input_postprocessing_fn=lambda x: x,
                 device='cuda'):
        self.args = args
        ## -----------what is the use of Litebase models-------
        LiteBaseModel.__init__(self, device)
        self.return_features = return_features

        ## this add_module is from the Litebase model
        self.add_module('classifier', classifier)
        assert isinstance(num_classes, int) and num_classes > 0, \
            f"num_classes={num_classes}"
        self.num_classes = num_classes
        assert callable(input_postprocessing_fn), \
            "'input_processing_fn' must be callable!"
        self.input_postprocessing_fn = input_postprocessing_fn

        ##entropy based loss
        self._xent_criterion = nn.CrossEntropyLoss(reduction='none')

    def save(self, file_path, *args, **kwargs):
        save_obj = dict()

        C = self.classifier
        if hasattr(C, 'module') and C.module is not None:
            C = C.module
        save_obj['classifier_state_dict'] = C.cpu().state_dict()
        C.to(self.device)
        torch.save(save_obj, file_path, *args, **kwargs)


    def load(self, file_path, *args, **kwargs):
        print("Load saved classifier!")
        save_obj = torch.load(file_path, *args, **kwargs)

        C = self.classifier
        if hasattr(C, 'module') and C.module is not None:
            C = C.module
        C.load_state_dict(save_obj['classifier_state_dict'])


    def postproc_and_classify(self, x):
        return self.classifier(self.input_postprocessing_fn(x), self.return_features)


    def classify(self, x, y=None):
        y_logit = self.postproc_and_classify(x)
        y_prob = torch.softmax(y_logit, dim=-1)
        y_pred = torch.argmax(y_prob, dim=-1)

        if y is not None:
            xent = self._xent_criterion(y_logit, y).mean(0)
            xent = xent.data.cpu().item()

            acc = y_pred.eq(y).to(torch.float32).mean(0)
            acc = acc.data.cpu().item()
        else:
            xent = None ##don't know, is it x_entropy?
            acc = None

        return {
            'y_prob': y_prob,
            'y_pred': y_pred,
            'acc': acc,
            'xent': xent,
        }


    def get_loss(self, x, y):

        # ----------------- #
        ##nothing is doing on the input data
        print (x.shape)
        y_logit = self.postproc_and_classify(x)

        ##cross entropy
        xent = self._xent_criterion(y_logit, y).mean(0)
        loss = xent

        # ----------------- #
        y_pred = torch.argmax(y_logit, dim=-1)
        matched = y_pred.eq(y)
        matched = matched.to(torch.float32)

        outputs = {
            'loss': loss,
            'xent': xent.data.cpu().item(),
            'acc_clean': matched[:].mean(0).data.cpu().item(),
        }

        return outputs

    def evaluate(self, x, y):
        with torch.no_grad():
            y_logit = self.postproc_and_classify(x)
            acc_clean = torch.argmax(y_logit, dim=-1).eq(y).to(torch.float32).mean(0)

        return {
            'acc_clean': acc_clean,
        }
