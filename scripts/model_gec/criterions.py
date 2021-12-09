import torch
import logging


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self, label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, tgt, mask):
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        loss_tag = self.ce(out["tag_out"][mask][att_mask_out], tgt[mask][att_mask_in])

        return loss_tag


class DecisionLoss(torch.nn.Module):

    def __init__(self, label_smoothing=0.0):
        super(DecisionLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, tgt, mask, x):
        assert "decision_out" in out
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        loss_decision = self.ce(
            out["decision_out"][mask][att_mask_out],
            tgt[mask][att_mask_in].bool().long()
        )
        error_mask = tgt[mask][att_mask_in].ne(0)
        loss_tag = self.ce(
            out["tag_out"][mask][att_mask_out][error_mask],
            tgt[mask][att_mask_in][error_mask] - 1)
        if not error_mask.any():
            loss_tag = torch.zeros_like(loss_tag)

        return loss_decision + loss_tag


class CompensationLoss(torch.nn.Module):

    def __init__(self, label_smoothing=0.0):
        super(CompensationLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, tgt, mask, x):
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y = out["tag_out"][mask][att_mask_out]
        y0 = y[:, 0]
        y1 = torch.logsumexp(y[:, 1:], -1) - torch.logsumexp(y, -1)
        decision = torch.stack((y0, y1), -1)
        loss_decision = self.ce(decision, tgt[mask][att_mask_in].bool().long())
        loss_tag = self.ce(y, tgt[mask][att_mask_in])

        return loss_decision + loss_tag
