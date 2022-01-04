import torch
import logging


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self, label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, tgt, mask, x, mask_keep_prob=0):
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y = out["tag_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in]

        mask_final = t.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device) > mask_keep_prob)

        loss_tag = self.ce(y[mask_final], t[mask_final])

        return loss_tag


class DecisionLoss(torch.nn.Module):

    def __init__(self, label_smoothing=0.0, beta=0.1):
        super(DecisionLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.beta = beta
        logging.info("decision loss weight = " + str(self.beta))

    def forward(self, out, tgt, mask, x, mask_keep_prob=0):
        assert "decision_out" in out
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        # logging.debug("attention mask in " + str(att_mask_in[:4].long()))
        # logging.debug("attention mask in " + str(att_mask_in[:4].long().sum(-1)))
        # logging.debug("attention mask out " + str(att_mask_out[:4].long()))
        # logging.debug("attention mask out " + str(att_mask_out[:4].long().sum(-1)))
        # logging.debug("tgt " + str(tgt[mask][:5]))
        #
        # logging.debug("out = " + str(out["decision_out"][mask][att_mask_out][:20]))
        # logging.debug("tgt = " + str(tgt[mask][att_mask_in].bool().long()[:20]))
        y = out["decision_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in].bool().long()
        mask_final = t.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device) > mask_keep_prob)
        loss_decision = self.ce(
            y[mask_final],
            t[mask_final],
        )
        # logging.debug(mask_final.long())
        error_mask = tgt[mask][att_mask_in].ne(0)
        y = out["tag_out"][mask][att_mask_out][error_mask]
        t = tgt[mask][att_mask_in][error_mask] - 1
        loss_tag = self.ce(
            y,
            t,
        )
        if not error_mask.any():
            loss_tag = torch.zeros_like(loss_tag)
        else:
            # logging.info(out["decision_out"][mask][att_mask_out].shape)
            # logging.info(tgt[mask][att_mask_in].bool().long().shape)
            # logging.info(tgt[mask][att_mask_in].bool().long())
            ...
        return loss_decision + self.beta * loss_tag


class CompensationLoss(torch.nn.Module):

    def __init__(self, label_smoothing=0.0):
        super(CompensationLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, tgt, mask, x, mask_keep_prob=0):
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y = out["tag_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in]
        y0 = y[:, 0]
        y1 = torch.logsumexp(y[:, 1:], -1) - torch.logsumexp(y, -1)
        decision = torch.stack((y0, y1), -1)
        loss_decision = self.ce(decision, tgt[mask][att_mask_in].bool().long())
        mask_final = t.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device) > mask_keep_prob)
        loss_tag = self.ce(y[mask_final], t[mask_final])

        return loss_decision + loss_tag
