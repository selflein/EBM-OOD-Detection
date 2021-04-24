import torch
import torch.nn as nn


class JEM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.f = model

    def forward(self, x, return_logits=False, y=None):
        logits = self.classify(x)

        if y is not None:
            return logits[torch.arange(len(x)), y]

        if return_logits:
            return logits.logsumexp(1), logits
        else:
            return logits.logsumexp(1)

    def classify(self, x):
        return self.f(x)


class HDGE(JEM):
    def __init__(self, model, n_classes, contrast_k, contrast_t):
        super(HDGE, self).__init__(model)

        self.K = contrast_k
        self.T = contrast_t
        self.dim = n_classes

        # create the queue
        init_logit = torch.randn(n_classes, contrast_k)
        self.register_buffer("queue_logit", init_logit)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, logits):
        # gather logits before updating queue
        batch_size = logits.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the logits at ptr (dequeue and enqueue)
        self.queue_logit[:, ptr : ptr + batch_size] = logits.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def joint(self, img, dist=None, evaluation=False):
        f_logit = self.class_output(self.f(img))  # queries: NxC
        ce_logit = f_logit  # cross-entropy loss logits
        prob = nn.functional.normalize(f_logit, dim=1)
        # positive logits: Nx1
        l_pos = dist * prob  # NxC
        l_pos = torch.logsumexp(l_pos, dim=1, keepdim=True)  # Nx1
        # negative logits: NxK
        buffer = nn.functional.normalize(self.queue_logit.clone().detach(), dim=0)
        l_neg = torch.einsum("nc,ck->nck", [dist, buffer])  # NxCxK
        l_neg = torch.logsumexp(l_neg, dim=1)  # NxK

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if not evaluation:
            self._dequeue_and_enqueue(f_logit)

        return logits, labels, ce_logit, l_neg.size(1)
