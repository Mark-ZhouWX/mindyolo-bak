import mindspore
from mindspore import Tensor, Parameter, nn, ParameterTuple, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class TrainOneStepWithEMA(nn.TrainOneStepWithLossScaleCell):
    """ Train one step with ema model """

    def __init__(self, network, optimizer, sens=1.0, ema=True, decay=0.9998, updates=0):
        super(TrainOneStepWithEMA, self).__init__(network, optimizer, sens)
        self.ema = ema
        self.decay = decay
        self.updates = Parameter(Tensor(updates, mindspore.float32))
        if self.ema:
            self.ema_weight = self.weights.clone("ema", init='same')
            self.moving_parameter = list()
            self.ema_moving_parameter = list()
            self.assign = ops.Assign()
            self.get_moving_parameters()

    def get_moving_parameters(self):
        for key, param in self.network.parameters_and_names():
            if "moving_mean" in key or "moving_variance" in key:
                new_param = param.clone()
                new_param.name = "ema." + param.name
                self.moving_parameter.append(param)
                self.ema_moving_parameter.append(new_param)
        self.moving_parameter = ParameterTuple(self.moving_parameter)
        self.ema_moving_parameter = ParameterTuple(self.ema_moving_parameter)

    def ema_update(self):
        """Update EMA parameters."""
        if self.ema:
            self.updates += 1
            d = self.decay * (1 - ops.Exp()(-self.updates / 2000))
            # update trainable parameters
            for ema_v, weight in zip(self.ema_weight, self.weights):
                tep_v = ema_v * d
                self.assign(ema_v, (1.0 - d) * weight + tep_v)

            for ema_moving, moving in zip(self.ema_moving_parameter, self.moving_parameter):
                tep_m = ema_moving * d
                self.assign(ema_moving, (1.0 - d) * moving + tep_m)
        return self.updates

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        if self.ema:
            self.ema_update()

        # if there is no overflow, do optimize
        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens
