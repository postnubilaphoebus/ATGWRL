import torch.autograd.Function as Function


class relu_with_ad(Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp) # save input for backward pass
        return Function.relu(inp)
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None # set output to None
        inp, = ctx.saved_tensors # restore input from context
        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

            

        return grad_input
        pass