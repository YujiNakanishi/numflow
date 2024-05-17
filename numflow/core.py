import numpy as np
import contextlib
import weakref

"""
# ============================================================== #
Fundamental variable for this module
# ============================================================== #
Attribute:
    data -> <np> data
    grad -> <np> gradient data
    creater -> <Function> function from which this tensor is
    generation -> <int> generation number counted from input of the graph
    name -> <str> tensor name
"""
class tensor:
    __array__priority__ = 200
    def __init__(self, data, name = None):
        self.data = as_array(data)
        self.grad = None
        self.creater = None
        self.generation = 0
        self.name = name
    
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    @property
    def size(self): return self.data.size
    @property
    def dtype(self): return self.data.dtype
    def __len__(self): return len(self.data)
    def __repr__(self):
        if self.data is None: return "tensor(None)"
        return "tensor(" + str(self.data).replace("\n", "\n"+" "*9)+")"
    
    """
    set creater function and generation when this tensor is created from nf.Function
    Input:func -> <nf.Function>
    """
    def set_creater(self, func):
        self.creater = func
        self.generation = func.generation + 1
    
    # clear gradient value
    def cleargrad(self):
        self.grad = None
    
    """
    back propagation
    Input: retain_grad -> <bool> If retain_grad is set, self.grad is deleted after backpropagation is gone. 
    """
    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x : x.generation)
        add_func(self.creater)

        while funcs:
            f = funcs.pop()

            dL_dys = [output().grad for output in f.outputs]
            dL_dxs = f.backward(*dL_dys)
            if not isinstance(dL_dxs, tuple):
                dL_dxs = (dL_dxs, )
            
            for x, dL_dx in zip(f.inputs, dL_dxs):
                x.grad = dL_dx if x.grad is None else x.grad + dL_dx
                if x.creater is not None:
                    add_func(x.creater)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


"""
# ============================================================== #
Absolute Function class and Alias

Attribution:
    inputs -> <list:tensor> input tensors
    outputs -> <list:output> output tensors
# ============================================================== #
"""
class Function:
    """
    Input: inputs -> <tensor...> unpacked tensors
    Output: outputs -> <list:tensor> or <tensor>
    """
    def __call__(self, *inputs):
        inputs = [as_tensor(input) for input in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [tensor(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creater(self)
            
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    """
    Input: xs -> <list:np>
    Output: ys -> <list:np> or <np>
    """
    def forward(self, xs):
        raise NotImplementedError
    
    """
    Input: dL_dys -> <np...> unpacked np.ndarray
    Output: dL_dxs -> <list:np> or <np>
    """
    def backward(self, dL_dys):
        raise NotImplementedError

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    def backward(self, dL_dy):
        return dL_dy, dL_dy
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        return x0*x1
    def backward(self, dL_dy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return dL_dy*x1, dL_dy*x0
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, dL_dy):
        return -dL_dy
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    def backward(self, dL_dy):
        return dL_dy, -dL_dy
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class DIV(Function):
    def forward(self, x0, x1):
        return x0/x1
    def backward(self, dL_dy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        dL_dx0 = dL_dy/x1
        dL_dx1 = -dL_dy*x0/(x1**2)
        return dL_dx0, dL_dx1
def div(x0, x1):
    x1 = as_array(x1)
    return DIV()(x0, x1)
def rdiv(x0, x1):
    x1 = as_array(x1)
    return DIV()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        return x**self.c
    def backward(self, dL_dy):
        x = self.inputs[0].data
        return self.c*x**(self.c-1)*dL_dy
def pow(x, c):
    return Pow(c)(x)


"""
# ============================================================== #
Utility functions
# ============================================================== #
"""
def as_array(x):
    return np.array(x) if np.isscalar(x) else x

def as_tensor(x):
    return x if isinstance(x, tensor) else tensor(x)

"""
# ============================================================== #
Configuration class
Attribution:
    enable_backprop -> <bool> If set, generation, inputs and outputs are saved during calc for backprop
# ============================================================== #
"""
class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_version = getattr(Config, name)
    try:
        yield
    finally:
        setattr(Config, name, old_version)

def no_grad():
    return using_config("enable_backprop", False)


"""
# ============================================================== #
Others
# ============================================================== #
"""
def set_tensor():
    tensor.__add__ = add
    tensor.__radd__ = add
    tensor.__mul__ = mul
    tensor.__rmul__ = mul
    tensor.__neg__ = neg
    tensor.__sub__ = sub
    tensor.__rsub__ = rsub
    tensor.__truediv__ = div
    tensor.__rtruediv__ = rdiv
    tensor.__pow__ = pow