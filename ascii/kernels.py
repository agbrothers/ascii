import torch


def flatten_kernel(weight, width, height, **kwargs) -> torch.Tensor:
    ## KERNEL EFFECTIVELY FLATTENS THE PATCH INTO A VECTOR
    in_channels = 1
    out_channels = width*height + 1
    filters = torch.zeros((out_channels, in_channels, height, width))
    for i in range(width):
        for j in range(height):
            filters[i+j,0,j,i] = 1.0

    ## FINAL FILTER BIASES HEAVILY BY OVERALL BRIGHTNESS
    filters[-1,0,...] = weight / (width*height)
    return torch.nn.Parameter(filters, requires_grad=False)


def default_kernel(weight, width, height, **kwargs) -> torch.Tensor:
    ## KERNEL USES HAND-CRAFTED FILTERS TO PULL OUT CHARACTER FEATURES
    in_channels = 1
    out_channels = 34
    size = (height, width)
    qH = round(height / 4)
    qW = round(width / 4)
    fH = round(height / 5)
    fW = round(width / 5)
    filters = torch.zeros((out_channels, in_channels, height, width))

    ##  HORIZONTAL BLOCKS
    ##  [=====]   |
    ##  [     ]   |
    ##  [     ]   v
    filters[1,0][0*fH:1*fH] = 1. 
    filters[2,0][1*fH:2*fH] = 1.
    filters[3,0][2*fH:3*fH] = 1.
    filters[4,0][3*fH:4*fH] = 1.
    filters[5,0][4*fH:5*fH] = 1.

    ##  VERTICAL BLOCKS
    ##  [||   ]  
    ##  [||   ]  -->
    ##  [||   ]  
    filters[ 6,0][:, 0*fW:1*fW] = 1. 
    filters[ 7,0][:, 1*fW:2*fW] = 1.
    filters[ 8,0][:, 2*fW:3*fW] = 1.
    filters[ 9,0][:, 3*fW:4*fW] = 1.

    ##  DIAG LINES
    ##  [ \\  ] 
    ##  [  \\ ] 
    ##  [   \\] 
    filters[10,0][
          torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 2*fW)
    ] = 1.0
    filters[11,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 2*fW)
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 1*fW)
    ] = 1.0
    filters[12,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 1*fW)
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 0*fW)
    ] = 1.0
    filters[13,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 0*fW)
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-1*fW)
    ] = 1.0
    filters[14,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-1*fW)
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-2*fW)
    ] = 1.0
    filters[15,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-2*fW)
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-3*fW)
    ] = 1.0
    filters[16,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-3*fW)
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-4*fW)
    ] = 1.0
    filters[17,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-4*fW)
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-5*fW)
    ] = 1.0
    filters[18,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-5*fW)
    ] = 1.0

    ##  DIAG LINES
    ##  [  // ]
    ##  [ //  ]
    ##  [//   ]
    filters[19,0][
          torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 2*fW).flip(dims=(1,))
    ] = 1.0
    filters[20,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 2*fW).flip(dims=(1,))
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 1*fW).flip(dims=(1,))
    ] = 1.0
    filters[21,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 1*fW).flip(dims=(1,))
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 0*fW).flip(dims=(1,))
    ] = 1.0
    filters[22,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal= 0*fW).flip(dims=(1,))
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-1*fW).flip(dims=(1,))
    ] = 1.0
    filters[23,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-1*fW).flip(dims=(1,))
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-2*fW).flip(dims=(1,))
    ] = 1.0
    filters[24,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-2*fW).flip(dims=(1,))
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-3*fW).flip(dims=(1,))
    ] = 1.0
    filters[25,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-3*fW).flip(dims=(1,))
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-4*fW).flip(dims=(1,))
    ] = 1.0
    filters[26,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-4*fW).flip(dims=(1,))
        & torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-5*fW).flip(dims=(1,))
    ] = 1.0
    filters[27,0][
        ~ torch.triu(torch.ones((size), dtype=torch.bool), diagonal=-5*fW).flip(dims=(1,))
    ] = 1.0

    ##  CENTER MASS  
    ##  [     ]  
    ##  [  0  ]  
    ##  [     ]  
    rW1 = 2*qW-1
    rH1 = 2*qH-2
    rW2 = round(1.5*qW)-1
    rH2 = round(1.5*qH)-2
    rW3 = 1*qW-1
    rH3 = 1*qH-2
    filters[28,0][rH1:-rH1, rW1:-rW1] = 1
    filters[29,0][rH2:-rH2, rW2:-rW2] = 1
    filters[30,0][rH3:-rH3, rW3:-rW3] = 1

    ##  CENTER VOID
    ##  [|||||]  
    ##  [|   |]  
    ##  [|||||]  
    filters[31,0] = 1
    filters[32,0] = 1
    filters[33,0] = 1
    filters[31,0][rH1:-rH1, rW1:-rW1] = 0
    filters[32,0][rH2:-rH2, rW2:-rW2] = 0
    filters[33,0][rH3:-rH3, rW3:-rW3] = 0
    
    ## NORMALIZE ALL FILTERS SO THEY INDICATE 
    ## FEATURE PRESENCE RATHER THAN BRIGHTNESS
    magnitude = filters.sum(dim=(-1,-2), keepdim=True)
    eps = torch.Tensor([1e-8])
    filters /= torch.maximum(magnitude, eps)

    ## ADD FILTER TO BIAS BY OVERALL BRIGHTNESS
    filters[0,0,...] = weight / (width*height)
    return torch.nn.Parameter(filters, requires_grad=False)


KERNELS = {
    "flatten": flatten_kernel,
    "default": default_kernel,
}


