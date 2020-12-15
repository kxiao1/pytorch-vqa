import torch

colors = [
    'red',
    'blue',
    'yellow',
    'green',
    'black',
    'white',
    'grey',
    'purple',
    'pink',
    'beige',
    'brown',
    'orange',
    'gold',
    'silver',
    'teal',
    'turquoise',
]

colors_rgb = {
        'red': [1,0,0],
        'blue': [0,0,1],
        'yellow': [1,1,0],
        'green': [0,0.5,0],
        'black': [0,0,0],
        'white': [1,1,1],
        'grey': [0.5,0.5,0.5],
        'purple': [0.5,0,0.5],
        'pink': [1,0.753,0.796],
        'beige': [0.961,0.961,0.863],
        'brown': [0.545,0.271,0.075],
        'orange': [1,0.647,0],
        'gold': [0.855,0.647,0.125],
        'silver': [0.75,0.75,0.75],
        'teal': [0,0.5,0.5],
        'turquoise': [0.251,0.878,0.816],
}


colors_set = set(colors)

num_colors = len(colors)

# t is a 3 x m x n tensor or 3 x n
def rgb_to_xyz(t):
    R = t[0]
    G = t[1]
    B = t[2]

    var_R = ((R+0.055)/1.055)**2.4 if R > 0.04045 else R/12.92
    var_G = ((G+0.055)/1.055)**2.4 if G > 0.04045 else G/12.92
    var_B = ((B+0.055)/1.055)**2.4 if B > 0.04045 else B/12.92

    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    return torch.Tensor([X,Y,Z])

def xyz_to_lab(t):
    var_X = t[0]
    var_Y = t[1]
    var_Z = t[2] / 0.60

    var_X = var_X**(1/3) if var_X>0.008856 else ( 7.787 * var_X ) + ( 16 / 116 )
    var_Y = var_Y**(1/3) if var_Y>0.008856 else ( 7.787 * var_Y ) + ( 16 / 116 )
    var_Z = var_Z**(1/3) if var_Z>0.008856 else ( 7.787 * var_Z ) + ( 16 / 116 )

    L = ( 116 * var_Y ) - 16
    a = 500 * ( var_X - var_Y )
    b = 200 * ( var_Y - var_Z )

    return torch.Tensor([L,a,b])/100

colors_lab = {k: xyz_to_lab(rgb_to_xyz(v)) for k, v in colors_rgb.items()}
