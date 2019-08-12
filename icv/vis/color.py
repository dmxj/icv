# -*- coding: UTF-8 -*-
from ..utils.itis import is_seq, is_str

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

# DARK_COLORS = [
#     (255, 48, 48), (148, 0, 211), (160, 82, 45), (139, 0, 0), (71, 71, 71), (34, 139, 34), (3, 3, 3), (0, 0, 238),
#     (238, 0, 238)
# ]

DARK_COLORS = ["darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
               "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred",
               "darksalmon", "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey",
               "darkturquoise", "darkviolet"]

LIGHT_COLORS = ['lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen',
                'lightgray', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
                'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow']

MASK_COLORS = [
    (0, 0, 139),
    (67, 205, 128),
    (0, 206, 209),
    (255, 106, 106),
    (192, 255, 62),
    (130, 130, 130),
    (139, 87, 66),
    (25, 25, 112)
]


def get_color_tuple(color, alpha=None):
    if alpha is not None:
        alpha = min(255, int(alpha)) if alpha > 1 else int(alpha * 255)
    if isinstance(color, (int, float)):
        if color < 1:
            color = int(255 * color)
        color = min(255, int(color))
        if alpha is not None:
            return (color, color, color, alpha)
        return (color, color, color)
    elif is_seq(color):
        color = tuple(color)[:4]
        if alpha is None:
            return color
        if len(color) == 3:
            color = color + (alpha,)
        elif len(color) == 4:
            color = color[:3] + (alpha)
        else:
            raise ValueError("color value invalid")
        return color
    elif is_str(color):
        from PIL import ImageColor
        return ImageColor.getrgb(color)
    else:
        raise ValueError("color value invalid")


def get_reverse_color(color):
    c = get_color_tuple(color)
    reverse_color = []
    for i in c:
        reverse_color.append(255 - i)
    return tuple(reverse_color)

