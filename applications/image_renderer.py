from itertools import cycle

import PIL.Image
import PIL.ImageDraw


RAW_SIZE = 256, 256
COLORS = ['#0095EF', '#3C50B1', '#6A38B3', '#A224AD', '#F31D64', '#FE433C']


class ImageRenderer:
    """Converts string with strokes into PIL image."""

    def __init__(self, mode='b/w', bg='black', fg='white', lw: int=4,
                 colors=None):

        mode = mode if mode in ('b/w', 'rgb') else 'b/w'

        self.render_fn = {
            'b/w': render_bw,
            'rgb': render_rgb
        }[mode]

        self.mode = mode
        self.bg = bg
        self.fg = fg
        self.lw = lw
        self.colors = cycle(colors or COLORS)

    def render(self, strokes: str, image_size: tuple):
        x_ref, y_ref = RAW_SIZE
        x_max, y_max = image_size
        ratio = x_max/float(x_ref), y_max/float(y_ref)
        return self.render_fn(self, strokes, ratio, image_size)


def render_bw(renderer, strokes, ratio, image_size):
    bg, fg, lw = [getattr(renderer, x) for x in 'bg fg lw'.split()]

    x_ratio, y_ratio = ratio
    canvas = PIL.Image.new('RGB', image_size, color=bg)
    draw = PIL.ImageDraw.Draw(canvas)

    for segment in strokes.split('|'):
        chunks = [int(x) for x in segment.split(',')]
        while len(chunks) >= 4:
            (x1, y1, x2, y2), chunks = chunks[:4], chunks[2:]
            scaled = (
                int(x1 * x_ratio), int(y1 * y_ratio),
                int(x2 * x_ratio), int(y2 * y_ratio))
            draw.line(tuple(scaled), fill=fg, width=lw)

    return canvas


def render_rgb(renderer, strokes, ratio, image_size):
    colors, bg, lw = [getattr(renderer, x) for x in 'colors bg lw'.split()]

    x_ratio, y_ratio = ratio
    canvas = PIL.Image.new('RGB', image_size, color=bg)
    draw = PIL.ImageDraw.Draw(canvas)

    for segment, color in zip(strokes.split('|'), colors):
        chunks = [int(x) for x in segment.split(',')]
        while len(chunks) >= 4:
            (x1, y1, x2, y2), chunks = chunks[:4], chunks[2:]
            scaled = (
                int(x1 * x_ratio), int(y1 * y_ratio),
                int(x2 * x_ratio), int(y2 * y_ratio))
            draw.line(tuple(scaled), fill=color, width=lw)

    return canvas


default_renderer = ImageRenderer('rgb', bg='white')
