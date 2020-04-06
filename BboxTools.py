import numpy as np

try:
    import cv2

    enable_vc2 = True
    resize_method = cv2.INTER_AREA
except:
    enable_vc2 = False

try:
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw

    enable_PIL = True
    if enable_PIL:
        font = ImageFont.truetype('times.ttf', size=14)
    else:
        font = None
except:
    enable_PIL = False


def change_default_font(font_):
    global font
    font = font_


class Bbox2D(object):
    bbox = None
    boundary = None

    def __init__(self, bbox, image_boundary=None):
        self.set_bbox(bbox, check=False)
        if isinstance(image_boundary, np.ndarray):
            image_boundary = image_boundary.tolist()
        if not self.bbox:
            return
        if image_boundary:
            self.set_boundary(list(image_boundary))
        if not self.check_box(bbox=self.bbox, img_shape=self.boundary):
            self.illegal_bbox_exception()
        object.__setattr__(self, 'attributes', {})

    def __len__(self):
        if not self.bbox:
            return 0
        return 2

    def __getitem__(self, item):
        if not self.bbox:
            raise Exception('Cannot get item from empty box!')
        if item == 'x' or item == 0:
            return tuple(self.bbox[0])
        elif item == 'y' or item == 1:
            return tuple(self.bbox[1])
        else:
            if type(item) == int:
                raise Exception(' Input index out of range, which should be 0,1 but get: ' + str(item))
            raise Exception('Illegal index!')

    def __add__(self, other):
        output = self.copy()
        if type(other) == list:
            if len(other) != 2:
                raise Exception('Invaild input!')
            if type(other[0]) == list or type(other[0]) == tuple:
                for i in range(2):
                    for j in range(2):
                        output.one_point(2, 2, other[i][j], change=True)
            else:
                output.shift([0, 1], other)
        if type(other) == Bbox2D:
            for i in range(2):
                for j in range(2):
                    output.one_point(2, 2, other[i][j], change=True)
        return output

    def __eq__(self, other):
        for i in range(2):
            if not self.bbox[i][0] == other.bbox[i][0]:
                return False
            if not self.bbox[i][1] == other.bbox[i][1]:
                return False
        return True

    def __or__(self, other):
        if not self.bbox:
            return other.copy()
        if not other.bbox:
            return self.copy()

        tem = []
        for i in range(2):
            tem.append((min(self.bbox[i][0], other.bbox[i][0]), max(self.bbox[i][1], other.bbox[i][1])))
        equal = False
        if self.boundary and other.boundary:
            equal = True
            for i in range(2):
                if not self.boundary[i] == other.boundary[i]:
                    equal = False
        if equal:
            boundary = self.boundary
        else:
            boundary = None

        return Bbox2D(tem, boundary)

    def __and__(self, other):
        if not self.bbox:
            return self.copy()
        if not other.bbox:
            return other.copy()

        tem = []
        for i in range(2):
            tem.append((max(self.bbox[i][0], other.bbox[i][0]), min(self.bbox[i][1], other.bbox[i][1])))
        equal = False
        if self.boundary and other.boundary:
            equal = True
            for i in range(2):
                if not self.boundary[i] == other.boundary[i]:
                    equal = False
        if equal:
            if not self.check_box(tem, self.boundary):
                return Bbox2D(None)
        else:
            if not self.check_box(tem):
                return Bbox2D(None)

        if equal:
            boundary = self.boundary
        else:
            boundary = None

        return Bbox2D(tem, boundary)

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            if other < 0:
                raise Exception('Input must be positive! Got ' + str(other))
            self_ = self.copy()
            # new_shape = [int(tem * other) for tem in self.shape]
            # self.set_shape(new_shape, center=self.center)
            self_.bbox[0] = (int(self_.bbox[0][0] * other), int(self_.bbox[0][1] * other))
            self_.bbox[1] = (int(self_.bbox[1][0] * other), int(self_.bbox[1][1] * other))

            if self_.boundary:
                self_.boundary = (int(self_.boundary[0] * other), int(self_.boundary[1] * other))
            return self_
        else:
            raise Exception('Multiply method only support int or float input!')

    def __copy__(self):
        return Bbox2D(self.bbox, self.boundary)

    def __str__(self):
        if not self.bbox:
            return '<class "Bbox2D", Empty box>'
        if not self.boundary:
            out = '<class "Bbox2D", shape=[(%d, %d), (%d, %d)]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1])
        else:
            out = '<class "Bbox2D", shape=[(%d, %d), (%d, %d)], boundary=[%d, %d]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1], self.boundary[0], self.boundary[1])
        for k in self.attributes.keys():
            if not k == 'attributes':
                out += ', ' + k + '=' + str(self.attributes[k])
        out += '>'
        return out

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        if self.bbox:
            return True
        else:
            return False

    def __setattr__(self, key, value):
        if key == 'bbox':
            object.__setattr__(self, key, value)
            return
        if key == 'boundary':
            object.__setattr__(self, key, value)
            return
        if type(key) == int:
            raise Exception('In order to avoid misunderstanding, setitem method does not accept int as attribute index.'
                            'Please use one_point method to set the shape of bbox.')
        if not type(key) == str:
            raise Exception('In order to avoid misunderstanding, keys of box attributes can only be str.')
        self.attributes[key] = value

    def __getattr__(self, item):
        if not item in list(self.attributes.keys()) + ['bbox', 'boundary']:
            raise Exception('No attributes %s!' % item)
        if item == 'bbox':
            return self.bbox
        if item == 'boundary':
            return self.boundary
        return self.attributes[item]

    @property
    def shape(self):
        if not self.bbox:
            return 0, 0
        return self._shape_along_axis(0), self._shape_along_axis(1)

    @property
    def size(self):
        if not self.bbox:
            return 0
        return self._shape_along_axis(0) * self._shape_along_axis(1)

    @property
    def center(self):
        if not self.bbox:
            return None
        return self._get_center()

    @property
    def lu(self):
        if not self.bbox:
            return None
        return self.bbox[0][0], self.bbox[1][0]

    @property
    def rb(self):
        if not self.bbox:
            return None
        return self.bbox[0][1], self.bbox[1][1]

    @property
    def ru(self):
        if not self.bbox:
            return None
        return self.bbox[0][1], self.bbox[1][0]

    @property
    def lb(self):
        if not self.bbox:
            return None
        return self.bbox[0][0], self.bbox[1][1]

    def pillow_bbox(self):
        """
        Get bbox in pillow format
        :return: [x0, y0, x1, y1]
        """
        return [self.bbox[1][0], self.bbox[0][0], self.bbox[1][1], self.bbox[0][1]]

    def get_four_corner(self):
        return self.lu, self.rb, self.ru, self.lb

    def assign_attr(self, **kwargs):
        for k, v in kwargs.items():
            self.attributes[str(k)] = v

    def copy(self):
        return Bbox2D(self.bbox, self.boundary)

    def if_include(self, other):
        out = True
        for i in range(2):
            if self.bbox[i][0] > other.bbox[i][0]:
                out = False
            if self.bbox[i][1] < other.bbox[i][1]:
                out = False
        return out

    def exclude(self, other, axis):
        out = self.copy()
        if not self.if_include(other):
            raise Exception('The other box is not inside this box')
        if self.bbox[axis][0] == other.bbox[axis][0]:
            out.one_point(axis, 0, other[axis][1])
        elif self.bbox[axis][1] == other.bbox[axis][1]:
            out.one_point(axis, 1, other[axis][0])
        else:
            raise Exception('Boundary unmatched! Cannot exclude one from the other!')
        if not out.check_box():
            self.illegal_bbox_exception()
        return out

    def inside(self, point):
        if point[0] < self.bbox[0][0] or point[0] >= self.bbox[0][1]:
            return False
        if point[1] < self.bbox[1][0] or point[1] >= self.bbox[1][1]:
            return False
        return True

    def set_bbox(self, bbox, check=True):
        if not bbox:
            self.bbox = None
            return
        if type(bbox) == Bbox2D:
            self.bbox = bbox.bbox
            self.boundary = bbox.boundary
        else:
            self.bbox = [[int(tem[0]), int(tem[1])] for tem in bbox]
        if check:
            if not self.check_box(bbox=self.bbox, img_shape=self.boundary):
                self.illegal_bbox_exception()

    def one_point(self, axis, num, value, change=False, auto_correct=True):
        if axis > 1 or axis < 0 or num > 1 or num < 0:
            raise Exception('Axis or num beyond limit!')
        if change:
            self.bbox[axis][num] += value
        else:
            self.bbox[axis][num] = value
        tem = None
        if auto_correct:
            tem = self._limit_to_boundary(self.boundary)
        if not self.check_box(bbox=self.bbox, img_shape=self.boundary):
            self.illegal_bbox_exception()
        return tem

    def set_boundary(self, boundary):
        if len(boundary) == 3:
            boundary = boundary[0:2]
        self.boundary = boundary
        self._limit_to_boundary(boundary)
        return self

    def _shape_along_axis(self, axis):
        return self.bbox[axis][1] - self.bbox[axis][0]

    def _limit_to_boundary(self, boundary):
        if self.check_box(img_shape=boundary):
            return
        output = None
        for i in range(2):
            if boundary[i] < self.bbox[i][1]:
                output = self.bbox[i][1] - boundary[i]
                self.bbox[i] = [self.bbox[i][0], boundary[i]]
            if self.bbox[i][0] < 0:
                output = -self.bbox[i][0]
                self.bbox[i] = [0, self.bbox[i][1]]
        return output

    def check_box(self, bbox=None, img_shape=None):
        if not bbox:
            bbox = self.bbox
        if not img_shape:
            img_shape = self.boundary
        if not len(bbox) == 2:
            return False
        for tem in bbox:
            if not (type(tem) == list or type(tem) == tuple):
                return False
            if not len(tem) == 2:
                return False
            if not (type(tem[0]) == int and type(tem[1]) == int):
                return False
            if tem[1] < tem[0]:
                return False

        if img_shape:
            for i in range(2):
                if img_shape[i] < bbox[i][1]:
                    return False
                if bbox[i][0] < 0:
                    return False

        return True

    def illegal_bbox_exception(self):
        if self.bbox and self.boundary:
            raise Exception('Illegal boundary box with shape [(%d, %d), (%d, %d)] and boundary [%d, %d]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1], self.boundary[0], self.boundary[1]))
        elif self.bbox:
            raise Exception('Illegal boundary box with shape [(%d, %d), (%d, %d)]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1]))
        else:
            raise Exception('Empty boundary box!')

    def apply(self, image, copy=False):
        if copy:
            return _apply_bbox(image, self.bbox).copy()
        return _apply_bbox(image, self.bbox)

    def assign(self, image, value, auto_fit=True):
        if type(value) == int:
            if len(image.shape) == 2:
                outshape = self.shape
            else:
                outshape = self.shape + image.shape[2::]
            value = np.ones(outshape, dtype=image.dtype) * value
        elif not self.shape == value.shape[0:2]:
            if auto_fit:
                if enable_vc2:
                    value = cv2.resize(value, (self.shape[1], self.shape[0]), interpolation=resize_method)
                else:
                    raise Exception('Unable to import opencv for resize unpaired image to shape of bbox, box shape: '
                                    + str(self.shape) + ' image shape: ' + str(value.shape))
            else:
                raise Exception('Unpaired shape: box shape: ' + str(self.shape) + ' image shape: ' + str(value.shape))
        _assign_bbox(image, value, self.bbox)

    def remove_boundary(self):
        self.boundary = None
        return self

    def putback(self, dest, inbox):
        return _bbox_putback(dest, inbox, self.bbox)

    def pad(self, pad, axis=None, fix_size=False):
        self.bbox = _box_pad(self.bbox, pad, self.boundary, axis=axis, fix_size=fix_size)
        return self

    def shift(self, value, axis=None, force=False):
        if not axis:
            if (type(value) == list or type(value) == tuple) and len(value) == 2:
                axis = [0, 1]
            else:
                raise Exception('Can only use default axis when shift value matched total axes!')
        self.bbox = _box_shift(self.bbox, axis, value, self.boundary, force)
        if not self.bbox:
            return None
        return self

    def transpose(self):
        tem = Bbox2D(bbox=[self.bbox[1], self.bbox[0]], image_boundary=(self.boundary[1], self.boundary[0]))
        for att in self.attributes.keys():
            tem.__setattr__(att, self.__getattr__(att))
        return tem

    def _get_center(self):
        return (self.bbox[0][0] + self.bbox[0][1]) // 2, (self.bbox[1][0] + self.bbox[1][1]) // 2

    def set_shape(self, shape, center=None):
        if not center:
            center = self.center
        new_box = [[int(center[0] - shape[0] // 2), int(center[0] + shape[0] - shape[0] // 2)],
                   [int(center[1] - shape[1] // 2), int(center[1] + shape[1] - shape[1] // 2)]]
        self.bbox = new_box

        if self.boundary:
            self._limit_to_boundary(boundary=self.boundary)

        return self

    def box_in_box(self, boxin):
        output = boxin.copy()
        output = output.shift([-self.bbox[0][0], -self.bbox[1][0]], [0, 1], force=True)
        if not output:
            raise Exception('Out of boundary')
        output.set_boundary(self.shape)
        return output

    def box_out_box(self, boxin):
        output = boxin.copy()
        output = output.shift([0, 1], [self.bbox[0][0], self.bbox[1][0]], force=True)
        if not output:
            raise Exception('Out of boundary')
        output.set_boundary(self.boundary)
        return output

    def numpy(self, save_image_boundary=True, dtype=np.float32):
        return list_box_to_numpy([self], save_image_boundary=save_image_boundary, dtype=dtype)[0]


def box_by_shape(shape, center, image_boundary=None):
    out = Bbox2D([(0, 1), (0, 1)], image_boundary=image_boundary)
    out.set_shape(shape, center)
    return out


def from_numpy(bbox, image_boundary=None, sorts=('y0', 'y1', 'x0', 'x1'), load_boundary_if_possible=True):
    if type(bbox) == list:
        bbox = np.array(bbox)

    if len(bbox.shape) == 2:
        out_ = []
        for i in range(bbox.shape[0]):
            out_.append(from_numpy(bbox[i], image_boundary=image_boundary, sorts=sorts,
                                   load_boundary_if_possible=load_boundary_if_possible))
        return out_
    bbox = bbox.astype(np.int32)
    box_ = [(bbox[sorts.index('y0')], bbox[sorts.index('y1')]), (bbox[sorts.index('x0')], bbox[sorts.index('x1')])]
    if (not image_boundary) and load_boundary_if_possible and bbox.size >= 6:
        image_boundary = [bbox[4], bbox[5]]
    return Bbox2D(box_, image_boundary=image_boundary)


def list_box_to_numpy(box_list, save_image_boundary=False, attributes=tuple(), dtype=np.float32):
    # Current only support scale attributes
    length = len(box_list)
    if save_image_boundary:
        width = 6 + len(attributes)
        output = np.zeros((length, width), dtype=dtype)
        for i in range(length):
            if not box_list[i].boundary:
                raise Exception('Bbox %d have no boundary when save_image_boundary is enabled.' % i)
            output[i, 0:4] = np.array(box_list[i].bbox, dtype=dtype).ravel()
            output[i, 4:6] = np.array(box_list[i].boundary, dtype=dtype)

            for attr, j in zip(attributes, range(6, width)):
                output[i, j] = box_list[i].__getattr__(attr)
    else:
        width = 4 + len(attributes)
        output = np.zeros((length, width), dtype=dtype)
        for i in range(length):
            output[i, 0:4] = np.array(box_list[i].bbox, dtype=dtype).ravel()
            for attr, j in zip(attributes, range(4, width)):
                output[i, j] = box_list[i].__getattr__(attr)
    return output


def pad(box, pad, axis=None, fix_size=False):
    box.pad(pad, axis, fix_size)
    return box


def shift(box, axis, value, force=False):
    box.shift(axis, value, force)
    return box


def nonzero(image):
    non = np.nonzero(image)
    box = [(int(np.min(non[0])), int(np.max(non[0]))), (int(np.min(non[1])), int(np.max(non[1])))]
    return Bbox2D(bbox=box, image_boundary=image.shape)


def contain_points(points, image_boundary=None):
    if type(points) == list:
        points = np.array(points)

    if len(points.shape) == 1:
        points = np.reshape(points, (1, -1))

    return Bbox2D(list(zip(*[points.min(axis=0), points.max(axis=0)])), image_boundary=image_boundary)


def draw_bbox(image, box, boundary=None, fill=None, boundary_width=2, text=None):
    """
    Draw bbox on a image. IMPORTANT: input image will be changed, in order to keep original array unchange,
    please use image.copy()
    :param image: (ndarry) 2D image array, with size (W, H) or (W, H, C)
    :param box: (Bbox2D) box need to draw
    :param boundary: (list or tuple) boundary color, can be (R, G, B) or (R, G, B, A)
    :param fill: (list or tuple) fill color, can be (R, G, B) or (R, G, B, A)
    :param boundary_width: int
    :return: modified image array
    """
    dtype = image.dtype
    if not (boundary or fill):
        raise Exception('Must choose boundary or fill or both! Otherwise will return original image!')

    if len(image.shape) == 2:
        image = np.repeat(image.reshape(image.shape + (1,)), 3, axis=2)

    if fill:
        color = fill
        tem = box.apply(image)
        if len(color) == 4:
            tem[:, :, 0] = tem[:, :, 0] * (1 - color[3])
            tem[:, :, 0] = tem[:, :, 0] + color[3] * color[0]
            tem[:, :, 1] = tem[:, :, 1] * (1 - color[3])
            tem[:, :, 1] = tem[:, :, 1] + color[3] * color[1]
            tem[:, :, 2] = tem[:, :, 2] * (1 - color[3])
            tem[:, :, 2] = tem[:, :, 2] + color[3] * color[2]
        elif len(color) == 3:
            tem[:, :, 0] = color[0]
            tem[:, :, 1] = color[1]
            tem[:, :, 2] = color[2]
        else:
            raise Exception('Filling color must list or tuple with len 3 or 4, but get ' + str(color))

    if boundary:
        mask = np.zeros_like(image, dtype=np.uint8)
        box_outer = box.copy().pad(boundary_width // 2)
        box_inner = box.copy().pad(boundary_width // 2 - boundary_width)
        box_outer.assign(mask, 1)
        box_inner.assign(mask, 0)
        color = boundary
        if len(color) == 4:
            image = image - image * mask * color[3]
            mask[:, :, 0] = mask[:, :, 0] * color[0] * color[3]
            mask[:, :, 1] = mask[:, :, 1] * color[1] * color[3]
            mask[:, :, 2] = mask[:, :, 2] * color[2] * color[3]
            image += mask
        elif len(color) == 3:
            image -= image * mask
            mask[:, :, 0] *= color[0]
            mask[:, :, 1] *= color[1]
            mask[:, :, 2] *= color[2]
            image += mask

    if text:
        if not enable_PIL:
            raise Exception('To add text on bbox, PIL is required!')
        img = Image.fromarray(image)
        draw_one_annotation(img, (box.bbox[0][0], box.bbox[1][0]), text)
        image[:] = np.array(img)

    return image.astype(dtype)


def draw_one_annotation(img, position, cate_s):
    y, x = position
    draw = ImageDraw.Draw(img)
    w, h = font.getsize(cate_s)
    draw.rectangle((x, y, x + w, y + h), fill='white')
    draw.text((x, y), cate_s, fill=(0, 0, 0), font=font)


def _box_shift(bbox, axis, value, boundary=None, force=False):
    if type(axis) == int and type(value) == int:
        axis = [axis]
        value = [value]

    bbox = [list(tem) for tem in bbox]
    for a, x in zip(axis, value):
        bbox[a][0] += x
        bbox[a][1] += x

        if boundary and bbox[a][0] < 0:
            if force:
                bbox[a][0] = 0
            else:
                bbox[a][1] -= bbox[a][0]
                bbox[a][0] -= bbox[a][0]

        if boundary and bbox[a][1] > boundary[a]:
            if force:
                bbox[a][1] = boundary[a]
            else:
                bbox[a][0] -= bbox[a][1] - boundary[a]
                bbox[a][1] -= bbox[a][1] - boundary[a]

    return bbox


def _bbox_putback(whole, inbox, bbox):
    if not (inbox.shape[0] == bbox[0][1] - bbox[0][0] and inbox.shape[1] == bbox[1][1] - bbox[1][0]):
        raise Exception(
            'Unpaired Shape, get box: %s, patch size: (%d, %d)' % (bbox.__str__(), inbox.shape[0], inbox.shape[1]))
    whole[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = inbox
    return whole


def _apply_bbox(source, bbox, mode='numpy'):
    if mode == 'numpy':
        if len(source.shape) == 3:
            get = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
        else:
            get = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
        return get
    if mode == 'torch':
        if len(source.shape) == 4:
            get = source[:, :, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
        elif len(source.shape) == 3:
            get = source[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
        else:
            get = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
        return get
    return


def _assign_bbox(source, value, bbox, mode='numpy'):
    if mode == 'numpy':
        if len(source.shape) == 3:
            source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :] = value
        else:
            source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        return
    if mode == 'torch':
        if len(source.shape) == 4:
            source[:, :, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        elif len(source.shape) == 3:
            source[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        else:
            source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        return


def _box_pad(bbox, padding, boundary=None, axis=None, fix_size=False):
    if not (type(axis) == int or type(axis) == list or type(axis) == tuple):
        axis = np.arange(len(bbox)).tolist()

    if type(axis) == int:
        axis = [axis]

    bbox_out = [list(temp) for temp in bbox]
    if not boundary:
        for tem in axis:
            bbox_out[tem][0] -= padding
            bbox_out[tem][1] += padding

    elif not fix_size:
        for tem in axis:
            bbox_out[tem][0] = max(0, bbox_out[tem][0] - padding)
            bbox_out[tem][1] = min(boundary[tem], bbox_out[tem][1] + padding)

    else:
        for tem in axis:
            if bbox_out[tem][1] - bbox_out[tem][0] + 2 * padding >= boundary[tem]:
                raise Exception('Unable to apply expected size')
            length = bbox_out[tem][1] - bbox_out[tem][0] + 2 * padding
            bbox_out[tem][0] = max(0, bbox_out[tem][0] - padding)
            bbox_out[tem][1] = bbox_out[tem][0] + length
            if bbox_out[tem][1] > boundary[tem]:
                bbox_out[tem][1] = boundary[tem]
                bbox_out[tem][0] = bbox_out[tem][1] - length

    return bbox_out


def projection_function_by_boxes(source_box, target_box, compose=True, max_dim=2):
    # output: lambda x: (x - p0) * ratio + p1
    foos = []
    for axis_ in range(max_dim):
        p0 = source_box.bbox[axis_][0]
        p1 = target_box.bbox[axis_][0]
        ratio = target_box.shape[axis_] / source_box.shape[axis_]

        # hacking -> to avoid return same function
        foos.append(lambda x, p0=p0, ratio=ratio, p1=p1: (x - p0) * ratio + p1)

    if compose:
        return lambda mat_: np.concatenate([foos[t](mat_[:, t:t + 1]) for t in range(max_dim)], axis=1)
    return foos


if __name__ == '__main__':
    # set_size = 14
    # from PIL import Image
    # im = Image.open('vc_pool5_thr_cum_vs_coverage.png')
    # a = np.array(im)
    # box1 = Bbox2D([(10, 100), (50, 500)]).pad(5).shift((20, 20))
    # get = draw_bbox(a, box1, boundary=(255, 0, 0))
    # Image.fromarray(get).show()
    #
    # print(box1.apply(a).shape)
    

    a = Bbox2D([(152, 376), (8, 776)], image_boundary=[535, 813])
    b = Bbox2D([(148, 372), (11, 810)], image_boundary=[535, 813])

    a.score = 5
    print(a)
    print(a.transpose())

    # Image.fromarray(draw_bbox(a, box, boundary=(255, 0, 255, 0.7), fill=(255, 0, 0, 0.3)).astype(np.uint8)).show()

