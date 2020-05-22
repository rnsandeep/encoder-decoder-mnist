#from keras_anchor_bbox import AnchorBoxes
import numpy as np
from bounding_box_utils import convert_coordinates
import cv2, sys

class Anchors:
    
    def __init__(self, img_height, img_width, feature_map_size, aspect_ratios, this_scale, this_steps, this_offsets):
        self.img_height = img_height
        self.img_width = img_width
        self.feature_map_size = feature_map_size
        self.aspect_ratios = aspect_ratios
        self.this_scale = this_scale
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = True

        self.anchors = self.generate_anchor_boxes_for_layer(feature_map_size, aspect_ratios, this_scale, this_steps, this_offsets)


    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):
        '''
        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor layer
        of size `feature_map_size == [feature_map_height, feature_map_width]`.
        Arguments:
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            diagnostics (bool, optional): If true, the following additional outputs will be returned:
                1) A list of the center point `x` and `y` coordinates for each spatial location.
                2) A list containing `(width, height)` for each box aspect ratio.
                3) A tuple containing `(step_height, step_width)`
                4) A tuple containing `(offset_height, offset_width)`
                This information can be useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their spatial distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.
        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if False: #self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        #print(feature_map_size[0])
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords


        print(boxes_tensor)

        return boxes_tensor

def drawBoxes(I, boxes):

    for box in boxes:
        color = (255, 0, 0)
        thickness = 1
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        start_point = (box[0], box[1]) # (box[0]-int(box[2]/2), box[1]-int(box[3]/2))
        end_point = (box[2], box[3]) #(box[0]+box[2], box[1]+box[3])
        I = cv2.rectangle(I, start_point, end_point, color, thickness)
    return I
#    cv2.imwrite(os.path.join(output_path, os.path.basename(image)), image)        

a = Anchors(300, 300, [18, 18], [1.0], 0.13, 16, 0.5)

anchors = a.anchors

print(anchors.shape)

I = cv2.imread(sys.argv[1])

print(I.shape)

merge_boxes = []
for i, f_box in enumerate(anchors):
    if i % 2 == 0:
       for j, box in enumerate(f_box):
           if j % 2 == 0:
              print(box)
              merge_boxes.append([int(i) for i in box[0]])

print(len(merge_boxes))
I = drawBoxes(I, merge_boxes)

cv2.imwrite("boxes_drawn.jpg", I)

print("anchor length:", len(anchors), len(anchors[0]))
