import mrcnn.model as modellib
import numpy as np

from samples.shapes.shapes import ShapesConfig
from samples.shapes.shapes import ShapesDataset
from fixtures import model_data

from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite


def test_training(tmpdir, model_data):
    config = ShapesConfig()
    # Training dataset
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShapesDataset()
    dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=tmpdir)
    model.load_weights(model_data, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


class FixShapesDataset(ShapesDataset):

    def random_image(self, height, width):
        """ Return fixed image for testing
        """
        # Pick random background color
        bg_color = np.array([202, 3, 25])
        shapes = [('square', (28, 39, 158), (99, 65, 21)), ('circle', (158, 237, 77), (57, 26, 29))]
        return bg_color, shapes


def x_test_load_image_gt(tmpdir, model_data):
    config = ShapesConfig()
    dataset_train = FixShapesDataset()
    dataset_train.load_shapes(5, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    ds = modellib.DataSequence(dataset_train, config)
    image, image_meta, class_ids, bbox, mask = ds.load_image_gt(0, None)
    assert (image[0][0] == [202, 3, 25]).all()
    assert (image_meta == [0, 128, 128, 3, 128, 128, 3, 0, 0, 128, 128, 1, 1, 1, 1, 1]).all()
    assert (class_ids == [1, 2]).all()
    assert (bbox == [[44, 78, 87, 121], [0, 28, 56, 87]]).all()
    assert (mask[0][0] == [0, 0]).all()


def x_test_data_sequence():
    config = ShapesConfig()
    dataset = FixShapesDataset()
    dataset.load_shapes(1, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset.prepare()
    dg = iter_sequence_infinite(modellib.DataSequence(dataset, config))
    inputs, output = next(dg)

    assert output == []
    batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs

    assert (batch_gt_boxes[0][:3] == [[44, 78, 87, 121], [0, 28, 56, 87], [0, 0, 0, 0]]).all()
    assert (batch_image_meta[0] == [0, 128, 128, 3, 128, 128, 3, 0, 0, 128, 128, 1, 1, 1, 1, 1]).all()


