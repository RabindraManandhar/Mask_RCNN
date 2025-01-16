import numpy as np
import cv2


class GrabCutProcessor:
    """
    A utility class for GrabCut refinement of segmentation masks.
    """

    @staticmethod
    def convertCoords(coords):
        """
        Converts ROI coordinates from (y1, x1, y2, x2) to (x1, y1, width, height).

        Args:
            coords (tuple): Bounding box coordinates in (y1, x1, y2, x2) format.

        Returns:
            tuple: Converted coordinates in (x1, y1, width, height) format.
        """
        y1, x1, y2, x2 = coords
        width, height = int(x2 - x1), int(y2 - y1)
        return (int(x1), int(y1), width, height)

    @staticmethod
    def separateEntities(r):
        """
        Extracts masks and converted bounding boxes from Mask R-CNN results.

        Args:
            r (dict): Results from Mask R-CNN inference.

        Returns:
            tuple: List of masks and list of bounding boxes.
        """
        if len(r["class_ids"]) == 0:
            return [], []

        masks = [
            r["masks"][:, :, i].astype("uint8") for i in range(len(r["class_ids"]))
        ]
        rects = [GrabCutProcessor.convertCoords(roi) for roi in r["rois"]]
        return masks, rects

    @staticmethod
    def applyGrabCut(image, mask, rect, iters=5):
        """
        Refines a segmentation mask using the GrabCut algorithm.

        Args:
            image (numpy.ndarray): Original image (H, W, 3).
            mask (numpy.ndarray): Initial binary mask (H, W).
            rect (tuple): Bounding box in (x1, y1, width, height) format.
            iters (int): Number of iterations for GrabCut.

        Returns:
            dict: Masks for definite and probable foreground/background.
        """
        # Ensure the image is a 3-channel uint8 image
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel RGB image, but got shape {image.shape}"
            )
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Ensure the mask is of type uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Validate rect format
        if (
            not isinstance(rect, tuple)
            or len(rect) != 4
            or not all(isinstance(x, int) for x in rect)
        ):
            raise ValueError(
                f"Invalid rect format: {rect}. Expected (x1, y1, width, height) as integers."
            )

        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")

        # Apply GrabCut
        mask_grab, _, _ = cv2.grabCut(
            image,
            mask,
            rect,
            bgModel,
            fgModel,
            iterCount=iters,
            mode=cv2.GC_INIT_WITH_RECT,
        )

        # Generate masks for different classes
        values = {
            "Definite Background": cv2.GC_BGD,
            "Probable Background": cv2.GC_PR_BGD,
            "Definite Foreground": cv2.GC_FGD,
            "Probable Foreground": cv2.GC_PR_FGD,
        }
        valueMasks = {
            name: (mask_grab == value).astype("uint8") for name, value in values.items()
        }

        return valueMasks
