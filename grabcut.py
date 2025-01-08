import numpy as np
import cv2


class GrabCutProcessor:
    @staticmethod
    def apply_grabcut(image, mask):
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        mask[mask > 0] = cv2.GC_PR_FGD
        mask[mask == 0] = cv2.GC_BGD

        refined_mask = cv2.grabCut(
            image, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK
        )[0]
        final_mask = np.where((refined_mask == 2) | (refined_mask == 0), 0, 1).astype(
            "uint8"
        )
        return final_mask
